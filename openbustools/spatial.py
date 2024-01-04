import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import rasterio
from rasterio.sample import sample_gen
from rasterio.warp import calculate_default_transform, reproject, Resampling
from scipy.spatial import KDTree

from openbustools.drivecycle import trajectory


def resample_to_len(ary, new_len, xp=None):
    """
    Resamples an array to a specified length using linear interpolation.

    Parameters:
        ary (ndarray): The input array to be resampled.
        new_len (int): The desired length of the resampled array.
        xp (ndarray, optional): The x-coordinates of the input array. If not provided, it is assumed to be a linearly spaced array.

    Returns:
        ndarray: The resampled array.
    """
    if xp is None:
        xp = np.arange(0, ary.shape[0], 1)
    eval_x = np.linspace(np.min(xp), np.max(xp), new_len)
    # If 1D, interp directly, else apply along axis
    if ary.ndim == 1:
        res = np.interp(eval_x, xp, ary)
    else:
        res = np.apply_along_axis(lambda y: np.interp(eval_x, xp, y), 0, ary)
    return res


def calculate_gps_metrics(gdf, lon_col, lat_col):
    """Calculate metrics between consecutive gps locations.

    Args:
        gdf (GeoDataFrame): A GeoDataFrame containing GPS locations.
        lon_col (str): The name of the column containing longitude values.
        lat_col (str): The name of the column containing latitude values.

    Returns:
        tuple: A tuple containing the distance and forward azimuth between consecutive GPS locations.
    """
    # The first row of each trip will overlap previous and should be removed
    gdf_shifted = gdf.shift()
    geodesic = pyproj.Geod(ellps='WGS84')
    # Fwd azimuth is CW deg from N==0, pointed towards the latter point
    f_azm, b_azm, distance = geodesic.inv(gdf_shifted[lon_col], gdf_shifted[lat_col], gdf[lon_col], gdf[lat_col])
    return distance, f_azm


def get_point_distances(points, query_points):
    """
    Calculates the distances between a set of points and a set of query points using a KDTree.

    Parameters:
        points (array-like): The coordinates of the points in the form of a 2D array-like object.
        query_points (array-like): The coordinates of the query points in the form of a 2D array-like object.

    Returns:
        dists (array-like): The distances between the points and the query points.
        idxs (array-like): The indices of the nearest points in the points array for each query point.
    """
    tree = KDTree(points)
    dists, idxs = tree.query(query_points)
    return dists, idxs


def reproject_raster(in_file, out_file, dst_crs):
    """
    Reprojects a raster file to a new coordinate reference system (CRS).

    Args:
        in_file (str): The path to the input raster file.
        out_file (str): The desired path to the output raster file.
        dst_crs (int): The EPSG code of the destination CRS.

    Returns:
        None
    """
    dst_crs = f"EPSG:{dst_crs}"
    with rasterio.open(in_file) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })
        with rasterio.open(out_file, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)
        return None


def sample_raster(points, dem_file):
    """
    Sample a raster at a set of points, currently used for elevation.

    Parameters:
        points (list): List of (x, y) coordinate tuples representing the points to sample.
        dem_file (str): Filepath of the raster file to sample.

    Returns:
        numpy.ndarray: Array of sampled values from the raster.
    """
    with rasterio.open(dem_file, "r") as src:
        z = np.array(list(sample_gen(src, points))).flatten()
    return z


def shingle(trace_df, min_break, max_break, min_len, max_len, **kwargs):
    """
    Split a dataframe into random even chunks, with random resampled len.

    Parameters:
        trace_df (DataFrame): The input dataframe to be split into shingles.
        min_break (int): The minimum number of breaks to be applied to each shingle.
        max_break (int): The maximum number of breaks to be applied to each shingle.
        min_len (int): The minimum length of each shingle.
        max_len (int): The maximum length of each shingle.
        **kwargs: Additional keyword arguments.

    Returns:
        DataFrame: The dataframe with shingle IDs assigned to each row.
    """
    # Set initial id based on unique file and trip id
    shingle_lens = trace_df.groupby(['file','trip_id']).count()['lat'].values
    shingle_ids = [id.repeat(grouplen) for id, grouplen in zip(np.arange(len(shingle_lens)), shingle_lens)]
    # Break each shingle into a random number of smaller shingles
    shingle_n_chunks = np.random.randint(min_break, max_break, len(shingle_ids))
    shingle_ids = [np.array_split(shingle, n_chunks) for shingle, n_chunks in zip(shingle_ids, shingle_n_chunks)]
    # Start shingle indexing from 0
    shingle_id_counter = 0
    all_shingles = []
    for i in range(len(shingle_ids)):
        for j in range(len(shingle_ids[i])):
            shingle_ids[i][j][:] = shingle_id_counter
            all_shingles.append(shingle_ids[i][j])
            shingle_id_counter += 1
    shingle_ids = np.concatenate(all_shingles)
    shingled_trace_df = trace_df.copy()
    shingled_trace_df['shingle_id'] = shingle_ids
    # Now resample each shingle to a random length between min and max
    sids, sidxs = np.unique(shingled_trace_df['shingle_id'], return_index=True)
    shingles = np.split(shingled_trace_df[['locationtime','lon','lat']].to_numpy(), sidxs[1:], axis=0)
    resample_lens = np.random.randint(min_len, max_len, len(shingles))
    resampled= []
    for shingle_data, resample_len in zip(shingles, resample_lens):
        resampled.append(resample_to_len(shingle_data, resample_len))
    resampled_sids = np.repeat(sids, resample_lens).astype(int)
    # Can't interpolate categoricals, so rejoin them after resampling
    cat_lookup = shingled_trace_df[['file','trip_id','shingle_id']].drop_duplicates()
    shingled_trace_df = pd.DataFrame(np.concatenate(resampled), columns=['locationtime','lon','lat'])
    shingled_trace_df['shingle_id'] = resampled_sids
    shingled_trace_df = pd.merge(shingled_trace_df, cat_lookup, on='shingle_id').sort_values(['shingle_id','locationtime'])
    return shingled_trace_df


def divide_fwd_back_fill(arr1, arr2):
    """
    Divide two arrays element-wise, while handling division by zero.
    Forward fill the resulting array to replace NaN values.

    Args:
        arr1 (numpy.ndarray): The numerator array.
        arr2 (numpy.ndarray): The denominator array.

    Returns:
        numpy.ndarray: The resulting array after division and forward filling.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        res = arr1 / arr2
    res[res==-np.inf] = np.nan
    res[res==np.inf] = np.nan
    res = pd.Series(res).ffill()
    res = res.bfill().to_numpy()
    return res


def create_bounded_gdf(data, lon_col, lat_col, epsg, coord_ref_center, grid_bounds, dem_file):
    """
    Create a geodataframe matching a grid and network.

    Args:
        data (pandas.DataFrame): Input data containing lon_col and lat_col columns.
        lon_col (str): Name of the column containing longitude values.
        lat_col (str): Name of the column containing latitude values.
        epsg (int): EPSG code specifying the coordinate reference system.
        coord_ref_center (tuple): Tuple containing the coordinates of the system reference center.
        grid_bounds (list): List containing the bounds of the grid [minx, miny, maxx, maxy].
        dem_file (str): File path to the digital elevation model.

    Returns:
        gpd.GeoDataFrame: Geodataframe containing the bounded data with additional columns.
    """
    _, coord_ref_center_xy = project_bounds(grid_bounds, coord_ref_center, epsg)
    data = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data[lon_col].to_numpy(), data[lat_col].to_numpy()), crs="EPSG:4326")
    data = data.cx[grid_bounds[0]:grid_bounds[2], grid_bounds[1]:grid_bounds[3]].copy().to_crs(f"EPSG:{epsg}")
    data['x'] = data.geometry.x
    data['y'] = data.geometry.y
    data['x_cent'] = data['x'] - coord_ref_center_xy[0]
    data['y_cent'] = data['y'] - coord_ref_center_xy[1]
    data['elev_m'] = sample_raster(data[['x','y']].values, dem_file)
    return data


def project_bounds(grid_bounds, coord_ref_center, epsg):
    """
    Projects the grid bounds and coordinate reference center to xy coordinates in integer meters.

    Args:
        grid_bounds (list): A list of four values representing the grid bounds in the format [min_x, min_y, max_x, max_y].
        coord_ref_center (list): A list of two values representing the coordinate reference center in the format [x, y].
        epsg (int): The EPSG code specifying the coordinate reference system.

    Returns:
        tuple: A tuple containing the projected grid bounds and coordinate reference center in the format (grid_bounds_prj, coord_ref_center_prj).
               grid_bounds_prj (list): A list of four integer values representing the projected grid bounds in the format [min_x, min_y, max_x, max_y].
               coord_ref_center_prj (list): A list of two integer values representing the projected coordinate reference center in the format [x, y].
    """
    grid_bounds_gdf = gpd.GeoDataFrame({'geometry': gpd.points_from_xy([grid_bounds[0], grid_bounds[2]], [grid_bounds[1], grid_bounds[3]])}, crs="EPSG:4326")
    grid_bounds_gdf = grid_bounds_gdf.to_crs(f"EPSG:{epsg}")
    grid_bounds_prj = [int(grid_bounds_gdf.geometry[0].x), int(grid_bounds_gdf.geometry[0].y), int(grid_bounds_gdf.geometry[1].x), int(grid_bounds_gdf.geometry[1].y)]
    coord_ref_center_gdf = gpd.GeoDataFrame({'geometry': gpd.points_from_xy([coord_ref_center[0]], [coord_ref_center[1]])}, crs="EPSG:4326")
    coord_ref_center_gdf = coord_ref_center_gdf.to_crs(f"EPSG:{epsg}")
    coord_ref_center_prj = [int(coord_ref_center_gdf.geometry[0].x), int(coord_ref_center_gdf.geometry[0].y)]
    return (grid_bounds_prj, coord_ref_center_prj)


# def poly_locate_line_points(poly_geo, line_geo):
#     if line_geo.intersects(poly_geo):
#         intersection = line_geo.intersection(poly_geo)
#         first_point = intersection.boundary.geoms[0]
#         last_point = intersection.boundary.geoms[-1]
#         res = (line_geo.project(first_point)*111111, line_geo.project(last_point)*111111)
#         return res
#     else:
#         return None
# intersected_trips['line_locs'] = intersected_trips.apply(lambda x: poly_locate_line_points(x['geometry_x'], x['geometry_y']), axis=1)


# def create_grid_of_shingles(point_resolution, grid_bounds, coord_ref_center):
#     # Create grid of coordinates to use as inference inputs
#     x_val = np.linspace(grid_bounds[0], grid_bounds[2], point_resolution)
#     y_val = np.linspace(grid_bounds[1], grid_bounds[3], point_resolution)
#     X,Y = np.meshgrid(x_val,y_val)
#     x_spacing = (grid_bounds[2] - grid_bounds[0]) / point_resolution
#     y_spacing = (grid_bounds[3] - grid_bounds[1]) / point_resolution
#     # Create shingle samples with only geometric variables
#     shingles = []
#     # All horizontal shingles W-E and E-W
#     for i in range(X.shape[0]):
#         curr_shingle = {}
#         curr_shingle["x"] = list(X[i,:])
#         curr_shingle["y"] = list(Y[i,:])
#         curr_shingle["dist_calc_km"] = [x_spacing/1000 for x in curr_shingle["x"]]
#         curr_shingle["bearing"] = [0 for x in curr_shingle["x"]]
#         shingles.append(curr_shingle)
#         rev_shingle = {}
#         rev_shingle["x"] = list(np.flip(X[i,:]))
#         rev_shingle["y"] = list(np.flip(Y[i,:]))
#         rev_shingle["dist_calc_km"] = [x_spacing/1000 for x in rev_shingle["x"]]
#         rev_shingle["bearing"] = [180 for x in rev_shingle["x"]]
#         shingles.append(rev_shingle)
#     # All vertical shingles N-S and S-N
#     for j in range(X.shape[1]):
#         curr_shingle = {}
#         curr_shingle["x"] = list(X[:,j])
#         curr_shingle["y"] = list(Y[:,j])
#         curr_shingle["dist_calc_km"] = [y_spacing/1000 for x in curr_shingle["x"]]
#         curr_shingle["bearing"] = [-90 for x in curr_shingle["x"]]
#         shingles.append(curr_shingle)
#         rev_shingle = {}
#         rev_shingle["x"] = list(np.flip(X[:,j]))
#         rev_shingle["y"] = list(np.flip(Y[:,j]))
#         rev_shingle["dist_calc_km"] = [y_spacing/1000 for x in rev_shingle["x"]]
#         rev_shingle["bearing"] = [90 for x in rev_shingle["x"]]
#         shingles.append(rev_shingle)
#     # Add dummy and calculated variables
#     shingle_id = 0
#     for curr_shingle in shingles:
#         curr_shingle['lat'] = curr_shingle['y']
#         curr_shingle['lon'] = curr_shingle['x']
#         curr_shingle['shingle_id'] = [shingle_id for x in curr_shingle['lon']]
#         curr_shingle['timeID'] = [60*9 for x in curr_shingle['lon']]
#         curr_shingle['weekID'] = [3 for x in curr_shingle['lon']]
#         curr_shingle['x_cent'] = [x - coord_ref_center[0] for x in curr_shingle['x']]
#         curr_shingle['y_cent'] = [y - coord_ref_center[1] for y in curr_shingle['y']]
#         curr_shingle['locationtime'] = [1 for x in curr_shingle['lon']]
#         curr_shingle['time_calc_s'] = [1 for x in curr_shingle['lon']]
#         curr_shingle['dist_cumulative_km'] = [1 for x in curr_shingle['lon']]
#         curr_shingle['time_cumulative_s'] = [1 for x in curr_shingle['lon']]
#         curr_shingle['speed_m_s'] = [1 for x in curr_shingle['lon']]
#         curr_shingle['timeID_s'] = [1 for x in curr_shingle['lon']]
#         shingle_id += 1
#     # Convert to tabular format, create shingle lookup
#     res_dict = {}
#     for cname in shingles[0].keys():
#         res_dict[cname] = []
#         vals = np.array([s[cname] for s in shingles]).flatten()
#         res_dict[cname].extend(vals)
#     res = pd.DataFrame(res_dict)
#     return res