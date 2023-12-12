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
    if xp is None:
        xp = np.arange(0, ary.shape[0], 1)
    eval_x = np.linspace(np.min(xp), np.max(xp), new_len)
    if ary.ndim == 1:
        res = np.interp(eval_x, xp, ary)
    else:
        res = np.apply_along_axis(lambda y: np.interp(eval_x, xp, y), 0, ary)
    return res


def calculate_gps_metrics(gdf, lon_col, lat_col):
    """Calculate metrics between consecutive gps locations."""
    # Ensure no repeated time obs; can still have same time trip end/next start
    gdf_shifted = gdf.shift()
    geodesic = pyproj.Geod(ellps='WGS84')
    # Fwd azimuth is CW deg from N==0, pointed towards the latter point
    f_azm, b_azm, distance = geodesic.inv(gdf_shifted[lon_col], gdf_shifted[lat_col], gdf[lon_col], gdf[lat_col])
    # The first row of each trip will overlap previous and should be removed
    return distance, f_azm


def get_point_distances(points, query_points):
    tree = KDTree(points)
    dists, idxs = tree.query(query_points)
    return dists, idxs


def reproject_raster(in_file, out_file, dst_crs):
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
    """Sample a raster at a set of points, used for elevation."""
    with rasterio.open(dem_file, "r") as src:
        z = np.array(list(sample_gen(src, points))).flatten()
    return z


def shingle(trace_df, min_break, max_break, min_len, max_len, **kwargs):
    """Split a df into even chunks randomly between min and max length."""
    # Set initial id based on unique file and trip id
    shingle_lens = trace_df.groupby(['file','trip_id']).count()['lat'].values
    shingle_ids = [id.repeat(grouplen) for id, grouplen in zip(np.arange(len(shingle_lens)), shingle_lens)]
    # Break each shingle into a random number of smaller shingles
    shingle_n_chunks = np.random.randint(min_break, max_break, len(shingle_ids))
    shingle_ids = [np.array_split(shingle, n_chunks) for shingle, n_chunks in zip(shingle_ids, shingle_n_chunks)]
    # Start from 0
    shingle_id_counter = 0
    all_shingles = []
    for i in range(len(shingle_ids)):
        for j in range(len(shingle_ids[i])):
            shingle_ids[i][j][:] = shingle_id_counter
            all_shingles.append(shingle_ids[i][j])
            shingle_id_counter += 1
    shingle_ids = np.concatenate(all_shingles)
    z = trace_df.copy()
    z['shingle_id'] = shingle_ids
    # Resample each shingle to a random length between min and max
    sids, sidxs = np.unique(z['shingle_id'], return_index=True)
    shingles = np.split(z[['locationtime','lon','lat']].to_numpy(), sidxs[1:], axis=0)
    resample_lens = np.random.randint(min_len, max_len, len(shingles))
    resampled= []
    for shingle_data, resample_len in zip(shingles, resample_lens):
        resampled.append(resample_to_len(shingle_data, resample_len))
    resampled_sids = np.repeat(sids, resample_lens).astype(int)
    # Can't interpolate categoricals, so rejoin them after resampling
    cat_lookup = z[['file','trip_id','shingle_id']].drop_duplicates()
    z = pd.DataFrame(np.concatenate(resampled), columns=['locationtime','lon','lat'])
    z['shingle_id'] = resampled_sids
    z = pd.merge(z, cat_lookup, on='shingle_id').sort_values(['shingle_id','locationtime'])
    return z


def create_bounded_gdf(data, lon_col, lat_col, epsg, coord_ref_center, grid_bounds, dem_file):
    """Create a geodataframe matching a grid and network."""
    data = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data[lon_col].to_numpy(), data[lat_col].to_numpy()), crs="EPSG:4326").to_crs(f"EPSG:{epsg}")
    data = data.cx[grid_bounds[0]:grid_bounds[2], grid_bounds[1]:grid_bounds[3]].copy()
    data['x'] = data.geometry.x
    data['y'] = data.geometry.y
    data['x_cent'] = data['x'] - coord_ref_center[0]
    data['y_cent'] = data['y'] - coord_ref_center[1]
    data['elev_m'] = sample_raster(data[['x','y']].values, dem_file)
    return data


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