import pickle

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import rasterio
from rasterio.sample import sample_gen
from rasterio.warp import calculate_default_transform, reproject, Resampling
from scipy.spatial import KDTree
import scipy
import shapely
from srai.neighbourhoods.h3_neighbourhood import H3Neighbourhood
from srai.regionalizers import H3Regionalizer


def manhattan_distance(x1, y1, x2, y2):
    """
    Calculate the Manhattan distance between two points in a 2D plane.
    
    Args:
        x1 (float): The x-coordinate of the first point.
        y1 (float): The y-coordinate of the first point.
        x2 (float): The x-coordinate of the second point.
        y2 (float): The y-coordinate of the second point.
    
    Returns:
        float: The Manhattan distance between the two points.
    """
    return abs(x1 - x2) + abs(y1 - y2)


def make_polygon(bbox):
    """
    Create a polygon from a bounding box.

    Args:
        bbox (tuple): A tuple containing the coordinates of the bounding box in the format (minx, miny, maxx, maxy).

    Returns:
        shapely.geometry.Polygon: A polygon representing the bounding box.
    """
    polygon = shapely.geometry.Polygon([
        (bbox[0], bbox[1]),
        (bbox[0], bbox[3]),
        (bbox[2], bbox[3]),
        (bbox[2], bbox[1]),
        (bbox[0], bbox[1])
    ])
    gdf = gpd.GeoDataFrame({'geometry': [polygon]}, crs='epsg:4326')
    return gdf


def calculate_gps_metrics(gdf, lon_col, lat_col, time_col=None):
    """Calculate metrics between consecutive gps locations.

    Args:
        gdf (GeoDataFrame): A GeoDataFrame containing GPS locations.
        lon_col (str): The name of the column containing longitude values.
        lat_col (str): The name of the column containing latitude values.

    Returns:
        tuple: A tuple containing the distance and forward azimuth between consecutive GPS locations.
    """
    # The first row of each trip will overlap previous trip's last row
    gdf_shifted = gdf.shift(1)
    geodesic = pyproj.Geod(ellps='WGS84')
    # Fwd azimuth is CW deg from N==0, pointed towards the latter point
    f_azm, b_azm, distance = geodesic.inv(gdf_shifted[lon_col], gdf_shifted[lat_col], gdf[lon_col], gdf[lat_col])
    # Can optionally calculate time between points if the points were observed
    if time_col is not None:
        time_diff = gdf[time_col] - gdf_shifted[time_col]
        return distance, f_azm, time_diff
    else:
        return distance, f_azm


def get_point_distances(points, query_points):
    """
    Calculates the distances between a set of points and a set of query points using a KDTree.

    Args:
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

    Args:
        points (list): List of (x, y) coordinate tuples representing the points to sample.
        dem_file (str): Filepath of the raster file to sample.

    Returns:
        numpy.ndarray: Array of sampled values from the raster.
    """
    with rasterio.open(dem_file, "r") as src:
        z = np.array(list(sample_gen(src, points))).flatten()
    return z


def divide_fwd_back_fill(arr1, arr2):
    """
    Divide two arrays element-wise, while handling division by zero.
    Forward fill then back fill the resulting array to replace NaN values.

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
    res = pd.Series(res).ffill().bfill().to_numpy()
    return res


def create_bounded_gdf(data, lon_col, lat_col, epsg, coord_ref_center, grid_bounds, dem_file):
    """
    Create a geodataframe of tracks matching a grid and network.

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


def create_regions(grid_bounds, embeddings_folder):
    """
    Create regions based on the given grid bounds and save them to the spatial folder.

    Args:
        grid_bounds (list): A list of four coordinates representing the bounding box of the grid.
        embeddings_folder (str): The path to the embeddings_folder folder where the regions will be saved.
    """
    geo = shapely.Polygon((
        (grid_bounds[0], grid_bounds[1]),
        (grid_bounds[0], grid_bounds[3]),
        (grid_bounds[2], grid_bounds[3]),
        (grid_bounds[2], grid_bounds[1]),
        (grid_bounds[0], grid_bounds[1])
    ))
    # Define area for embeddings, get H3 regions
    area = gpd.GeoDataFrame({'region_id': [str(embeddings_folder)], 'geometry': [geo]}, crs='epsg:4326')
    area.set_index('region_id', inplace=True)
    regionalizer = H3Regionalizer(resolution=8)
    regions = regionalizer.transform(area)
    neighbourhood = H3Neighbourhood(regions_gdf=regions)
    # Save
    embeddings_folder.mkdir(parents=True, exist_ok=True)
    area.to_pickle(embeddings_folder / "area.pkl")
    regions.to_pickle(embeddings_folder / "regions.pkl")
    with open(embeddings_folder / 'neighbourhood.pkl', 'wb') as f:
        pickle.dump(neighbourhood, f)
    return (area, regions, neighbourhood)


def load_regions(embeddings_folder):
    """
    Load regions from the specified spatial folder.

    Args:
        spatial_folder (str): The path to the spatial folder where the regions are saved.

    Returns:
        tuple: A tuple containing the area, regions, and neighbourhood.
    """
    area = pd.read_pickle(embeddings_folder / "area.pkl")
    regions = pd.read_pickle(embeddings_folder / "regions.pkl")
    with open(embeddings_folder / 'neighbourhood.pkl', 'rb') as f:
        neighbourhood = pickle.load(f)
    return (area, regions, neighbourhood)


def apply_sg_filter(sequence, window_len_factor=.03, polyorder=5, clip_min=None, clip_max=None):
    # window len <= sequence len
    # polyorder < window len
    window_len = int(len(sequence) * window_len_factor)
    window_len = max([window_len, 3]) # Based on factor* len, but no less than 3 samples
    polyorder = min([polyorder, window_len-1]) # Based on input, but no greater than window length - 1
    filtered_sequence = scipy.signal.savgol_filter(sequence, window_length=window_len, polyorder=polyorder)
    if clip_min is not None:
        filtered_sequence = np.clip(filtered_sequence, a_min=clip_min, a_max=None)
    if clip_max is not None:
        filtered_sequence = np.clip(filtered_sequence, a_min=None, a_max=clip_max)
    return filtered_sequence


def apply_peak_filter(arr, scalar=2.0, window_len=3, clip_min=None, clip_max=None):
    assert window_len % 2 == 1, "Window length must be odd"
    # Create rolling windows
    rolling_windows = np.lib.stride_tricks.sliding_window_view(arr, window_len)
    # Pad the start and end to match input length
    num_padding = window_len // 2
    rolling_windows = np.pad(rolling_windows, pad_width=((num_padding,num_padding), (0,0)), mode='edge')
    # Apply peak filter to each window
    means = np.mean(rolling_windows, axis=1)
    peaked_windows = rolling_windows + (rolling_windows - means[:, np.newaxis]) * scalar
    # Get the middle value from each window
    peaked_windows = peaked_windows[:,num_padding]
    peaked_windows = np.clip(peaked_windows, clip_min, clip_max)
    return peaked_windows


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
    return c * r


def bbox_area(lon1, lat1, lon2, lat2):
    """
    Calculate the area of a bounding box given by four coordinates
    """
    # Calculate the side lengths
    side_a = haversine(lon1, lat1, lon2, lat1)
    side_b = haversine(lon1, lat1, lon1, lat2)

    # Calculate the area
    area = side_a * side_b
    return area