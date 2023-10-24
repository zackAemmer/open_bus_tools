import geopandas as gpd
import numpy as np
import pandas as pd


def calculate_speed(gdf, time_col):
    """Calculate speed between consecutive gps locations."""
    gdf_shifted = gdf.shift()
    consecutive_dist_m = gdf.distance(gdf_shifted, align=False)
    consecutive_time_s = gdf[time_col] - gdf_shifted[time_col]
    consecutive_speed_s = [d/t for (d,t) in zip(consecutive_dist_m, consecutive_time_s)]
    return consecutive_dist_m, consecutive_time_s, consecutive_speed_s