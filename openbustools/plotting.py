import os

import contextily as cx
import geopandas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly
import seaborn as sns
import statsmodels.api as sm

from openbustools import standardfeeds

HEIGHT=6
WIDTH=8
HEIGHT_WIDE=3
ASPECT_WIDE=4
HEIGHT_SQ=12
WIDTH_SQ=12
PLOT_FOLDER="../plots/"


def formatted_lineplot(plot_df, x_var, y_var, title_text="throwaway"):
    fig, axes = plt.subplots(1,1)
    fig.set_figheight(HEIGHT)
    fig.set_figwidth(WIDTH)
    sns.lineplot(plot_df, x=x_var, y=y_var, ax=axes)
    fig.suptitle(title_text, fontsize=16)
    fig.tight_layout()
    plt.savefig(f"{PLOT_FOLDER}{title_text}.eps", format='eps', dpi=600, bbox_inches='tight')
    plt.savefig(f"{PLOT_FOLDER}{title_text}.png", format='png', dpi=600, bbox_inches='tight')
    return None


def formatted_rel_lineplot(plot_df, x_var, y_var, rel_var, title_text="throwaway", xlim=None, ylim=None):
    g = sns.relplot(plot_df, x=x_var, y=y_var, row=rel_var, kind='line', height=HEIGHT_WIDE, aspect=ASPECT_WIDE)
    if xlim:
        g.set(xlim=xlim)
    if ylim:
        g.set(ylim=ylim)
    plt.savefig(f"{PLOT_FOLDER}{title_text}.eps", format='eps', dpi=600, bbox_inches='tight')
    plt.savefig(f"{PLOT_FOLDER}{title_text}.png", format='png', dpi=600, bbox_inches='tight')
    return None


def formatted_basemap_scatterplot(plot_gdf, title_text="throwaway"):
    fig, axes = plt.subplots(1,1)
    fig.set_figheight(HEIGHT_SQ)
    fig.set_figwidth(WIDTH_SQ)
    plot_gdf.plot(ax=axes, markersize=5)
    plot_gdf.iloc[0:1].plot(ax=axes, markersize=100, color='green', marker='x')
    plot_gdf.iloc[-1:].plot(ax=axes, markersize=100, color='red', marker='x')
    cx.add_basemap(ax=axes, crs=plot_gdf.crs.to_string(), alpha=0.6, source=cx.providers.MapBox(accessToken=os.getenv(key="MAPBOX_TOKEN")))
    fig.tight_layout()
    plt.savefig(f"{PLOT_FOLDER}{title_text}.eps", format='eps', dpi=600, bbox_inches='tight')
    plt.savefig(f"{PLOT_FOLDER}{title_text}.png", format='png', dpi=600, bbox_inches='tight')
    return None


def lowess_with_confidence_bounds(x, y, eval_x, N=200, conf_interval=0.95, lowess_kw=None):
    """Perform Lowess regression and determine a confidence interval by bootstrap resampling."""
    # Lowess smoothing
    smoothed = sm.nonparametric.lowess(exog=x, endog=y, xvals=eval_x, **lowess_kw)

    # Perform bootstrap resamplings of the data
    # and evaluate the smoothing at a fixed set of points
    smoothed_values = np.empty((N, len(eval_x)))
    for i in range(N):
        sample = np.random.choice(len(x), len(x), replace=True)
        sampled_x = x[sample]
        sampled_y = y[sample]

        smoothed_values[i] = sm.nonparametric.lowess(
            exog=sampled_x, endog=sampled_y, xvals=eval_x, **lowess_kw
        )

    # Get the confidence interval
    sorted_values = np.sort(smoothed_values, axis=0)
    bound = int(N * (1 - conf_interval) / 2)
    bottom = sorted_values[bound - 1]
    top = sorted_values[-bound]

    return smoothed, bottom, top


# def formatted_barplot(plot_df):
#     fig, axes = plt.subplots(1,1)
#     fig.set_figheight(HEIGHT)
#     fig.set_figwidth(WIDTH)
#     sns.barplot(plot_df, x=x_var, y=y_var, ax=axes)
#     axes.set_xlim([0, 0.5])
#     fig.suptitle('KCM Model Performance', fontsize=16)
#     fig.tight_layout()
#     plt.savefig("{PLOT_FOLDER}model_performances_kcm.eps", format='eps', dpi=600, bbox_inches='tight')
#     plt.savefig("{PLOT_FOLDER}model_performances_kcm.png", format='png', dpi=600, bbox_inches='tight')
#     return None


def plot_gtfsrt_trip(ax, trace_df, epsg, gtfs_folder):
    """Plot a single real-time bus trajectory on a map.
    """
    # Plot trip stops from GTFS
    trace_date = trace_df['file'].iloc[0]
    trip_id = trace_df['trip_id'].iloc[0]
    file_to_gtfs_map = standardfeeds.get_best_gtfs_lookup(trace_df, gtfs_folder)
    gtfs_data = standardfeeds.merge_gtfs_files(f"{gtfs_folder}{file_to_gtfs_map[trace_date]}/", epsg, [0,0])
    to_plot_gtfs = gtfs_data[gtfs_data['trip_id']==trip_id]
    to_plot_gtfs = geopandas.GeoDataFrame(to_plot_gtfs, geometry=geopandas.points_from_xy(to_plot_gtfs.stop_x, to_plot_gtfs.stop_y), crs=f"EPSG:{epsg}")
    to_plot_gtfs.plot(ax=ax, marker='x', color='lightblue', markersize=10)
    # Plot observations
    to_plot = trace_df.copy()
    to_plot = geopandas.GeoDataFrame(to_plot, geometry=geopandas.points_from_xy(to_plot.x, to_plot.y), crs=f"EPSG:{epsg}")
    to_plot_stop = trace_df.iloc[-1:,:].copy()
    to_plot_stop = geopandas.GeoDataFrame(to_plot_stop, geometry=geopandas.points_from_xy(to_plot_stop.stop_x, to_plot_stop.stop_y), crs=f"EPSG:{epsg}")
    to_plot.plot(ax=ax, marker='.', color='purple', markersize=20)
    # Plot first/last observations
    to_plot.iloc[:1,:].plot(ax=ax, marker='*', color='green', markersize=40)
    to_plot.iloc[-1:,:].plot(ax=ax, marker='*', color='red', markersize=40)
    # Plot closest stop to final observation
    to_plot_stop.plot(ax=ax, marker='x', color='blue', markersize=20)
    # Add custom legend
    ax.legend(["Scheduled Trip Stops","Shingle Observations","Shingle Start","Shingle End", "Closest Stop"], loc="upper right")
    return None