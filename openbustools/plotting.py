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
    """Plot a single sequence of points w/marked start and end."""
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
    return fig, axes


def formatted_shingle_scatterplot(plot_gdf, title_text="throwaway"):
    """Plot realtime for a single day/trip combo across its shingles."""
    fig, axes = plt.subplots(1,1)
    fig.set_figheight(HEIGHT_SQ)
    fig.set_figwidth(WIDTH_SQ)
    plot_gdf.plot(ax=axes, markersize=5, column='shingle_id')
    plot_gdf.iloc[0:1].plot(ax=axes, markersize=100, color='green', marker='x')
    plot_gdf.iloc[-1:].plot(ax=axes, markersize=100, color='red', marker='x')
    cx.add_basemap(ax=axes, crs=plot_gdf.crs.to_string(), alpha=0.4, source=cx.providers.MapBox(accessToken=os.getenv(key="MAPBOX_TOKEN")))
    fig.tight_layout()
    plt.savefig(f"{PLOT_FOLDER}{title_text}.eps", format='eps', dpi=600, bbox_inches='tight')
    plt.savefig(f"{PLOT_FOLDER}{title_text}.png", format='png', dpi=600, bbox_inches='tight')
    return fig, axes


def formatted_feature_distributions_histplot(plot_df, title_text="throwaway"):
    """Plot distributions of labels and key features."""
    fig, axes = plt.subplots(2,2)
    axes = axes.flatten()
    fig.set_figheight(HEIGHT)
    fig.set_figwidth(WIDTH)
    sample_groups = plot_df.groupby('shingle_id')
    metric = sample_groups.count()['locationtime']
    sns.histplot(metric, ax=axes[0])
    axes[0].set_xlabel("Observations (n)")
    axes[0].set_xlim(0,70)
    metric = sample_groups.last()['cumul_dist_km']
    sns.histplot(metric, ax=axes[1])
    axes[1].set_xlabel("Travel Dist (km)")
    axes[1].set_xlim(0,20)
    metric = sample_groups.last()['cumul_time_s']
    sns.histplot(metric, ax=axes[2])
    axes[2].set_xlabel("Travel Time (s)")
    axes[2].set_xlim(-300,3000)
    metric = sample_groups.last()['sch_time_s']
    sns.histplot(metric, ax=axes[3])
    axes[3].set_xlabel("Scheduled Time (s)")
    axes[3].set_xlim(-300,3000)
    fig.tight_layout()
    plt.savefig(f"{PLOT_FOLDER}{title_text}.eps", format='eps', dpi=600, bbox_inches='tight')
    plt.savefig(f"{PLOT_FOLDER}{title_text}.png", format='png', dpi=600, bbox_inches='tight')
    return fig, axes


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


# def save_grid_anim(data, file_name):
#     # Plot first 4 channels of second axis
#     fig, axes = plt.subplots(1, data.shape[1])
#     axes = axes.reshape(-1)
#     fig.tight_layout()
#     # Define the update function that will be called for each frame of the animation
#     def update(frame):
#         fig.suptitle(f"Frame {frame}")
#         for i in range(data.shape[1]):
#             d = data[:,i,:,:]
#             vmin=np.min(d[~np.isnan(d)])
#             vmax=np.max(d[~np.isnan(d)])
#             ax = axes[i]
#             ax.clear()
#             im = ax.imshow(data[frame,i,:,:], cmap='plasma', vmin=vmin, vmax=vmax, origin="lower")
#     # Create the animation object
#     ani = animation.FuncAnimation(fig, update, frames=data.shape[0])
#     # Save the animation object
#     ani.save(f"../plots/{file_name}", fps=10, dpi=300)


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

# Annotated geopandas plots
# fig, ax = plt.subplots(figsize=(20, 10))
# data.iloc[1:5].plot(ax=ax)
# data.iloc[1:5].apply(lambda x: ax.annotate(text=np.round(x['calc_bear_d']), xy=x.geometry.centroid.coords[0], ha='left'), axis=1)
# data.iloc[1:5].apply(lambda x: ax.annotate(text=x['locationtime'], xy=x.geometry.centroid.coords[0], ha='right'), axis=1)


# def plot_gtfsrt_trip(ax, trace_df, epsg, gtfs_folder):
#     """Plot a single real-time bus trajectory on a map.
#     """
#     # Plot trip stops from GTFS
#     trace_date = trace_df['file'].iloc[0]
#     trip_id = trace_df['trip_id'].iloc[0]
#     file_to_gtfs_map = standardfeeds.get_best_gtfs_lookup(trace_df, gtfs_folder)
#     gtfs_data = standardfeeds.merge_gtfs_files(f"{gtfs_folder}{file_to_gtfs_map[trace_date]}/", epsg, [0,0])
#     to_plot_gtfs = gtfs_data[gtfs_data['trip_id']==trip_id]
#     to_plot_gtfs = geopandas.GeoDataFrame(to_plot_gtfs, geometry=geopandas.points_from_xy(to_plot_gtfs.stop_x, to_plot_gtfs.stop_y), crs=f"EPSG:{epsg}")
#     to_plot_gtfs.plot(ax=ax, marker='x', color='lightblue', markersize=10)
#     # Plot observations
#     to_plot = trace_df.copy()
#     to_plot = geopandas.GeoDataFrame(to_plot, geometry=geopandas.points_from_xy(to_plot.x, to_plot.y), crs=f"EPSG:{epsg}")
#     to_plot_stop = trace_df.iloc[-1:,:].copy()
#     to_plot_stop = geopandas.GeoDataFrame(to_plot_stop, geometry=geopandas.points_from_xy(to_plot_stop.stop_x, to_plot_stop.stop_y), crs=f"EPSG:{epsg}")
#     to_plot.plot(ax=ax, marker='.', color='purple', markersize=20)
#     # Plot first/last observations
#     to_plot.iloc[:1,:].plot(ax=ax, marker='*', color='green', markersize=40)
#     to_plot.iloc[-1:,:].plot(ax=ax, marker='*', color='red', markersize=40)
#     # Plot closest stop to final observation
#     to_plot_stop.plot(ax=ax, marker='x', color='blue', markersize=20)
#     # Add custom legend
#     ax.legend(["Scheduled Trip Stops","Shingle Observations","Shingle Start","Shingle End", "Closest Stop"], loc="upper right")
#     return None