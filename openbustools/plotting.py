import os

import contextily as cx
import geopandas
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd
from pathlib import Path
import plotly
import scipy.stats as stats
import seaborn as sns
import statsmodels.api as sm

from openbustools import standardfeeds, spatial

HEIGHT=10
WIDTH=8
HEIGHT_WIDE=3
ASPECT_WIDE=4
HEIGHT_SQ=6
WIDTH_SQ=6
PLOT_FOLDER="../plots/"
PALETTE="tab10"


def drive_cycle_energy_plot(plot_simdrives, title_text="throwaway"):
    plot_in_vars = {
        "Speed (mph)": "mph",
        "Grade (%/100)": "grade"
    }
    plot_out_vars = {
        "Total (kWh)": "ess_kw_out_ach",
        "Acceleration (kWh)": "accel_kw",
        "Rolling (kWh)": "rr_kw",
        "Loss (kWh)": "ess_loss_kw",
        "Ascent (kWh)": "ascent_kw",
        "Drag (kWh)": "drag_kw",
        "Aux (kWh)": "aux_in_kw"
    }
    ax_lims = {
        "Speed (mph)": [0, 70],
        "Grade (%/100)": [-10, 10],
        "Total (kWh)": [0, 100],
        "Acceleration (kWh)": [0, 5],
        "Rolling (kWh)": [0, 50],
        "Loss (kWh)": [0, 5],
        "Ascent (kWh)": [0, 15],
        "Drag (kWh)": [0, 50],
        "Aux (kWh)": [0, 20]
    }
    fig, axes = plt.subplots(len(plot_in_vars)+len(plot_out_vars), len(plot_simdrives), figsize=(20,20))
    if axes.ndim == 1:
        axes = axes.reshape(1,-1)
    fig.tight_layout()
    for traj_n in range(len(plot_simdrives)):
        axes[0,traj_n].set_title(f"Trajectory {traj_n}")
        sim_drive = plot_simdrives[traj_n]
        for source, res in sim_drive.items():
            for i, (title, var) in enumerate(plot_in_vars.items()):
                ax = axes[i,traj_n]
                ax.set_ylabel(title)
                ax.plot(res[1].cyc.time_s, getattr(res[1].cyc, var), label=source)
            for i, (title, var) in enumerate(plot_out_vars.items()):
                i += len(plot_in_vars)
                ax = axes[i,traj_n]
                ax.set_ylabel(title)
                ax.set_ylim(ax_lims[title])
                ax.plot(res[1].cyc.time_s, np.cumsum(getattr(res[1], var)) / 60 /60, label=source)
    axes[0,0].legend()
    return None


def drive_cycle_comparison(plot_energy_res, title_text="throwaway"):
    fig, axes = plt.subplots(2,1,figsize=(15,6))
    fig.tight_layout()
    axes = axes.flatten()
    sns.lineplot(x=plot_energy_res['agg'][0].time_s, y=plot_energy_res['agg'][0].mps, ax=axes[0], label="Aggregated")
    sns.lineplot(x=plot_energy_res['pred'][0].time_s, y=plot_energy_res['pred'][0].mps, ax=axes[0], label="Predicted")
    agg_smooth = spatial.apply_sg_filter(plot_energy_res['agg'][0].mps, 8, 0, 30)
    pred_smooth = spatial.apply_sg_filter(plot_energy_res['pred'][0].mps, 8, 0, 30)
    sns.lineplot(x=plot_energy_res['agg'][0].time_s, y=agg_smooth, ax=axes[1], label="Aggregated")
    sns.lineplot(x=plot_energy_res['pred'][0].time_s, y=pred_smooth, ax=axes[1], label="Predicted")
    axes[0].set_title("Example Drive Cycle Constructed with Aggregated vs. Predicted Speed Methods")
    axes[0].set_ylabel("Speed (m/s)")
    axes[1].set_ylabel("Smoothed Speed (m/s)")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend("")
    axes[0].set_ylim(0, 20)
    axes[1].set_ylim(0, 20)
    plt.savefig(Path(PLOT_FOLDER, title_text).with_suffix(".png"), format='png', dpi=600, bbox_inches='tight')
    return None

def formatted_lineplot(plot_df, x_var, y_var, title_text="throwaway"):
    fig, axes = plt.subplots(1,1)
    fig.set_figheight(HEIGHT)
    fig.set_figwidth(WIDTH)
    sns.lineplot(plot_df, x=x_var, y=y_var, ax=axes)
    fig.suptitle(title_text, fontsize=16)
    fig.tight_layout()
    plt.savefig(Path(PLOT_FOLDER, title_text).with_suffix(".png"), format='png', dpi=600, bbox_inches='tight')
    return None


def formatted_rel_lineplot(plot_df, x_var, y_var, rel_var, title_text="throwaway", xlim=None, ylim=None):
    g = sns.relplot(plot_df, x=x_var, y=y_var, row=rel_var, kind='line', height=HEIGHT_WIDE, aspect=ASPECT_WIDE)
    if xlim:
        g.set(xlim=xlim)
    if ylim:
        g.set(ylim=ylim)
    plt.savefig(Path(PLOT_FOLDER, title_text).with_suffix(".png"), format='png', dpi=600, bbox_inches='tight')
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
    plt.savefig(Path(PLOT_FOLDER, title_text).with_suffix(".png"), format='png', dpi=600, bbox_inches='tight')
    return fig, axes


def formatted_shingle_scatterplot(plot_gdf, title_text="throwaway"):
    """Plot realtime for a single day/trip combo across its shingles."""
    fig, axes = plt.subplots(1,1)
    fig.set_figheight(HEIGHT_SQ)
    fig.set_figwidth(WIDTH_SQ)
    plot_gdf.plot(ax=axes)
    plot_gdf.iloc[0:1].plot(ax=axes, markersize=1000, color='green', marker='x')
    plot_gdf.iloc[-1:].plot(ax=axes, markersize=1000, color='red', marker='x')
    cx.add_basemap(ax=axes, crs=plot_gdf.crs.to_string(), alpha=0.4, source=cx.providers.MapBox(accessToken=os.getenv(key="MAPBOX_TOKEN")))
    fig.tight_layout()
    plt.savefig(Path(PLOT_FOLDER, title_text).with_suffix(".png"), format='png', dpi=600, bbox_inches='tight')
    return fig, axes


def formatted_feature_distributions_histplot(plot_df, title_text="throwaway"):
    """Plot distributions of labels and key features."""
    data_folder_list = list(pd.unique(plot_df['realtime_foldername']))
    fig, axes = plt.subplots(4,2)
    axes = axes.flatten()
    [ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f')) for ax in axes]
    fig.set_figheight(HEIGHT)
    fig.set_figwidth(WIDTH)
    for i, data_folder in enumerate(data_folder_list):
        plot_df_folder = plot_df[plot_df['realtime_foldername']==data_folder]
        sample_groups = plot_df_folder.groupby('shingle_id')
        metric = sample_groups.count()['locationtime']
        sns.histplot(metric, ax=axes[0], stat='density', binwidth=1, color=sns.color_palette(PALETTE)[i])
        axes[0].set_xlabel("Points per Sample (n)")
        axes[0].set_xlim(0,70)
        axes[0].legend(["Seattle (KCM)", "Trondheim (AtB)"])
        metric = sample_groups.last()['cumul_dist_km']
        sns.histplot(metric, ax=axes[1], stat='density', binwidth=.5, color=sns.color_palette(PALETTE)[i])
        axes[1].set_xlabel("Sample Travel Distance (km)")
        axes[1].set_xlim(0,20)
        metric = sample_groups.last()['cumul_time_s']
        sns.histplot(metric, ax=axes[2], stat='density', binwidth=50, color=sns.color_palette(PALETTE)[i])
        axes[2].set_xlabel("Sample Travel Time (s)")
        axes[2].set_xlim(-300,3000)
        metric = sample_groups.last()['sch_time_s']
        sns.histplot(metric, ax=axes[3], stat='density', binwidth=100, color=sns.color_palette(PALETTE)[i])
        axes[3].set_xlabel("Sample Scheduled Time (s)")
        axes[3].set_xlim(-300,3000)
        metric = plot_df_folder['x_cent']/1000
        sns.histplot(metric, ax=axes[4], stat='density', binwidth=1, color=sns.color_palette(PALETTE)[i])
        axes[4].set_xlabel("Point CBD-X (km)")
        axes[4].set_xlim(-20,20)
        metric = plot_df_folder['y_cent']/1000
        sns.histplot(metric, ax=axes[5], stat='density', binwidth=1, color=sns.color_palette(PALETTE)[i])
        axes[5].set_xlabel("Point CBD-Y (km)")
        axes[5].set_xlim(-20,20)
        # axes[5].xticks(np.arange(min(-30000), max(x)+1, 1.0))
        metric = plot_df_folder['calc_time_s']
        sns.histplot(metric, ax=axes[6], stat='density', binwidth=1, color=sns.color_palette(PALETTE)[i])
        axes[6].set_xlabel("Point Measured Time (s)")
        axes[6].set_xlim(0,1*60)
        metric = plot_df_folder['calc_speed_m_s']
        sns.histplot(metric, ax=axes[7], stat='density', binwidth=1, color=sns.color_palette(PALETTE)[i])
        axes[7].set_xlabel("Avg Speed (m/s)")
        axes[7].set_xlim(0,35)
    fig.tight_layout()
    plt.savefig(Path(PLOT_FOLDER, title_text).with_suffix(".png"), format='png', dpi=600, bbox_inches='tight')
    return fig, axes


def formatted_trajectory_lineplot(traj_df, title_text="throwaway"):
    if 'Source' not in traj_df.columns:
        traj_df['Source'] = 'unknown'
    if 'cumul_time_s' not in traj_df.columns:
        traj_df['cumul_time_s'] = np.arange(len(traj_df))
    plot_df = traj_df.reset_index().melt(id_vars=['cumul_time_s', 'Source'])
    plot_df = plot_df[plot_df['variable'].isin(['Distance','Velocity','Acceleration'])]
    plot_df['Variable'] = plot_df['variable']
    plot_df['Value'] = plot_df['value']
    plot_df.loc[plot_df['Variable']=='Velocity', 'Variable'] = 'Velocity (m/s)'
    plot_df.loc[plot_df['Variable']=='Distance', 'Variable'] = 'Distance (m)'
    plot_df.loc[plot_df['Variable']=='Acceleration', 'Variable'] = 'Acceleration (m/s2)'
    plot_df['Cumulative Time (s)'] = plot_df['cumul_time_s']
    fig = sns.FacetGrid(plot_df, row='Variable', hue='Source', height=1.7, aspect=4, sharey=False)
    fig.map(sns.lineplot, 'Cumulative Time (s)', 'Value')
    fig.add_legend()
    fig.tight_layout()
    plt.savefig(Path(PLOT_FOLDER, title_text).with_suffix(".png"), format='png', dpi=600, bbox_inches='tight')
    return fig

def formatted_forces_lineplot(traj_df, title_text="throwaway"):
    if 'Source' not in traj_df.columns:
        traj_df['Source'] = 'unknown'
    if 'cumul_time_s' not in traj_df.columns:
        traj_df['cumul_time_s'] = np.arange(len(traj_df))
    plot_df = traj_df.reset_index().melt(id_vars=['cumul_time_s', 'Source'])
    plot_df = plot_df[plot_df['variable'].isin(['F_aero','F_grav','F_roll', 'F_acc', 'P_tot'])]
    plot_df['Variable'] = plot_df['variable']
    plot_df['Value'] = plot_df['value']
    plot_df.loc[plot_df['Variable']=='F_aero', 'Variable'] = 'Aerodynamic (kg m/s2)'
    plot_df.loc[plot_df['Variable']=='F_grav', 'Variable'] = 'Gravity (kg m/s2)'
    plot_df.loc[plot_df['Variable']=='F_roll', 'Variable'] = 'Rolling (kg m/s2)'
    plot_df.loc[plot_df['Variable']=='F_acc', 'Variable'] = 'Acceleration (kg m/s2)'
    plot_df.loc[plot_df['Variable']=='P_tot', 'Variable'] = 'Power (W)'
    plot_df['Cumulative Time (s)'] = plot_df['cumul_time_s']
    fig = sns.FacetGrid(plot_df, row='Variable', hue='Source', height=1.7, aspect=4, sharey=False)
    fig.map(sns.lineplot, 'Cumulative Time (s)', 'Value')
    fig.add_legend()
    fig.tight_layout()
    plt.savefig(Path(PLOT_FOLDER, title_text).with_suffix(".png"), format='png', dpi=600, bbox_inches='tight')
    return fig


def formatted_residuals_plot(plot_df, title_text="throwaway"):
    """Plot residuals of a model."""
    fig, axes = plt.subplots(3,1)
    fig.set_figheight(HEIGHT_SQ)
    fig.set_figwidth(WIDTH_SQ)
    sns.residplot(plot_df, ax=axes[0], x='labels', y='preds', lowess=True, scatter_kws={'marker': '.'}, line_kws={'color': 'red'})
    sm.qqplot(plot_df['residuals'], ax=axes[1], dist=stats.t, distargs=(len(plot_df)-1,), line='45', fit=True)
    sns.histplot(plot_df['residuals'], ax=axes[2], bins=100)
    fig.suptitle(title_text, fontsize=16)
    fig.tight_layout()
    plt.savefig(Path(PLOT_FOLDER, title_text).with_suffix(".png"), format='png', dpi=600, bbox_inches='tight')
    return None


def formatted_grid_animation(data, title_text="throwaway"):
    fig, axes = plt.subplots(1, 1)
    fig.tight_layout()
    # Define the update function that will be called for each frame of the animation
    def update(frame):
        fig.suptitle(f"Frame {frame}")
        for i in range(1):
            d = data[:,:,:]
            vmin=np.min(d[~np.isnan(d)])
            vmax=np.max(d[~np.isnan(d)])
            axes.clear()
            im = axes.imshow(data[frame,:,:], cmap='plasma', vmin=vmin, vmax=vmax, origin="lower")
    # Create the animation object
    ani = animation.FuncAnimation(fig, update, frames=data.shape[0])
    writergif = animation.PillowWriter(fps=30)
    # Save the animation object
    ani.save(f"../plots/{title_text}.gif", writer=writergif)


# def formatted_barplot(plot_df):
#     fig, axes = plt.subplots(1,1)
#     fig.set_figheight(HEIGHT)
#     fig.set_figwidth(WIDTH)
#     sns.barplot(plot_df, x=x_var, y=y_var, ax=axes)
#     axes.set_xlim([0, 0.5])
#     fig.suptitle('KCM Model Performance', fontsize=16)
#     fig.tight_layout()
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