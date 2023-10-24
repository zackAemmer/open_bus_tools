import os

import contextily as cx
import geopandas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly
import seaborn as sns

HEIGHT=6
WIDTH=8
HEIGHT_WIDE=3
ASPECT_WIDE=4
HEIGHT_SQ=12
WIDTH_SQ=12
PLOT_FOLDER="../plots/"


def formatted_lineplot(plot_df, x_var, y_var, title_text):
    fig, axes = plt.subplots(1,1)
    fig.set_figheight(HEIGHT)
    fig.set_figwidth(WIDTH)
    sns.lineplot(plot_df, x=x_var, y=y_var, ax=axes)
    fig.suptitle(title_text, fontsize=16)
    fig.tight_layout()
    plt.savefig(f"{PLOT_FOLDER}{title_text}.eps", format='eps', dpi=600, bbox_inches='tight')
    plt.savefig(f"{PLOT_FOLDER}{title_text}.png", format='png', dpi=600, bbox_inches='tight')
    return None

def formatted_rel_lineplot(plot_df, x_var, y_var, rel_var, title_text, xlim=None, ylim=None):
    g = sns.relplot(plot_df, x=x_var, y=y_var, row=rel_var, kind='line', height=HEIGHT_WIDE, aspect=ASPECT_WIDE)
    if xlim:
        g.set(xlim=xlim)
    if ylim:
        g.set(ylim=ylim)
    plt.savefig(f"{PLOT_FOLDER}{title_text}.eps", format='eps', dpi=600, bbox_inches='tight')
    plt.savefig(f"{PLOT_FOLDER}{title_text}.png", format='png', dpi=600, bbox_inches='tight')
    return None

def formatted_basemap_scatterplot(plot_gdf, title_text):
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