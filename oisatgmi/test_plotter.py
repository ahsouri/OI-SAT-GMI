import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.ticker import FormatStrFormatter
import numpy as np

def test_plotter(X,Y,Z,fname):

    pc = ccrs.PlateCarree()
    fig = plt.figure(figsize=(8, 8))
    ax = plt.axes(projection=pc)
    ax.set_extent([np.nanmin(X.flatten()), np.nanmax(X.flatten()),
                   np.nanmin(Y.flatten()), np.nanmax(Y.flatten())], crs=pc)
    ax.imshow(Z, origin='lower',
              extent=[np.nanmin(X.flatten()), np.nanmax(X.flatten()),
                      np.nanmin(Y.flatten()), np.nanmax(Y.flatten())],
              interpolation='nearest', aspect='auto', vmin=0, vmax=30)
    ax.coastlines(resolution='50m', color='black', linewidth = 2)
    # fixing tickers
    x_ticks = np.arange(np.nanmin(X.flatten()),
                        np.nanmax(X.flatten()), 20)
    x_labels = np.linspace(np.nanmin(X.flatten()), np.nanmax(
        X.flatten()), np.size(x_ticks))
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontsize=13)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    y_ticks = np.arange(np.nanmin(Y.flatten()), np.nanmax(Y.flatten()), 20)
    y_labels = np.linspace(np.nanmin(Y.flatten()), np.nanmax(Y.flatten()), np.size(y_ticks))
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize = 13)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    # plotting lat and lon
    plt.xlabel('Lon', fontsize=18)
    plt.ylabel('Lat', fontsize=18)
    fig.savefig(fname, format='png', dpi=300)
    plt.close()
