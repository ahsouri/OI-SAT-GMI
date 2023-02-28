import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.ticker import FormatStrFormatter
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


def test_plotter(X, Y, Z, fname:str, title:str, unit:int):

    pc = ccrs.PlateCarree()
    fig = plt.figure(figsize=(16, 8))
    ax = plt.axes(projection=pc)
    ax.set_extent([-180, 180,
                   -90, 90], crs=pc)
    im = ax.imshow(Z, origin='lower',
                   extent=[-180, 180,
                           -90, 90],
                   interpolation='nearest', aspect='auto', vmin=np.nanpercentile(Z.flatten(),5), vmax=np.percentile(Z.flatten(),98))
    ax.coastlines(resolution='50m', color='black', linewidth=4)
    # fixing tickers
    x_ticks = np.arange(-180,
                        180, 40)
    x_labels = np.linspace(-180, 80, np.size(x_ticks))
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontsize=18)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    y_ticks = np.arange(-90, 90, 20)
    y_labels = np.linspace(-90, 90, np.size(y_ticks))
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=18)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    # plotting lat and lon
    plt.xlabel('Lon', fontsize=20)
    plt.ylabel('Lat', fontsize=20)
    cbar = plt.colorbar(im)
    cbar.ax.tick_params(labelsize=18)
    if unit == 1:
       cbar.set_label(r'$[ \times 10^{15}molec.cm^{-2}] $', fontsize=18)
    elif unit==2:
       cbar.set_label('$ [Unitless] $', fontsize=18)
    plt.title(title, loc='left', fontweight='bold', fontsize=20)
    plt.tight_layout()
    fig.savefig(fname, format='png', dpi=300)
    plt.close()
