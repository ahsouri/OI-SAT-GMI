import os.path
import os
import numpy as np
from fpdf import FPDF
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.ticker import FormatStrFormatter
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plotter(X, Y, Z, fname: str, title: str, unit: int, vmin,vmax):

    pc = ccrs.PlateCarree()
    fig = plt.figure(figsize=(16, 8))
    ax = plt.axes(projection=pc)
    ax.set_extent([-180, 180,
                   -90, 90], crs=pc)
    im = ax.imshow(Z, origin='lower',
                   extent=[-180, 180,
                           -90, 90],
                   interpolation='nearest', aspect='auto', vmin=vmin, vmax=vmax)
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
    elif unit == 2:
        cbar.set_label('$ [Unitless] $', fontsize=18)
    plt.title(title, loc='left', fontweight='bold', fontsize=20)
    plt.tight_layout()
    fig.savefig(fname, format='png', dpi=300)
    plt.close()


def report(lon: np.ndarray, lat: np.ndarray, ctm_vcd_before: np.ndarray, ctm_vcd_after: np.ndarray,
           sat_vcd: np.ndarray, increment: np.ndarray, averaging_kernel: np.ndarray, error_OI: np.ndarray, 
           new_amf: np.ndarray, old_amf: np.ndarray, fname):
    '''
    '''
    if not os.path.exists('temp'):
        os.makedirs('temp')
    plotter(lon, lat, ctm_vcd_before, 'temp/ctm_vcd_before_' +
            fname + '.png', 'CTM VCD (prior)', 1,0.0,10.0)
    plotter(lon, lat, ctm_vcd_after, 'temp/ctm_vcd_after_' +
            fname + '.png', 'CTM VCD (posterior)', 1,0.0,10.0)
    plotter(lon, lat, sat_vcd, 'temp/sat_vcd_' + fname +
            '.png', 'Satellite Observation (Y)', 1,0.0,10.0)
    plotter(lon, lat, increment, 'temp/increment_' +
            fname + '.png', 'Increment', 1, -5.0,5.0)
    plotter(lon, lat, averaging_kernel, 'temp/ak_' +
            fname + '.png', 'Averaging Kernels', 2,0.0,1.0)
    plotter(lon, lat, error_OI, 'temp/error_' +
            fname + '.png', 'OI estimate error', 1,0.0,10.0)
    plotter(lon, lat, new_amf, 'temp/new_amf_' +
            fname + '.png', 'new AMF', 2,0.0,4)
    plotter(lon, lat, old_amf, 'temp/old_amf_' +
            fname + '.png', 'old AMF', 2,0.0,4)