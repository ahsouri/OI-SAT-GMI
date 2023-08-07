import numpy as np
from pathlib import Path
import datetime
import sys
import os
from netCDF4 import Dataset
import warnings
import time

warnings.filterwarnings("ignore", category=RuntimeWarning)

def _read_nc(filename, var):
    # reading nc files without a group
    nc_f = filename
    nc_fid = Dataset(nc_f, 'r')
    out = np.array(nc_fid.variables[var])
    nc_fid.close()
    return np.squeeze(out)

N_A = 6.02214076e23
R = 8.314e4  # cm^3 mbar /K /mol

merra2_path = '/css/merra2gmi/pub'

ext_files_folder = Path(str(sys.argv[1]))
if not os.path.exists(ext_files_folder.as_posix()):
    os.makedirs(ext_files_folder.as_posix())

year = 2005
for mm in range(1,13):
    time_diag = datetime.datetime(
               int(year), int(mm), 1) + datetime.timedelta(seconds=int(0.0))
    merra2_dir = merra2_path + '/Y' + \
                str(time_diag.year) + '/M' + f"{time_diag.month:02}" + '/'
    merra2_dir = str(merra2_dir)
    # extracting OH values
    OH = _read_nc(merra2_dir + 'MERRA2_GMI.tavg24_3d_dac_Nv.monthly.' + str(time_diag.year) +
                  f"{time_diag.month:02}" + '.nc4', 'OH')
    lat = _read_nc(merra2_dir + 'MERRA2_GMI.tavg24_3d_dac_Nv.monthly.'  + str(time_diag.year) +
                               f"{time_diag.month:02}" + '.nc4', 'lat')
    lon = _read_nc(merra2_dir + 'MERRA2_GMI.tavg24_3d_dac_Nv.monthly.' + str(time_diag.year) +
                               f"{time_diag.month:02}" + '.nc4', 'lon')
    lev = _read_nc(merra2_dir + 'MERRA2_GMI.tavg24_3d_dac_Nv.monthly.' + str(time_diag.year) +
                               f"{time_diag.month:02}" + '.nc4', 'lev')
    # met vars
    PL = _read_nc( merra2_dir + 'MERRA2_GMI.tavg3_3d_met_Nv.monthly.' + str(time_diag.year) +\
                f"{time_diag.month:02}"  + '.nc4','PL')/100.0
    T = _read_nc( merra2_dir + 'MERRA2_GMI.tavg3_3d_met_Nv.monthly.' + str(time_diag.year) +\
                f"{time_diag.month:02}"  + '.nc4','T')
    OH =  OH*N_A*PL/R/T # molec/cm3

    # write to a ncfile
    fname = 'OH_Conc_' + str(year) + f"{time_diag.month:02}" + '.nc'
    ext_data = Dataset(ext_files_folder.as_posix() + "/" +
                           fname, "w", format="NETCDF4")
    time_dim = ext_data.createDimension("time", 1)
    lat_dim = ext_data.createDimension("lat", np.size(lat))
    lon_dim = ext_data.createDimension("lon", np.size(lon))
    lev_dim = ext_data.createDimension("lev", 72)
    times = ext_data.createVariable("time", "f8", ("time",))
    times.long_name = "time"
    times.units = "hours since " + time_diag.strftime("%Y-%m-%d %H:%M:%S")
    latitudes = ext_data.createVariable("lat", "f8", ("lat",))
    latitudes.units = "degrees_north"
    latitudes.long_name = "latitude"
    longitudes = ext_data.createVariable("lon", "f8", ("lon",))
    longitudes.units = "degrees_east"
    longitudes.long_name = "longitude"
    levels = ext_data.createVariable("lev", "f8", ("lev",))
    levels.units = "layer"
    levels.long_name = "vertical layer"
    levels.positive = "down"

    OH_var = ext_data.createVariable(
            "OH", "f8", ("time","lev", "lat", "lon",))
    OH_var.units = "molec cm^-3"

    times[:] = 0.0
    latitudes[:] = lat
    longitudes[:] = lon
    OH_var[:, :, :] = OH
    levels[:] = lev
    # global attributes
    ext_data.Source = "OI-SAT-GMI tool (https://doi.org/10.5281/zenodo.7757427)"
    ext_data.Version = "0.0.8"
    ext_data.Institution = "NASA GSFC Code 614"
    ext_data.Contact = "Amir Souri (a.souri@nasa.gov or ahsouri@gmail.com)"
    t = time.localtime()
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", t)
    ext_data.creation_time = str(current_time)
    ext_data.close()
