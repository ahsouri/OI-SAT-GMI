import numpy as np
from pathlib import Path
import datetime
import sys
import os
import glob
from netCDF4 import Dataset
import warnings
import time
from scipy.io import savemat

warnings.filterwarnings("ignore", category=RuntimeWarning)


def _read_nc(filename, var):
    # reading nc files without a group
    nc_f = filename
    nc_fid = Dataset(nc_f, 'r')
    out = np.array(nc_fid.variables[var])
    nc_fid.close()
    return np.squeeze(out)

# inputs
oi_diag_folder = Path(str(sys.argv[1]))
oi_diag_files = sorted(
    glob.glob(oi_diag_folder.as_posix() + "/*.nc"))
ext_files_folder = Path(str(sys.argv[2]))
if not os.path.exists(ext_files_folder.as_posix()):
    os.makedirs(ext_files_folder.as_posix())

for fname in oi_diag_files:
    # read the OI diag files
    print("Now processing " + fname)
    date = (fname.split('.')[-2]).split('_')[-1]
    time_diag = datetime.datetime(
        int(date[0:4]), int(date[4:6]), 1) + datetime.timedelta(seconds=int(0.0))
    lat = _read_nc(fname, 'lat')
    lon = _read_nc(fname, 'lon')
    scaling_factors = _read_nc(fname, 'scaling_factor')
    fbasename = os.path.basename(fname)

    # write to the new scheme
    ext_data = Dataset(ext_files_folder.as_posix() + "/" +
                       fbasename, "w", format="NETCDF4")

    time_dim = ext_data.createDimension("time", 1)
    lat_dim = ext_data.createDimension("lat", np.shape(lat)[0])
    lon_dim = ext_data.createDimension("lon", np.shape(lat)[1])

    times = ext_data.createVariable("time", "f8", ("time",))
    times.long_name = "time"
    times.units = "Hours since " + time_diag.strftime("%Y-%m-%d %H:%M:%S")

    latitudes = ext_data.createVariable("lat", "f8", ("lat",))
    latitudes.units = "degrees_north"
    latitudes.long_name = "latitude"
    longitudes = ext_data.createVariable("lon", "f8", ("lon",))
    longitudes.units = "degrees_east"
    longitudes.long_name = "longitude"

    sf = ext_data.createVariable("SF", "f8", ("time", "lat", "lon",))
    sf.units = "fraction"

    times = 0.0
    print(np.shape(lat))
    latitudes[:] = lat[:, 0].squeeze()
    longitudes[:] = lon[0, :].squeeze()
    sf[:, :, :] = scaling_factors

    # global attributes
    ext_data.Source = "OI-SAT-GMI tool (https://doi.org/10.5281/zenodo.7757427)"
    ext_data.Version = "0.0.7"
    ext_data.Institution = "NASA GSFC Code 614"
    ext_data.Contact = "Amir Souri (a.souri@nasa.gov or ahsouri@gmail.com)"
    t = time.localtime()
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", t)
    ext_data.creation_time = str(current_time)
    ext_data.close()
