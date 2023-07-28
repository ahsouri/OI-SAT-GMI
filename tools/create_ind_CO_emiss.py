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

# first we should calculate monthly-basis scaling factors from OMI (2005-2019)
omi_hcho_sf_path = "/discover/nobackup/asouri/SHARED/OI_HCHO_OMI_2005_2019/"
for mm in range(1,13):
    sf_all = []
    for yr in range(2005,2020):
        sf = _read_nc(omi_hcho_sf_path + 'HCHO_' + str(yr) + f"{mm:02}" + '.nc','SF')
        sf_all.append(sf)

OMI_SF = np.nanmean(np.array(sf_all),axis=0)
reactions_subject_to_SF = ['QQJ011','QQJ012','QQK046']
merra2_path = '/css/merra2gmi/pub'

ext_files_folder = Path(str(sys.argv[1]))
if not os.path.exists(ext_files_folder.as_posix()):
    os.makedirs(ext_files_folder.as_posix())

reactions = {}
reactions["rj2"] = ['QQJ011', 'QQJ012', 'QQJ047', 'QQJ050']
reactions["rk2"] = ['QQK204', 'QQK212', 'QQK213','QQK222','QQK039']
reactions["rk3"] = ['QQK046', 'QQK066']
reactions["rk4"] = ['QQK091', 'QQK101', 'QQK103', 'QQK109']
reactions["bio"] = ['EMBIOCOMETH','EMBIOCOMONOT']

factors = [1, 1, 1, 1, 0.42, 2.0, 1, 0.05, -1.0, 1, 1, 1, 1, 1, 1]
for yr in range(1990, 2020):
    for mm in range(1, 13):

        time_diag = datetime.datetime(
            int(yr), int(mm), 1) + datetime.timedelta(seconds=int(0.0))
        fname = 'CO_Indirect_MERRA2GMI_' + \
            str(time_diag.year) + f"{time_diag.month:02}" + '.nc'
        print("Now processing " + fname)
        merra2_dir = merra2_path + '/Y' + \
            str(time_diag.year) + '/M' + f"{time_diag.month:02}" + '/'
        merra2_dir = str(merra2_dir)

        var = np.zeros((72, 361, 576))
        var_bio = np.zeros(( 361, 576))
        cnt = -1
        for groups in reactions:
            for react in reactions[groups]:
                cnt += 1
                if str(groups) == "bio":
                   reaction = _read_nc(merra2_dir + 'MERRA2_GMI.tavg24_2d_dad_Nx.monthly.' + str(time_diag.year) +
                                     f"{time_diag.month:02}" + '.nc4', react)
                else:
                   reaction = _read_nc(merra2_dir + 'MERRA2_GMI.tavg24_3d_' + str(groups) + '_Nv.monthly.' + str(time_diag.year) +
                                     f"{time_diag.month:02}" + '.nc4', react)
                if react in reactions_subject_to_SF: # correct HCHO+hv and HCHO+OH using OMI-HCHO
                   for k in range(0,72):
                       var[k,:,:] = var[k,:,:] + reaction[k,:,:]*float(factors[cnt])*OMI_SF
                else:
                   if str(groups) == "bio":
                      var_bio = var_bio + reaction
                   else:
                      var = var + reaction*float(factors[cnt])

                lat = _read_nc(merra2_dir + 'MERRA2_GMI.tavg24_3d_' + 'rk2' + '_Nv.monthly.' + str(time_diag.year) +
                               f"{time_diag.month:02}" + '.nc4', 'lat')
                lon = _read_nc(merra2_dir + 'MERRA2_GMI.tavg24_3d_' + 'rk2' + '_Nv.monthly.' + str(time_diag.year) +
                               f"{time_diag.month:02}" + '.nc4', 'lon')
                lev = _read_nc(merra2_dir + 'MERRA2_GMI.tavg24_3d_' + 'rk2' + '_Nv.monthly.' + str(time_diag.year) +
                               f"{time_diag.month:02}" + '.nc4', 'lev')

        fname_height_mid = merra2_dir + 'MERRA2_GMI.tavg3_3d_met_Nv.monthly.' + str(time_diag.year) +\
            f"{time_diag.month:02}" + '.nc4'
        fname_height_edge = merra2_dir + 'MERRA2_GMI.tavg3_3d_mst_Ne.monthly.' + str(time_diag.year) +\
            f"{time_diag.month:02}" + '.nc4'
        height_mid = _read_nc(fname_height_mid, 'H')
        height_edge = _read_nc(fname_height_edge, 'ZLE')
        # thickness
        dh = -2.0*(height_edge[1:, :, :] - height_mid)

        # from mole/m3/s to kg/m2/s
        var = var*dh*28.01/1000.0
        # add terp+meth
        var[-1,:,:] = var_bio + var[-1,:,:]
        # write to a ncfile
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

        emiss = ext_data.createVariable(
            "emiss", "f8", ("time","lev", "lat", "lon",))
        emiss.units = "kg m^-2 s^-1"

        times[:] = 0.0
        latitudes[:] = lat
        longitudes[:] = lon
        emiss[:, :, :, :] = var
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
