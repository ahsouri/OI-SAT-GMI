import numpy as np
import datetime
from netCDF4 import Dataset
import warnings
import time
from scipy.io import savemat, loadmat
from scipy.spatial import Delaunay
from scipy.interpolate import NearestNDInterpolator
from joblib import Parallel, delayed

warnings.filterwarnings("ignore", category=RuntimeWarning)


def _daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + datetime.timedelta(n)


def _read_nc(filename, var):
    # reading nc files without a group
    nc_f = filename
    nc_fid = Dataset(nc_f, 'r')
    out = np.array(nc_fid.variables[var])
    nc_fid.close()
    return np.squeeze(out)


def _savetonc(fname, folder, timetag, lat, lon, Z1, Z2, emis):
    print("Now saving " + fname)
    time_diag = timetag
    # write to the new scheme
    # ext_data = Dataset(folder + "/" +
    #                   fname, "w", format="NETCDF4")

    # write to the new scheme
    ext_data = Dataset(fname, "w", format="NETCDF4")

    time_dim = ext_data.createDimension("time", 24)
    lat_dim = ext_data.createDimension("lat", 1800)
    lon_dim = ext_data.createDimension("lon", 3600)

    times = ext_data.createVariable("time", "f8", ("time",))
    times.long_name = "time"
    times.units = "minutes since " + time_diag.strftime("%Y-%m-%d %H:%M:%S")
    times.begin_date = int(time_diag.strftime("%Y%m%d"))
    times.begin_time = int(0)
    times.time_increment = int(10000)

    latitudes = ext_data.createVariable("lat", "f8", ("lat",))
    latitudes.units = "degrees_north"
    latitudes.long_name = "latitude"
    longitudes = ext_data.createVariable("lon", "f8", ("lon",))
    longitudes.units = "degrees_east"
    longitudes.long_name = "longitude"

    emis_ff_data = ext_data.createVariable(
        "emis_ff", "f8", ("time", "lat", "lon",), fill_value=1e15)
    emis_ff_data.long_name = f"{emis} from fossil fuel"
    emis_ff_data.units = "kg m^(-2) s^(-1)"
    emis_ff_data.missing_value = np.float32(1e15)
    emis_ff_data.fmissing_value = np.float32(1e15)
    emis_ff_data.vmin = np.float32(1e15)
    emis_ff_data.vmax = np.float32(1e15)

    emis_bf_data = ext_data.createVariable(
        "emis_bf", "f8", ("time", "lat", "lon",), fill_value=1e15)
    emis_bf_data.long_name = f"{emis} from biofuel"
    emis_bf_data.units = "kg m^(-2) s^(-1)"
    emis_bf_data.missing_value = np.float32(1e15)
    emis_bf_data.fmissing_value = np.float32(1e15)
    emis_bf_data.vmin = np.float32(1e15)
    emis_bf_data.vmax = np.float32(1e15)

    times[:] = np.arange(0, 60 * 24, 60)  # 24 hours, 60 minutes intervals
    latitudes[:] = lat.squeeze()
    longitudes[:] = lon.squeeze()
    emis_ff_data[:, :, :] = Z1
    emis_bf_data[:, :, :] = Z2
    # global attributes
    ext_data.Source = "TEMPO Science Team Project - PI: Amir Souri"
    ext_data.Version = "0.0.1"
    ext_data.Institution = "NASA GSFC Code 614"
    ext_data.Contact = "Amir Souri (a.souri@nasa.gov or ahsouri@gmail.com)"
    t = time.localtime()
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", t)
    ext_data.creation_time = str(current_time)
    ext_data.close()


def merger(emis, corrs_NEI_emis, date_i):

    emis_to_saved_ff = np.zeros((24, 1800, 3600))
    emis_to_saved_bf = np.zeros((24, 1800, 3600))
    # pick the CCMI (the easiest)
    if emis == "NO":
        CCMI_file = "/discover/nobackup/projects/gmao/merra2_gmi/work/ExtData/CCMI_0.1_OS/CCMI_emis01_OS_" +\
            emis + "_" + str(date_i.year) + "_t12.nc4"
        # read NO ship emissions
        CCMI_ship_file = "/discover/nobackup/projects/gmao/merra2_gmi/work/ExtData/CCMI_0.1/CCMI_emis01_" +\
            emis + "_shp_" + str(date_i.year) + "_t12.nc4"
        ship_emissons =  _read_nc(CCMI_ship_file,'NO_shp')
        ship_emissons = ship_emissons[int(date_i.month)-1, :, :].squeeze()
    else:
        CCMI_file = "/discover/nobackup/projects/gmao/merra2_gmi/work/ExtData/CCMI_0.1/CCMI_emis01_" +\
            emis + "_" + str(date_i.year) + "_t12.nc4"
    print(f"Reading the {emis} from: " + CCMI_file)
    lat_org_compressed = _read_nc(CCMI_file, 'lat')
    lon_org_compressed = _read_nc(CCMI_file, 'lon')
    lon_org, lat_org = np.meshgrid(lon_org_compressed, lat_org_compressed)
    # global anthro emissions are monthly
    try:
        emis_ff = _read_nc(CCMI_file, f"{emis}_ff")
        emis_ff = emis_ff[int(date_i.month)-1, :, :].squeeze()
        emis_ff_exist = True
    except:
        print("there is no ff in this file, zeroing")
        emis_ff_exist = False
        emis_ff = np.zeros((1800, 3600))
    try:
        emis_bf = _read_nc(CCMI_file, f"{emis}_bf")
        emis_bf = emis_bf[int(date_i.month)-1, :, :].squeeze()
        emis_bf_exist = True
    except:
        print("there is no bf in this file, zeroing")
        emis_bf = np.zeros((1800, 3600))
        emis_bf_exist = False

    # pick the soil NOx emission only if emis == NO
    soilNOx_01 = np.zeros((24, 1800, 3600))
    if emis == "NO":
        # soil emissions are hourly
        soilNOx_file = "/discover/nobackup/asouri/SHARED/SOIL_NOX/soilnox_" + str(date_i.year) + "/" +\
            f"{date_i.month:02d}" + "/" + "soilnox_025." + \
            f"{date_i.year}{date_i.month:02d}{date_i.day:02d}.nc"
        print("Reading the soil file from " + soilNOx_file)
        lat_soil = _read_nc(soilNOx_file, 'lat')
        lon_soil = _read_nc(soilNOx_file, 'lon')
        lon_soil, lat_soil = np.meshgrid(lon_soil, lat_soil)
        soilNOx = _read_nc(soilNOx_file, 'SOIL_NOx')
        # we need to interpolate this data from 0.25x0.25 into 0.1x0.1
        points = np.zeros((np.size(lon_soil), 2))
        points[:, 0] = lon_soil.flatten()
        points[:, 1] = lat_soil.flatten()
        tri = Delaunay(points)
        for hour in range(0, 24):
            interpolator = NearestNDInterpolator(
                tri, (soilNOx[hour, :, :]).flatten())
            soilNOx_01[hour, :, :] = interpolator((lon_org, lat_org))

    # NEI2016
    NEI_file = "/discover/nobackup/asouri/SHARED/NEI_2016/nei2016_monthly/2016fh_16j_merge_0pt1degree_month_" +\
        f"{date_i.month:02d}" + ".ncf"
    print("Reading NEI file from " + NEI_file)
    if corrs_NEI_emis == "NO":
        NEI_emis = _read_nc(NEI_file, 'NO')*(30.0/46.0) + \
            _read_nc(NEI_file, 'NO2')
    else:
        NEI_emis = _read_nc(NEI_file, corrs_NEI_emis)
    lon_NEI = _read_nc(NEI_file, "lon")
    lat_NEI = _read_nc(NEI_file, "lat")
    lon_NEI, lat_NEI = np.meshgrid(lon_NEI, lat_NEI)
    points = np.zeros((np.size(lon_NEI), 2))
    points[:, 0] = lon_NEI.flatten()
    points[:, 1] = lat_NEI.flatten()
    tri = Delaunay(points)
    interpolator = NearestNDInterpolator(tri, (NEI_emis[:, :]).flatten())
    NEI_emis_mapped = interpolator((lon_org, lat_org))
    # remove data outside of lon_NEI and lat_NEI max and mins
    inside_box = (
        (lat_org >= np.min(lat_NEI.flatten())) &
        (lat_org <= np.max(lat_NEI.flatten())) &
        (lon_org >= np.min(lon_NEI.flatten())) &
        (lon_org <= np.max(lon_NEI.flatten()))
    )
    # apply mask to NOx data: keep values inside the box, zero out others
    NEI_emis_mapped = np.where(inside_box, NEI_emis_mapped, 0.0)
    # zero inside the CCMI ff
    emis_ff_masked = np.where(~inside_box, emis_ff, 0.0)
    emis_bf_masked = np.where(~inside_box, emis_bf, 0.0)
    if emis ==  'NO':
       ship_emissons_masked = np.where(~inside_box, ship_emissons, 0.0)
    else:
       ship_emissons_masked = np.zeros_like(emis_bf)
       ship_emissons = np.zeros_like(emis_bf)
    # apply the diurnal factor
    diurnal_scale_file = "/discover/nobackup/asouri/SHARED/NEI_2016/diurnal_scales/Scales_2016" +\
        f"{date_i.month:02d}.mat"
    print("Reading the scaling factor file from " + diurnal_scale_file)
    diurnal_scales = loadmat(diurnal_scale_file)
    # Check if it's a weekend
    if date_i.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        diurnal_scales = diurnal_scales[f"{corrs_NEI_emis}_weekend"]
    else:
        diurnal_scales = diurnal_scales[f"{corrs_NEI_emis}_weekday"]
    lat_scale = _read_nc(
        "/discover/nobackup/asouri/SHARED/NEI_2016/diurnal_scales/GRIDCRO2D_20190201.nc4", 'LAT')
    lon_scale = _read_nc(
        "/discover/nobackup/asouri/SHARED/NEI_2016/diurnal_scales/GRIDCRO2D_20190201.nc4", 'LON')
    points = np.zeros((np.size(lat_scale), 2))
    points[:, 0] = lon_scale.flatten()
    points[:, 1] = lat_scale.flatten()
    tri = Delaunay(points)
    for hour in range(0, 24):
        interpolator = NearestNDInterpolator(
            tri, (diurnal_scales[hour, :, :]).flatten())
        diurnal_scales_mapped = interpolator((lon_org, lat_org))
        # make the dirunal scales = 1.0 outside of the domain
        inside_box = (
            (lat_org >= np.min(lat_scale.flatten())) &
            (lat_org <= np.max(lat_scale.flatten())) &
            (lon_org >= np.min(lon_scale.flatten())) &
            (lon_org <= np.max(lon_scale.flatten()))
        )
        # apply the mask
        diurnal_scales_mapped = np.where(
            inside_box, diurnal_scales_mapped, 1.0)
        # we are summing everything, note that soilNOx is always zero unless NO is used
        # soil nox won't be applied to biofuels for NO (it will be applied to ff)
        # here if only populate ff based on soil+CCMI+NEI
        if ((emis_ff_exist) and (not emis_bf_exist)):
            emis_to_saved_ff[hour, :, :] = diurnal_scales_mapped * \
                NEI_emis_mapped+soilNOx_01[hour, :, :]+emis_ff_masked+ship_emissons_masked
            emis_to_saved_bf[hour, :, :] = 0.0
        # here if only populate bf based on CCMI+NEI
        if ((not emis_ff_exist) and (emis_bf_exist)):
            emis_to_saved_ff[hour, :, :] = 0.0
            emis_to_saved_bf[hour, :, :] = diurnal_scales_mapped * \
                NEI_emis_mapped+emis_bf_masked
        # we populate both, but we only apply NEI-2011 once
        if ((emis_ff_exist) and (emis_bf_exist)):
            emis_to_saved_ff[hour, :, :] = diurnal_scales_mapped * \
                NEI_emis_mapped+soilNOx_01[hour, :, :]+emis_ff_masked+ship_emissons_masked
            emis_to_saved_bf[hour, :, :] = emis_bf_masked
    # last touch to add non zeros
    mask = emis_to_saved_ff == 0
    emis_to_saved_ff[mask] = np.broadcast_to(emis_ff, emis_to_saved_ff.shape)[mask]
    mask = emis_to_saved_ff == 0
    emis_to_saved_ff[mask] = np.broadcast_to(ship_emissons, emis_to_saved_ff.shape)[mask]
    mask = emis_to_saved_bf == 0
    emis_to_saved_bf[mask] = np.broadcast_to(emis_bf, emis_to_saved_bf.shape)[mask]
    # saving the output
    _savetonc(f"./CCMI_SOIL_NEI2016_{emis}_{date_i.year}{date_i.month:02d}{date_i.day:02d}.nc", "", date_i, lat_org_compressed,
              lon_org_compressed, emis_to_saved_ff, emis_to_saved_bf, emis)
    return None

if __name__ == "__main__":

    emission_names_GMI = ["ALD2", "ALK4", "C2H6",
                      "PRPE", "C3H8", "CH2O", "MEK", "CO", "NO"]
    corrs_NEI_emis_list = ["ALD2", "PAR", "ETHA",
                  "IOLE", "PRPA", "FORM", "KET", "CO", "NO"]

    # loop over whole days ranging from 2023 till the end of 2024
    for date_i in _daterange(datetime.date(2023, 1, 1), datetime.date(2023, 2, 1)):
        out = Parallel(n_jobs=12)(delayed(merger)(
           emission_names_GMI[k], corrs_NEI_emis_list[k], date_i) for k in range(len(emission_names_GMI)))
