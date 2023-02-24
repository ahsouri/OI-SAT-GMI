import numpy as np
from pathlib import Path
import datetime
import glob
import os
from joblib import Parallel, delayed
from netCDF4 import Dataset
from config import satellite, ctm_model
from interpolator import interpolator
import warnings
from scipy.io import savemat

warnings.filterwarnings("ignore", category=RuntimeWarning)


def _read_nc(filename, var):
    # reading nc files without a group
    nc_f = filename
    nc_fid = Dataset(nc_f, 'r')
    out = np.array(nc_fid.variables[var])
    nc_fid.close()
    return np.squeeze(out)


def _get_nc_attr(filename, var):
    # getting attributes
    nc_f = filename
    nc_fid = Dataset(nc_f, 'r')
    attr = {}
    for attrname in nc_fid.variables[var].ncattrs():
        attr[attrname] = getattr(nc_fid.variables[var], attrname)
    nc_fid.close()
    return attr


def _read_group_nc(filename, num_groups, group, var):
    # reading nc files with a group structure
    nc_f = filename
    nc_fid = Dataset(nc_f, 'r')
    if num_groups == 1:
        out = np.array(nc_fid.groups[group].variables[var])
    elif num_groups == 2:
        out = np.array(nc_fid.groups[group[0]].groups[group[1]].variables[var])
    elif num_groups == 3:
        out = np.array(
            nc_fid.groups[group[0]].groups[group[1]].groups[group[2]].variables[var])
    nc_fid.close()
    return np.squeeze(out)


def GMI_reader(product_dir: str, YYYYMM: str, gases_to_be_saved: list, frequency_opt='3-hourly', num_job=1) -> ctm_model:
    '''
       GMI reader
       Inputs:
             product_dir [str]: the folder containing the GMI data
             YYYYMM [str]: the target month and year, e.g., 202005 (May 2020)
             gases_to_be_saved [list]: name of gases to be loaded. e.g., ['NO2']
             frequency_opt: the frequency of data
                        1 -> hourly 
                        2 -> 3-hourly
                        3 -> daily
            num_obj [int]: number of jobs for parallel computation
       Output:
             gmi_fields [ctm_model]: a dataclass format (see config.py)
    '''
    # a nested function
    def gmi_reader_wrapper(fname_met: str, fname_gas: str, gasnames: list) -> ctm_model:
        # read the data
        print("Currently reading: " + fname_met.split('/')[-1])
        ctmtype = "GMI"
        lon = _read_nc(fname_met, 'lon')
        lat = _read_nc(fname_met, 'lat')
        lons_grid, lats_grid = np.meshgrid(lon, lat)
        latitude = lats_grid
        longitude = lons_grid
        time_min_delta = _read_nc(fname_met, 'time')
        time_attr = _get_nc_attr(fname_met, 'time')
        timebegin_date = str(time_attr["begin_date"])
        timebegin_time = str(time_attr["begin_time"])
        if len(timebegin_time) == 5:
            timebegin_time = "0" + timebegin_time
        timebegin_date = [int(timebegin_date[0:4]), int(
            timebegin_date[4:6]), int(timebegin_date[6:8])]
        timebegin_time = [int(timebegin_time[0:2]), int(
            timebegin_time[2:4]), int(timebegin_time[4:6])]
        time = []
        for t in range(0, np.size(time_min_delta)):
            time.append(datetime.datetime(timebegin_date[0], timebegin_date[1], timebegin_date[2],
                                          timebegin_time[0], timebegin_time[1], timebegin_time[2]) +
                        datetime.timedelta(minutes=int(time_min_delta[t])))
        delta_p = _read_nc(fname_met, 'DELP').astype('float16')/100.0
        delta_p = np.flip(delta_p, axis=1)  # from bottom to top
        pressure_mid = _read_nc(fname_met, 'PL').astype('float16')/100.0
        pressure_mid = np.flip(pressure_mid, axis=1)  # from bottom to top
        tempeature_mid = _read_nc(fname_met, 'T').astype('float16')
        tempeature_mid = np.flip(tempeature_mid, axis=1)  # from bottom to top
        gas_profile = {}
        for gas in gasnames:
            gas_profile[gas] = np.flip(_read_nc(
                fname_gas, gas).astype('float32'), axis=1)

        gmi_data = ctm_model(latitude, longitude, time, gas_profile,
                             pressure_mid, tempeature_mid, delta_p, ctmtype, [], [])
        return gmi_data

    if frequency_opt == '3-hourly':
        # read meteorological and chemical fields
        tavg3_3d_met_files = sorted(
            glob.glob(product_dir + "/*tavg3_3d_met_Nv*" + str(YYYYMM) + "*.nc4"))
        tavg3_3d_gas_files = sorted(
            glob.glob(product_dir + "/*tavg3_3d_tac_Nv*" + str(YYYYMM) + "*.nc4"))
        if len(tavg3_3d_gas_files) != len(tavg3_3d_met_files):
            raise Exception(
                "the data are not consistent")
        # define gas profiles to be saved
        outputs = Parallel(n_jobs=num_job)(delayed(gmi_reader_wrapper)(
            tavg3_3d_met_files[k], tavg3_3d_gas_files[k], gases_to_be_saved) for k in range(len(tavg3_3d_met_files)))
        return outputs


def tropomi_reader_hcho(fname: str, ctm_models_coordinate=None, read_ak=True) -> satellite:
    '''
       TROPOMI HCHO L2 reader
       Inputs:
             fname [str]: the name path of the L2 file
             ctm_models_coordinate [dict]: a dictionary containing ctm lat and lon
             read_ak [bool]: true for reading averaging kernels. this must be true for amf_recal 
       Output:
             tropomi_hcho [satellite]: a dataclass format (see config.py)
    '''
    # hcho reader
    print("Currently reading: " + fname.split('/')[-1])
    # read time
    time = _read_group_nc(fname, 1, 'PRODUCT', 'time') +\
        np.nanmean(np.array(_read_group_nc(
            fname, 1, 'PRODUCT', 'delta_time')), axis=1)/1000.0
    time = np.nanmean(time, axis=0)
    time = np.squeeze(time)
    time = datetime.datetime(
        2010, 1, 1) + datetime.timedelta(seconds=int(time))
    #print(datetime.datetime.strptime(str(tropomi_hcho.time),"%Y-%m-%d %H:%M:%S"))
    # read lat/lon at corners
    latitude_corner = _read_group_nc(fname, 3, ['PRODUCT', 'SUPPORT_DATA', 'GEOLOCATIONS'],
                                     'latitude_bounds').astype('float16')
    longitude_corner = _read_group_nc(fname, 3, ['PRODUCT', 'SUPPORT_DATA',
                                                 'GEOLOCATIONS'], 'longitude_bounds').astype('float16')
    # read total amf
    amf_total = _read_group_nc(fname, 3, ['PRODUCT', 'SUPPORT_DATA', 'DETAILED_RESULTS'],
                               'formaldehyde_tropospheric_air_mass_factor')
    # read trop no2
    vcd = _read_group_nc(fname, 1, 'PRODUCT',
                         'formaldehyde_tropospheric_vertical_column')
    scd = _read_group_nc(fname, 1, 'PRODUCT', 'formaldehyde_tropospheric_vertical_column') *\
        amf_total
    vcd = (vcd*6.02214*1e19*1e-15).astype('float16')
    scd = (scd*6.02214*1e19*1e-15).astype('float16')
    # read quality flag
    quality_flag = _read_group_nc(
        fname, 1, 'PRODUCT', 'qa_value').astype('float16')
    # read pressures for SWs
    tm5_a = _read_group_nc(
        fname, 3, ['PRODUCT', 'SUPPORT_DATA', 'INPUT_DATA'], 'tm5_constant_a')/100.0
    tm5_b = _read_group_nc(
        fname, 3, ['PRODUCT', 'SUPPORT_DATA', 'INPUT_DATA'], 'tm5_constant_b')
    ps = _read_group_nc(fname, 3, [
                        'PRODUCT', 'SUPPORT_DATA', 'INPUT_DATA'], 'surface_pressure').astype('float16')/100.0
    p_mid = np.zeros(
        (34, np.shape(vcd)[0], np.shape(vcd)[1])).astype('float32')
    if read_ak == True:
        SWs = np.zeros(
            (34, np.shape(vcd)[0], np.shape(vcd)[1])).astype('float16')
        AKs = _read_group_nc(fname, 3, [
            'PRODUCT', 'SUPPORT_DATA', 'DETAILED_RESULTS'], 'averaging_kernel').astype('float16')
    else:
        SWs = np.empty((1))
    # for some reason, in the HCHO product, a and b values are the center instead of the edges (unlike NO2)
    for z in range(0, 34):
        p_mid[z, :, :] = (tm5_a[z]+tm5_b[z]*ps[:, :])
        if read_ak == True:
            SWs[z, :, :] = AKs[:, :, z]*amf_total
    # remove bad SWs
    SWs[np.isinf(SWs)] = 0.0
    SWs[np.isnan(SWs)] = 0.0
    SWs[SWs > 100.0] = 0.0
    SWs[SWs < 0] = 0.0
    # read the precision
    uncertainty = _read_group_nc(fname, 1, 'PRODUCT',
                                 'formaldehyde_tropospheric_vertical_column_precision')
    uncertainty = (uncertainty*6.02214*1e19*1e-15).astype('float16')
    tropomi_hcho = satellite(vcd, scd, time, [], np.empty((1)), [], [
    ], latitude_corner, longitude_corner, uncertainty, quality_flag, p_mid, [], SWs, [])
    # interpolation
    if (ctm_models_coordinate is not None):
        print('Currently interpolating ...')
        grid_size = 1.0  # degree
        tropomi_hcho = interpolator(
            1, grid_size, tropomi_hcho, ctm_models_coordinate, flag_thresh=0.5)
    # return
    return tropomi_hcho


def tropomi_reader_no2(fname: str, ctm_models_coordinate=None, read_ak=True) -> satellite:
    '''
       TROPOMI NO2 L2 reader
       Inputs:
             fname [str]: the name path of the L2 file
             ctm_models_coordinate [dict]: a dictionary containing ctm lat and lon
             read_ak [bool]: true for reading averaging kernels. this must be true for amf_recal 
       Output:
             tropomi_no2 [satellite]: a dataclass format (see config.py)
    '''
    # say which file is being read
    print("Currently reading: " + fname.split('/')[-1])
    # read time
    time = _read_group_nc(fname, 1, 'PRODUCT', 'time') +\
        np.nanmean(np.array(_read_group_nc(
            fname, 1, 'PRODUCT', 'delta_time')), axis=0)/1000.0
    time = np.squeeze(time)
    tropomi_no2 = satellite
    time = datetime.datetime(
        2010, 1, 1) + datetime.timedelta(seconds=int(time))
    #print(datetime.datetime.strptime(str(tropomi_no2.time),"%Y-%m-%d %H:%M:%S"))
    # read lat/lon at corners
    latitude_corner = _read_group_nc(fname, 3, ['PRODUCT', 'SUPPORT_DATA',
                                                'GEOLOCATIONS'], 'latitude_bounds').astype('float32')
    longitude_corner = _read_group_nc(fname, 3, ['PRODUCT', 'SUPPORT_DATA',
                                                 'GEOLOCATIONS'], 'longitude_bounds').astype('float32')
    # read total amf
    amf_total = _read_group_nc(fname, 1, 'PRODUCT', 'air_mass_factor_total')
    # read trop no2
    vcd = _read_group_nc(
        fname, 1, 'PRODUCT', 'nitrogendioxide_tropospheric_column')
    scd = _read_group_nc(fname, 1, 'PRODUCT', 'nitrogendioxide_tropospheric_column') *\
        _read_group_nc(fname, 1, 'PRODUCT', 'air_mass_factor_troposphere')
    vcd = (vcd*6.02214*1e19*1e-15).astype('float16')
    scd = (scd*6.02214*1e19*1e-15).astype('float16')
    # read quality flag
    quality_flag = _read_group_nc(
        fname, 1, 'PRODUCT', 'qa_value').astype('float16')
    # read pressures for SWs
    tm5_a = _read_group_nc(fname, 1, 'PRODUCT', 'tm5_constant_a')/100.0
    tm5_a = np.concatenate((tm5_a[:, 0], 0), axis=None)
    tm5_b = _read_group_nc(fname, 1, 'PRODUCT', 'tm5_constant_b')
    tm5_b = np.concatenate((tm5_b[:, 0], 0), axis=None)

    ps = _read_group_nc(fname, 3, [
                        'PRODUCT', 'SUPPORT_DATA', 'INPUT_DATA'], 'surface_pressure').astype('float32')/100.0
    p_mid = np.zeros(
        (34, np.shape(vcd)[0], np.shape(vcd)[1])).astype('float16')
    if read_ak == True:
        SWs = np.zeros(
            (34, np.shape(vcd)[0], np.shape(vcd)[1])).astype('float16')
        AKs = _read_group_nc(fname, 1, 'PRODUCT',
                             'averaging_kernel').astype('float16')
    else:
        SWs = np.empty((1))
    for z in range(0, 34):
        p_mid[z, :, :] = 0.5*(tm5_a[z]+tm5_b[z]*ps[:, :] +
                              tm5_a[z+1]+tm5_b[z+1]*ps[:, :])
        if read_ak == True:
            SWs[z, :, :] = AKs[:, :, z]*amf_total
    # remove bad SWs
    SWs[np.isinf(SWs)] = 0.0
    SWs[np.isnan(SWs)] = 0.0
    SWs[SWs > 100.0] = 0.0
    SWs[SWs < 0] = 0.0
    # read the tropopause layer index
    trop_layer = _read_group_nc(
        fname, 1, 'PRODUCT', 'tm5_tropopause_layer_index')
    tropopause = np.zeros_like(trop_layer).astype('float16')
    for i in range(0, np.shape(trop_layer)[0]):
        for j in range(0, np.shape(trop_layer)[1]):
            if (trop_layer[i, j] > 0 and trop_layer[i, j] < 34):
                tropopause[i, j] = p_mid[trop_layer[i, j], i, j]
            else:
                tropopause[i, j] = np.nan
    # read the precision
    uncertainty = _read_group_nc(fname, 1, 'PRODUCT',
                                 'nitrogendioxide_tropospheric_column_precision')
    uncertainty = (uncertainty*6.02214*1e19*1e-15).astype('float16')
    # populate tropomi class
    tropomi_no2 = satellite(vcd, scd, time, [], tropopause, [], [
    ], latitude_corner, longitude_corner, uncertainty, quality_flag, p_mid, [], SWs, [])
    # interpolation
    if (ctm_models_coordinate is not None):
        print('Currently interpolating ...')
        grid_size = 2.5  # degree
        tropomi_no2 = interpolator(
            1, grid_size, tropomi_no2, ctm_models_coordinate, flag_thresh=0.75)
    # return
    return tropomi_no2


def tropomi_reader(product_dir: str, satellite_product_name: str, ctm_models_coordinate: dict, YYYYMM: str, read_ak=True, num_job=1):
    '''
        reading tropomi data
             product_dir [str]: the folder containing the tropomi data
             satellite_product_name [str]: so far we support:
                                         "NO2"
                                         "HCHO"
             ctm_models_coordinate [dict]: the ctm coordinates
             YYYYMM [int]: the target month and year, e.g., 202005 (May 2020)
             read_ak [bool]: true for reading averaging kernels. this must be true for amf_recal
             num_job [int]: the number of jobs for parallel computation
        Output [tropomi]: the tropomi @dataclass
    '''

    # find L2 files first
    L2_files = sorted(glob.glob(product_dir + "/*" + str(YYYYMM) + "*.nc"))
    # read the files in parallel
    if satellite_product_name.split('_')[-1] == 'NO2':
        outputs_sat = Parallel(n_jobs=num_job)(delayed(tropomi_reader_no2)(
            L2_files[k], ctm_models_coordinate=ctm_models_coordinate, read_ak=read_ak) for k in range(len(L2_files)))
    elif satellite_product_name.split('_')[-1] == 'HCHO':
        outputs_sat = Parallel(n_jobs=num_job)(delayed(tropomi_reader_hcho)(
            L2_files[k], ctm_models_coordinate=ctm_models_coordinate, read_ak=read_ak) for k in range(len(L2_files)))
    return outputs_sat


class readers(object):
    def __init__(self) -> None:
        pass

    def add_satellite_data(self, product_name: str, product_dir: Path):
        '''
            add L2 data
            Input:
                product_name [str]: a string specifying the type of data to read:
                                   1 > TROPOMI_NO2
                                   2 > TROPOMI_HCHO
                                   3 > TROPOMI_CH4
                                   4 > TROPOMI_CO     
                product_dir  [Path]: a path object describing the path of L2 files
        '''
        self.satellite_product_dir = product_dir
        self.satellite_product_name = product_name

    def add_ctm_data(self, product_name: int, product_dir: Path):
        '''
            add CTM data
            Input:
                product_name [str]: an string specifying the type of data to read:
                                "GMI"
                product_dir  [Path]: a path object describing the path of CTM files
        '''

        self.ctm_product_dir = product_dir
        self.ctm_product = product_name

    def read_satellite_data(self, YYYYMM: str, read_ak=True, num_job=1):
        '''
            read L2 satellite data
            Input:
             YYYYMM [str]: the target month and year, e.g., 202005 (May 2020)
             read_ak [bool]: true for reading averaging kernels. this must be true for amf_recal
             num_job [int]: the number of jobs for parallel computation
        '''

        ctm_models_coordinate = {}
        ctm_models_coordinate["Latitude"] = self.ctm_data[0].latitude
        ctm_models_coordinate["Longitude"] = self.ctm_data[0].longitude
        self.tropomi_data = tropomi_reader(self.satellite_product_dir.as_posix(),
                                           self.satellite_product_name, ctm_models_coordinate,
                                           YYYYMM,  read_ak=read_ak, num_job=num_job)

    def read_ctm_data(self, YYYYMM: str, gases: list, frequency_opt: str, num_job=1):
        '''
            read ctm data
            Input:
             YYYYMM [str]: the target month and year, e.g., 202005 (May 2020)
             gases_to_be_saved [list]: name of gases to be loaded. e.g., ['NO2']
             frequency_opt: the frequency of data
                        1 -> hourly 
                        2 -> 3-hourly
                        3 -> daily
             num_job [int]: the number of jobs for parallel computation
        '''
        if self.ctm_product == 'GMI':
            self.ctm_data = GMI_reader(self.ctm_product_dir.as_posix(), YYYYMM, gases,
                                       frequency_opt=frequency_opt, num_job=num_job)


# testing
if __name__ == "__main__":
    reader_obj = readers()
    reader_obj.add_ctm_data('GMI', Path('download_bucket/gmi/'))
    reader_obj.read_ctm_data('201905', ['NO2'], frequency_opt='3-hourly')
    reader_obj.add_satellite_data(
        'TROPOMI_NO2', Path('download_bucket/no2/'))
    reader_obj.read_satellite_data('201905', read_ak=True, num_job=1)

    latitude = reader_obj.tropomi_data[0].latitude_center
    longitude = reader_obj.tropomi_data[0].longitude_center

    output = np.zeros((np.shape(latitude)[0], np.shape(
        latitude)[1], len(reader_obj.tropomi_data)))
    counter = -1
    for trop in reader_obj.tropomi_data:
        counter = counter + 1
        output[:, :, counter] = trop.vcd

    output[output <= 0.0] = np.nan
    moutput = {}
    moutput["vcds"] = output
    moutput["lat"] = latitude
    moutput["lon"] = longitude
    savemat("vcds.mat", moutput)
