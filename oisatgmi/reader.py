import numpy as np
from pathlib import Path
import datetime
import glob
from joblib import Parallel, delayed
from netCDF4 import Dataset
from oisatgmi.config import satellite_amf, satellite_opt, ctm_model, satellite_ssmis
from oisatgmi.interpolator import interpolator
from oisatgmi.interpolator_ssmis import interpolator_ssmis
from oisatgmi.filler_gosat import filler_gosatxch4
import warnings
from scipy.io import savemat
import yaml
import os
import h5py

warnings.filterwarnings("ignore", category=RuntimeWarning)


def _read_nc(filename, var):
    # reading nc files without a group
    nc_f = filename
    nc_fid = Dataset(nc_f, 'r')
    out = np.array(nc_fid.variables[var])
    nc_fid.close()
    return np.squeeze(out)

def _read_ssmi(filename, var):
    # reading nc files without a group
    nc_f = filename
    nc_fid = h5py.File(nc_f, 'r')
    out = np.array(nc_fid[var])
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


def _get_nc_attr_group_mopitt(fname):
    # getting attributes for mopitt
    nc_f = fname
    nc_fid = Dataset(nc_f, 'r')
    attr = {}
    for attrname in nc_fid.groups['HDFEOS'].groups['ADDITIONAL'].groups['FILE_ATTRIBUTES'].ncattrs():
        attr[attrname] = getattr(
            nc_fid.groups['HDFEOS'].groups['ADDITIONAL'].groups['FILE_ATTRIBUTES'], attrname)
    nc_fid.close()
    return attr


def _read_group_nc(filename, group, var):
    # reading nc files with a group structure
    nc_f = filename
    nc_fid = Dataset(nc_f, 'r')
    num_groups = len(group)
    if num_groups == 1:
        out = np.array(nc_fid.groups[group[0]].variables[var])
    elif num_groups == 2:
        out = np.array(nc_fid.groups[group[0]].groups[group[1]].variables[var])
    elif num_groups == 3:
        out = np.array(
            nc_fid.groups[group[0]].groups[group[1]].groups[group[2]].variables[var])
    elif num_groups == 4:
        out = np.array(
            nc_fid.groups[group[0]].groups[group[1]].groups[group[2]].groups[group[3]].variables[var])
    nc_fid.close()
    return np.squeeze(out)

def _remove_empty_files(filelist:list):
    # remove empty files from a list
    for file in filelist:
        if os.path.getsize(file)<100:
            filelist.remove(file)
    return filelist

def GMI_reader(product_dir: str, YYYYMM: str, gas_to_be_saved: list, frequency_opt='3-hourly', num_job=1) -> ctm_model:
    '''
       GMI reader
       Inputs:
             product_dir [str]: the folder containing the GMI data
             YYYYMM [str]: the target month and year, e.g., 202005 (May 2020)
             gases_to_be_saved [list]: name of gases to be loaded. e.g., ['NO2']
             frequency_opt: the frequency of data
                        1 -> hourly 
                        2 -> 3-hourly (only supported)
                        3 -> daily
            num_obj [int]: number of jobs for parallel computation
       Output:
             gmi_fields [ctm_model]: a dataclass format (see config.py)
    '''
    # a nested function
    def gmi_reader_wrapper(fname_met: str, fname_gas: str, gasname: str) -> ctm_model:
        # read the data
        print("Currently reading: " + fname_met.split('/')[-1])
        ctmtype = "GMI"
        # read coordinates
        lon = _read_nc(fname_met, 'lon')
        lat = _read_nc(fname_met, 'lat')
        lons_grid, lats_grid = np.meshgrid(lon, lat)
        latitude = lats_grid
        longitude = lons_grid
        # read time
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
        # read pressure information
        delta_p = _read_nc(fname_met, 'DELP').astype('float32')/100.0
        delta_p = np.flip(delta_p, axis=1)  # from bottom to top
        pressure_mid = _read_nc(fname_met, 'PL').astype('float32')/100.0
        pressure_mid = np.flip(pressure_mid, axis=1)  # from bottom to top
        # read gas concentration
        if (gasname == 'HCHO' or gasname == 'FORM'):
            gasname = 'CH2O'
        temp = np.flip(_read_nc(
            fname_gas, gasname), axis=1)*1e9  # ppbv
        # the purpose of this part is to reduce memory usage
        if gasname != 'O3':
            gas_profile = temp.astype('float16')
        else:
            gas_profile = temp.astype('float32')
        temp = []
        # shape up the ctm class
        gmi_data = ctm_model(latitude, longitude, time, gas_profile,
                             pressure_mid, [], delta_p, ctmtype, False)
        return gmi_data

    if frequency_opt == '3-hourly':
        # read meteorological and chemical fields
        tavg3_3d_met_files = sorted(
            glob.glob(product_dir + "/*tavg3_3d_met_Nv." + str(YYYYMM) + "*.nc4"))
        tavg3_3d_gas_files = sorted(
            glob.glob(product_dir + "/*tavg3_3d_tac_Nv." + str(YYYYMM) + "*.nc4"))
        if len(tavg3_3d_gas_files) != len(tavg3_3d_met_files):
            raise Exception(
                "the data are not consistent")
        # define gas profiles to be saved
        outputs = Parallel(n_jobs=num_job)(delayed(gmi_reader_wrapper)(
            tavg3_3d_met_files[k], tavg3_3d_gas_files[k], gas_to_be_saved) for k in range(len(tavg3_3d_met_files)))
        return outputs


def ECCOH_reader(product_dir: str, YYYYMM: str, gas_to_be_saved: list, num_job=1) -> ctm_model:
    '''
       ECCOH reader
       Inputs:
             product_dir [str]: the folder containing the monthly ECCOH data
             YYYYMM [str]: the target month and year, e.g., 202005 (May 2020)
             gases_to_be_saved [list]: name of gases to be loaded. e.g., ['NO2']
             num_obj [int]: number of jobs for parallel computation
       Output:
             eccoh_fields [ctm_model]: a dataclass format (see config.py)
    '''
    # a nested function
    def eccoh_reader_wrapper(fname: str, gasname: str) -> ctm_model:
        # read the data
        print("Currently reading: " + fname.split('/')[-1])
        ctmtype = "ECCOH"
        # read coordinates
        lon = _read_nc(fname, 'lon')
        lat = _read_nc(fname, 'lat')
        lons_grid, lats_grid = np.meshgrid(lon, lat)
        latitude = lats_grid
        longitude = lons_grid
        # read time
        time_attr = _get_nc_attr(fname, 'time')
        timebegin_date = str(time_attr["begin_date"])
        timebegin_date = [int(timebegin_date[0:4]), int(
            timebegin_date[4:6]), int(timebegin_date[6:8])]

        time = [datetime.datetime(
            timebegin_date[0], timebegin_date[1], timebegin_date[2])]
        # read pressure information
        delta_p = _read_nc(fname, 'DELP').astype('float32')/100.0
        delta_p = np.flip(delta_p, axis=0)  # from bottom to top
        pressure_mid = _read_nc(fname, 'PL').astype('float32')/100.0
        pressure_mid = np.flip(pressure_mid, axis=0)  # from bottom to top
        if gasname == 'H2O': gasname = 'QV'
        # read gas concentration
        temp = np.flip(_read_nc(
            fname, gasname), axis=0)*1e9  # ppbv
        # the purpose of this part is to reduce memory usage
        gas_profile = temp.astype('float32')
        # if CH4 is read, we need to convert it to dry mixing ratio
        if gasname == 'CH4':
           QV = np.flip(_read_nc(fname, 'QV'),axis=0).astype('float32')
           water_vapor_mixing_ratio = QV/(1-QV)
           MW_air = 28.96 #g/mol
           MW_water = 18.015 #g/mol
           gas_profile = gas_profile*(1+water_vapor_mixing_ratio*(MW_air/MW_water))

        temp = []
        # shape up the ctm class
        eccoh_data = ctm_model(latitude, longitude, time, gas_profile,
                               pressure_mid, [], delta_p, ctmtype, False)
        return eccoh_data

    eccoh_files = sorted(
        glob.glob(product_dir + "/*eccoh_Nv." + str(YYYYMM) + "*.nc4"))
    # define gas profiles to be saved
    outputs = Parallel(n_jobs=num_job)(delayed(eccoh_reader_wrapper)(
        eccoh_files[k], gas_to_be_saved) for k in range(len(eccoh_files)))
    return outputs


def tropomi_reader_hcho(fname: str, ctm_models_coordinate=None, read_ak=True) -> satellite_amf:
    '''
       TROPOMI HCHO L2 reader
       Inputs:
             fname [str]: the name path of the L2 file
             ctm_models_coordinate [dict]: a dictionary containing ctm lat and lon
             read_ak [bool]: true for reading averaging kernels. this must be true for amf_recal 
       Output:
             tropomi_hcho [satellite_amf]: a dataclass format (see config.py)
    '''
    # hcho reader
    print("Currently reading: " + fname.split('/')[-1])
    # read time
    time = _read_group_nc(fname, ['PRODUCT'], 'time') +\
        np.nanmean(np.array(_read_group_nc(
            fname, ['PRODUCT'], 'delta_time')), axis=1)/1000.0
    time = np.nanmean(time, axis=0)
    time = np.squeeze(time)
    time = datetime.datetime(
        2010, 1, 1) + datetime.timedelta(seconds=int(time))
    #print(datetime.datetime.strptime(str(tropomi_hcho.time),"%Y-%m-%d %H:%M:%S"))
    # read lat/lon at centers
    latitude_center = _read_group_nc(
        fname, ['PRODUCT'], 'latitude').astype('float32')
    longitude_center = _read_group_nc(
        fname, ['PRODUCT'], 'longitude').astype('float32')
    # read total amf
    amf_total = _read_group_nc(fname, ['PRODUCT', 'SUPPORT_DATA', 'DETAILED_RESULTS'],
                               'formaldehyde_tropospheric_air_mass_factor')
    # read total hcho
    vcd = _read_group_nc(fname, ['PRODUCT'],
                         'formaldehyde_tropospheric_vertical_column')
    scd = _read_group_nc(fname, ['PRODUCT'], 'formaldehyde_tropospheric_vertical_column') *\
        amf_total
    vcd = (vcd*6.02214*1e19*1e-15).astype('float16')
    scd = (scd*6.02214*1e19*1e-15).astype('float16')
    # read quality flag
    quality_flag = _read_group_nc(
        fname, ['PRODUCT'], 'qa_value').astype('float16')
    # read pressures for SWs
    tm5_a = _read_group_nc(
        fname, ['PRODUCT', 'SUPPORT_DATA', 'INPUT_DATA'], 'tm5_constant_a')/100.0
    tm5_b = _read_group_nc(
        fname, ['PRODUCT', 'SUPPORT_DATA', 'INPUT_DATA'], 'tm5_constant_b')
    ps = _read_group_nc(fname, [
                        'PRODUCT', 'SUPPORT_DATA', 'INPUT_DATA'], 'surface_pressure').astype('float16')/100.0
    p_mid = np.zeros(
        (34, np.shape(vcd)[0], np.shape(vcd)[1])).astype('float32')
    if read_ak == True:
        SWs = np.zeros(
            (34, np.shape(vcd)[0], np.shape(vcd)[1])).astype('float16')
        AKs = _read_group_nc(fname, [
            'PRODUCT', 'SUPPORT_DATA', 'DETAILED_RESULTS'], 'averaging_kernel').astype('float16')
    else:
        SWs = np.empty((1))
    # for some reason, in the HCHO product, a and b values are the center instead of the edges (unlike NO2)
    for z in range(0, 34):
        p_mid[z, :, :] = (tm5_a[z]+tm5_b[z]*ps[:, :])
        if read_ak == True:
            SWs[z, :, :] = AKs[:, :, z]*amf_total
    # remove bad SWs
    SWs[np.where((np.isnan(SWs)) | (np.isinf(SWs)) |
                 (SWs > 100.0) | (SWs < 0.0))] = 0.0
    # read the precision
    uncertainty = _read_group_nc(fname, ['PRODUCT'],
                                 'formaldehyde_tropospheric_vertical_column_precision')
    uncertainty = (uncertainty*6.02214*1e19*1e-15).astype('float16')

    tropomi_hcho = satellite_amf(vcd, scd, time, np.empty((1)), latitude_center, longitude_center,
                                [], [], uncertainty, quality_flag, p_mid, SWs, [], [], [], [], [])
    # interpolation
    if (ctm_models_coordinate is not None):
        print('Currently interpolating ...')
        grid_size = 0.10  # degree
        tropomi_hcho = interpolator(
            1, grid_size, tropomi_hcho, ctm_models_coordinate, flag_thresh=0.5)
    # return
    return tropomi_hcho


def tropomi_reader_no2(fname: str, trop: bool, ctm_models_coordinate=None, read_ak=True) -> satellite_amf:
    '''
       TROPOMI NO2 L2 reader
       Inputs:
             fname [str]: the name path of the L2 file
             trop [bool]: true for considering the tropospheric region only
             ctm_models_coordinate [dict]: a dictionary containing ctm lat and lon
             read_ak [bool]: true for reading averaging kernels. this must be true for amf_recal 
       Output:
             tropomi_no2 [satellite_amf]: a dataclass format (see config.py)
    '''
    # say which file is being read
    print("Currently reading: " + fname.split('/')[-1])
    # read time
    time = _read_group_nc(fname, ['PRODUCT'], 'time') +\
        np.nanmean(np.array(_read_group_nc(
            fname, ['PRODUCT'], 'delta_time')), axis=0)/1000.0
    time = np.squeeze(time)
    time = datetime.datetime(
        2010, 1, 1) + datetime.timedelta(seconds=int(time))
    #print(datetime.datetime.strptime(str(tropomi_no2.time),"%Y-%m-%d %H:%M:%S"))
    # read lat/lon at centers
    latitude_center = _read_group_nc(
        fname, ['PRODUCT'], 'latitude').astype('float32')
    longitude_center = _read_group_nc(
        fname, ['PRODUCT'], 'longitude').astype('float32')
    # read total amf
    amf_total = _read_group_nc(fname, ['PRODUCT'], 'air_mass_factor_total')
    # read no2
    if trop == False:
        vcd = _read_group_nc(
            fname, ['PRODUCT', 'SUPPORT_DATA', 'DETAILED_RESULTS'], 'nitrogendioxide_total_column')
        scd = _read_group_nc(
            fname, ['PRODUCT', 'SUPPORT_DATA', 'DETAILED_RESULTS'], 'nitrogendioxide_slant_column_density')
        # read the precision
        uncertainty = _read_group_nc(fname, ['PRODUCT', 'SUPPORT_DATA', 'DETAILED_RESULTS'],
                                     'nitrogendioxide_total_column_precision')
    else:
        vcd = _read_group_nc(
            fname, ['PRODUCT'], 'nitrogendioxide_tropospheric_column')
        scd = vcd*_read_group_nc(
            fname, ['PRODUCT'], 'air_mass_factor_troposphere')
        # read the precision
        uncertainty = _read_group_nc(fname, ['PRODUCT'],
                                     'nitrogendioxide_tropospheric_column_precision')
    vcd = (vcd*6.02214*1e19*1e-15).astype('float16')
    scd = (scd*6.02214*1e19*1e-15).astype('float16')
    uncertainty = (uncertainty*6.02214*1e19*1e-15).astype('float16')
    # read quality flag
    quality_flag = _read_group_nc(
        fname, ['PRODUCT'], 'qa_value').astype('float16')
    # read pressures for SWs
    tm5_a = _read_group_nc(fname, ['PRODUCT'], 'tm5_constant_a')/100.0
    tm5_a = np.concatenate((tm5_a[:, 0], 0), axis=None)
    tm5_b = _read_group_nc(fname, ['PRODUCT'], 'tm5_constant_b')
    tm5_b = np.concatenate((tm5_b[:, 0], 0), axis=None)

    ps = _read_group_nc(fname, [
                        'PRODUCT', 'SUPPORT_DATA', 'INPUT_DATA'], 'surface_pressure').astype('float32')/100.0
    p_mid = np.zeros(
        (34, np.shape(vcd)[0], np.shape(vcd)[1])).astype('float16')
    if read_ak == True:
        SWs = np.zeros(
            (34, np.shape(vcd)[0], np.shape(vcd)[1])).astype('float16')
        AKs = _read_group_nc(fname, ['PRODUCT'],
                             'averaging_kernel').astype('float16')
    else:
        SWs = np.empty((1))
    for z in range(0, 34):
        p_mid[z, :, :] = 0.5*(tm5_a[z]+tm5_b[z]*ps[:, :] +
                              tm5_a[z+1]+tm5_b[z+1]*ps[:, :])
        if read_ak == True:
            SWs[z, :, :] = AKs[:, :, z]*amf_total
    # remove bad SWs
    SWs[np.where((np.isnan(SWs)) | (np.isinf(SWs)) |
                 (SWs > 100.0) | (SWs < 0.0))] = 0.0
    # read the tropopause layer index
    if trop == True:
        trop_layer = _read_group_nc(
            fname, ['PRODUCT'], 'tm5_tropopause_layer_index')
        tropopause = np.zeros_like(trop_layer).astype('float16')
        for i in range(0, np.shape(trop_layer)[0]):
            for j in range(0, np.shape(trop_layer)[1]):
                if (trop_layer[i, j] > 0 and trop_layer[i, j] < 34):
                    tropopause[i, j] = p_mid[trop_layer[i, j], i, j]
                else:
                    tropopause[i, j] = np.nan
    else:
        tropopause = np.empty((1))
    tropomi_no2 = satellite_amf(vcd, scd, time, tropopause, latitude_center, longitude_center,
                                [], [], uncertainty, quality_flag, p_mid, SWs, [], [], [], [], [])
    # interpolation
    if (ctm_models_coordinate is not None):
        print('Currently interpolating ...')
        grid_size = 0.10  # degree
        tropomi_no2 = interpolator(
            1, grid_size, tropomi_no2, ctm_models_coordinate, flag_thresh=0.75)
    # return
    return tropomi_no2


def omi_reader_no2(fname: str, trop: bool, ctm_models_coordinate=None, read_ak=True) -> satellite_amf:
    '''
       OMI NO2 L2 reader
       Inputs:
             fname [str]: the name path of the L2 file
             trop [bool]: true for considering the tropospheric region only
             ctm_models_coordinate [dict]: a dictionary containing ctm lat and lon
             read_ak [bool]: true for reading averaging kernels. this must be true for amf_recal 
       Output:
             omi_no2 [satellite_amf]: a dataclass format (see config.py)
    '''
    # say which file is being read
    print("Currently reading: " + fname.split('/')[-1])
    # read time
    time = _read_group_nc(fname, ['GEOLOCATION_DATA'], 'Time')
    time = np.squeeze(np.nanmean(time))
    time = datetime.datetime(
        1993, 1, 1) + datetime.timedelta(seconds=int(time))
    #print(datetime.datetime.strptime(str(time),"%Y-%m-%d %H:%M:%S"))
    # read lat/lon at centers
    latitude_center = _read_group_nc(
        fname, ['GEOLOCATION_DATA'], 'Latitude').astype('float32')
    longitude_center = _read_group_nc(
        fname, ['GEOLOCATION_DATA'], 'Longitude').astype('float32')
    # read no2
    if trop == False:
        vcd = _read_group_nc(
            fname, ['SCIENCE_DATA'], 'ColumnAmountNO2')
        scd = _read_group_nc(fname, ['SCIENCE_DATA'], 'AmfTrop') *\
            _read_group_nc(fname, ['SCIENCE_DATA'], 'ColumnAmountNO2Trop') +\
            _read_group_nc(fname, ['SCIENCE_DATA'], 'AmfStrat') *\
            _read_group_nc(fname, ['SCIENCE_DATA'], 'ColumnAmountNO2Strat')
        # read the precision
        uncertainty = _read_group_nc(fname, ['SCIENCE_DATA'],
                                     'ColumnAmountNO2Std')
    else:
        vcd = _read_group_nc(
            fname, ['SCIENCE_DATA'], 'ColumnAmountNO2Trop')
        scd = _read_group_nc(fname, ['SCIENCE_DATA'], 'AmfTrop') *\
            _read_group_nc(fname, ['SCIENCE_DATA'], 'ColumnAmountNO2Trop')
        # read the precision
        uncertainty = _read_group_nc(fname, ['SCIENCE_DATA'],
                                     'ColumnAmountNO2TropStd')
    vcd = (vcd*1e-15).astype('float16')
    scd = (scd*1e-15).astype('float16')
    uncertainty = (uncertainty*1e-15).astype('float16')
    # read quality flag
    cf_fraction = quality_flag_temp = _read_group_nc(
        fname, ['ANCILLARY_DATA'], 'CloudFraction').astype('float16')
    cf_fraction_mask = cf_fraction < 0.3
    cf_fraction_mask = np.multiply(cf_fraction_mask, 1.0).squeeze()

    train_ref = quality_flag_temp = _read_group_nc(
        fname, ['ANCILLARY_DATA'], 'TerrainReflectivity').astype('float16')
    train_ref_mask = train_ref < 0.2
    train_ref_mask = np.multiply(train_ref_mask, 1.0).squeeze()

    quality_flag_temp = _read_group_nc(
        fname, ['SCIENCE_DATA'], 'VcdQualityFlags').astype('float16')
    quality_flag = np.zeros_like(quality_flag_temp)*-100.0
    for i in range(0, np.shape(quality_flag)[0]):
        for j in range(0, np.shape(quality_flag)[1]):
            flag = '{0:08b}'.format(int(quality_flag_temp[i, j]))
            if flag[-1] == '0':
                quality_flag[i, j] = 1.0
            if flag[-1] == '1':
                if flag[-2] == '0':
                    quality_flag[i, j] = 1.0
    quality_flag = quality_flag*cf_fraction_mask*train_ref_mask
    # read pressures for SWs
    ps = _read_group_nc(fname, ['GEOLOCATION_DATA'],
                        'ScatteringWeightPressure').astype('float16')
    p_mid = np.zeros(
        (35, np.shape(vcd)[0], np.shape(vcd)[1])).astype('float16')
    if read_ak == True:
        SWs = _read_group_nc(fname, ['SCIENCE_DATA'],
                             'ScatteringWeight').astype('float16')
        SWs = SWs.transpose((2, 0, 1))
    else:
        SWs = np.empty((1))
    for z in range(0, 35):
        p_mid[z, :, :] = ps[z]
    # remove bad SWs
    SWs[np.where((np.isnan(SWs)) | (np.isinf(SWs)) |
                 (SWs > 100.0) | (SWs < 0.0))] = 0.0
    # read the tropopause pressure
    if trop == True:
        tropopause = _read_group_nc(
            fname, ['ANCILLARY_DATA'], 'TropopausePressure').astype('float16')
    else:
        tropopause = np.empty((1))
    # populate omi class
    omi_no2 = satellite_amf(vcd, scd, time, tropopause, latitude_center,
                            longitude_center, [], [], uncertainty, quality_flag, p_mid, SWs, [], [], [], [], [])
    # interpolation
    if (ctm_models_coordinate is not None):
        print('Currently interpolating ...')
        grid_size = 0.25  # degree
        omi_no2 = interpolator(
            1, grid_size, omi_no2, ctm_models_coordinate, flag_thresh=0.0)
    # return
    return omi_no2


def omi_reader_hcho(fname: str, ctm_models_coordinate=None, read_ak=True) -> satellite_amf:
    '''
       OMI HCHO L2 reader
       Inputs:
             fname [str]: the name path of the L2 file
             ctm_models_coordinate [dict]: a dictionary containing ctm lat and lon
             read_ak [bool]: true for reading averaging kernels. this must be true for amf_recal 
       Output:
             omi_hcho [satellite_amf]: a dataclass format (see config.py)
    '''
    # we add "try" because some files have format issue thus unreadable
    try:
        # say which file is being read
        print("Currently reading: " + fname.split('/')[-1])
        # read time
        time = _read_group_nc(fname, ['geolocation'], 'time')
        time = np.squeeze(np.nanmean(time))
        time = datetime.datetime(
            1993, 1, 1) + datetime.timedelta(seconds=int(time))
        # read lat/lon at centers
        latitude_center = _read_group_nc(
            fname, ['geolocation'], 'latitude').astype('float32')
        longitude_center = _read_group_nc(
            fname, ['geolocation'], 'longitude').astype('float32')
        # read hcho
        vcd = _read_group_nc(
            fname, ['key_science_data'], 'column_amount')
        scd = _read_group_nc(fname, ['support_data'], 'amf') *\
            _read_group_nc(fname, ['key_science_data'], 'column_amount')
        # read the precision
        uncertainty = _read_group_nc(fname, ['key_science_data'],
                                     'column_uncertainty')
        vcd = (vcd*1e-15).astype('float16')
        scd = (scd*1e-15).astype('float16')
        uncertainty = (uncertainty*1e-15).astype('float16')
        # read quality flag
        cf_fraction = _read_group_nc(
            fname, ['support_data'], 'cloud_fraction').astype('float16')
        cf_fraction_mask = cf_fraction < 0.4
        cf_fraction_mask = np.multiply(cf_fraction_mask, 1.0).squeeze()

        quality_flag = _read_group_nc(
            fname, ['key_science_data'], 'main_data_quality_flag').astype('float16')
        quality_flag = quality_flag == 0.0
        quality_flag = np.multiply(quality_flag, 1.0).squeeze()

        quality_flag = quality_flag*cf_fraction_mask
        # read pressures for SWs
        ps = _read_group_nc(fname, ['support_data'],
                            'surface_pressure').astype('float16')
        a0 = np.array([0., 0.04804826, 6.593752, 13.1348, 19.61311, 26.09201, 32.57081, 38.98201, 45.33901, 51.69611, 58.05321, 64.36264, 70.62198, 78.83422, 89.09992, 99.36521, 109.1817, 118.9586, 128.6959, 142.91, 156.26, 169.609, 181.619,
                       193.097, 203.259, 212.15, 218.776, 223.898, 224.363, 216.865, 201.192, 176.93, 150.393, 127.837, 108.663, 92.36572, 78.51231, 56.38791, 40.17541, 28.36781, 19.7916, 9.292942, 4.076571, 1.65079, 0.6167791, 0.211349, 0.06600001, 0.01])
        b0 = np.array([1., 0.984952, 0.963406, 0.941865, 0.920387, 0.898908, 0.877429, 0.856018, 0.8346609, 0.8133039, 0.7919469, 0.7706375, 0.7493782, 0.721166, 0.6858999, 0.6506349, 0.6158184, 0.5810415, 0.5463042,
                       0.4945902, 0.4437402, 0.3928911, 0.3433811, 0.2944031, 0.2467411, 0.2003501, 0.1562241, 0.1136021, 0.06372006, 0.02801004, 0.006960025, 8.175413e-09, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        p_mid = np.zeros(
            (np.size(a0)-1, np.shape(vcd)[0], np.shape(vcd)[1])).astype('float16')
        if read_ak == True:
            SWs = _read_group_nc(fname, ['support_data'],
                                 'scattering_weights').astype('float16')
        else:
            SWs = np.empty((1))
        for z in range(0, np.size(a0)-1):
            p_mid[z, :, :] = 0.5*((a0[z] + b0[z]*ps) + (a0[z+1] + b0[z+1]*ps))
        # remove bad SWs
        SWs[np.where((np.isnan(SWs)) | (np.isinf(SWs)) |
                     (SWs > 100.0) | (SWs < 0.0))] = 0.0
        # no need to read tropopause for hCHO
        tropopause = np.empty((1))
        # populate omi class
        omi_hcho = satellite_amf(vcd, scd, time, tropopause, latitude_center,
                                 longitude_center, [], [], uncertainty, quality_flag, p_mid, SWs, [], [], [], [], [])
        # interpolation
        if (ctm_models_coordinate is not None):
            print('Currently interpolating ...')
            grid_size = 0.25  # degree
            omi_hcho = interpolator(
                1, grid_size, omi_hcho, ctm_models_coordinate, flag_thresh=0.0)
        # return
        return omi_hcho
    except:
        return None


def omi_reader_o3(fname: str, ctm_models_coordinate=None, read_ak=True) -> satellite_amf:
    '''
       OMI total ozone L2 reader
       Inputs:
             fname [str]: the name path of the L2 file
             ctm_models_coordinate [dict]: a dictionary containing ctm lat and lon
             read_ak [bool]: true for reading averaging kernels. this must be true for amf_recal 
       Output:
             omi_hcho [satellite_amf]: a dataclass format (see config.py)
    '''

    # say which file is being read
    print("Currently reading: " + fname.split('/')[-1])
    # read timeHDFEOS/SWATHS/OMI Column Amount O3/Geolocation Fields/Latitude
    time = _read_group_nc(fname, ['HDFEOS', 'SWATHS',
                                  'OMI Column Amount O3', 'Geolocation Fields'], 'Time')
    time = np.squeeze(np.nanmean(time))
    time = datetime.datetime(
        1993, 1, 1) + datetime.timedelta(seconds=int(time))
    # read lat/lon at centers
    latitude_center = _read_group_nc(
        fname, ['HDFEOS', 'SWATHS',
                'OMI Column Amount O3', 'Geolocation Fields'], 'Latitude').astype('float32')
    longitude_center = _read_group_nc(
        fname, ['HDFEOS', 'SWATHS',
                'OMI Column Amount O3', 'Geolocation Fields'], 'Longitude').astype('float32')

    SZA = _read_group_nc(
        fname, ['HDFEOS', 'SWATHS',
                'OMI Column Amount O3', 'Geolocation Fields'], 'SolarZenithAngle').astype('float32')
    # read hcho
    vcd = _read_group_nc(
        fname, ['HDFEOS', 'SWATHS',
                'OMI Column Amount O3', 'Data Fields'], 'ColumnAmountO3')
    vcd[np.where((vcd <= 0) | (np.isinf(vcd)) | (SZA > 80.0))] = np.nan
    vcd = (vcd).astype('float16')
    # read quality flag
    quality_flag_temp = _read_group_nc(
        fname, ['HDFEOS', 'SWATHS',
                'OMI Column Amount O3', 'Data Fields'], 'QualityFlags').astype('float16')
    quality_flag = np.zeros_like(quality_flag_temp)*-100.0
    for i in range(0, np.shape(quality_flag)[0]):
        for j in range(0, np.shape(quality_flag)[1]):
            flag = '{0:08b}'.format(int(quality_flag_temp[i, j]))
            if flag[-1] == '0':
                quality_flag[i, j] = 1.0

    # 4 percent error based on several studies
    uncertainty = (vcd*0.04).astype('float16')

    # no need to read tropopause for total O3
    tropopause = np.empty((1))
    SWs = np.empty((1))
    # populate omi class
    omi_o3 = satellite_amf(vcd, vcd, time, tropopause, latitude_center,
                           longitude_center, [], [], uncertainty, quality_flag, [],  SWs, [], [], [], [], [])
    # interpolation
    if (ctm_models_coordinate is not None):
        print('Currently interpolating ...')
        grid_size = 0.25  # degree
        omi_o3 = interpolator(
            1, grid_size, omi_o3, ctm_models_coordinate, flag_thresh=0.0)
    # return
    return omi_o3


def mopitt_reader_co(fname: str, ctm_models_coordinate=None, read_ak=True) -> satellite_opt:
    '''
       MOPITT CO L3 reader
       Inputs:
             fname [str]: the name path of the L2 file
             ctm_models_coordinate [dict]: a dictionary containing ctm lat and lon
             read_ak [bool]: true for reading averaging kernels. this must be true for amf_recal 
       Output:
             mopitt_co [satellite_opt]: a dataclass format (see config.py)
    '''
    # say which file is being read
    print("Currently reading: " + fname.split('/')[-1])
    # read timeHDFEOS/SWATHS/OMI Column Amount O3/Geolocation Fields/Latitude
    attr_mopitt = _get_nc_attr_group_mopitt(fname)
    StartTime = (attr_mopitt["StartTime"])
    EndTime = (attr_mopitt["StopTime"])

    time = 0.5*(StartTime+EndTime)
    time = datetime.datetime(
        1993, 1, 1) + datetime.timedelta(seconds=int(time))
    # read lat/lon at centers
    latitude_center = _read_group_nc(
        fname, ['HDFEOS', 'GRIDS', 'MOP03',
                'Data Fields'], 'Latitude').astype('float32')
    longitude_center = _read_group_nc(
        fname, ['HDFEOS', 'GRIDS', 'MOP03',
                'Data Fields'], 'Longitude').astype('float32')
    longitude_center, latitude_center = np.meshgrid(
        longitude_center, latitude_center)
    longitude_center = np.transpose(longitude_center)
    latitude_center = np.transpose(latitude_center)
    # read total CO
    vcd = _read_group_nc(fname, ['HDFEOS', 'GRIDS', 'MOP03',
                                 'Data Fields'], 'RetrievedCOTotalColumnDay')
    vcd[np.where((vcd <= 0) | (np.isinf(vcd)))] = np.nan
    vcd = (vcd*1e-15).astype('float16')
    dryair_col = _read_group_nc(fname, ['HDFEOS', 'GRIDS', 'MOP03',
                                        'Data Fields'], 'DryAirColumnDay')
    x_col = (1e6*vcd/(dryair_col*1e-15)).astype('float32')
    apriori_profile = _read_group_nc(fname, ['HDFEOS', 'GRIDS', 'MOP03',
                                             'Data Fields'], 'APrioriCOMixingRatioProfileDay').transpose((2, 0, 1))
    apriori_profile[apriori_profile <= 0] = np.nan
    apriori_surface = _read_group_nc(fname, ['HDFEOS', 'GRIDS', 'MOP03',
                                             'Data Fields'], 'APrioriCOSurfaceMixingRatioDay')
    surface_pressure = _read_group_nc(fname, ['HDFEOS', 'GRIDS', 'MOP03',
                                              'Data Fields'], 'SurfacePressureDay')
    apriori_surface[apriori_surface <= 0] = np.nan
    apriori_col = _read_group_nc(fname, ['HDFEOS', 'GRIDS', 'MOP03',
                                         'Data Fields'], 'APrioriCOTotalColumnDay')
    apriori_col = (apriori_col*1e-15).astype('float16')
    apriori_col[apriori_col <= 0] = np.nan
    # read quality flag
    uncertainty = _read_group_nc(fname, ['HDFEOS', 'GRIDS', 'MOP03',
                                         'Data Fields'], 'RetrievedCOTotalColumnMeanUncertaintyDay')
    uncertainty = (uncertainty*1e-15).astype('float32')
    # no need to read tropopause for total CO
    tropopause = np.empty((1))
    # read pressures for AKs
    ps = _read_group_nc(fname, ['HDFEOS', 'GRIDS', 'MOP03',
                                'Data Fields'], 'Pressure').astype('float16')
    p_mid = np.zeros(
        (9, np.shape(vcd)[0], np.shape(vcd)[1])).astype('float16')
    if read_ak == True:
        AKs = _read_group_nc(fname, ['HDFEOS', 'GRIDS', 'MOP03',
                                     'Data Fields'], 'TotalColumnAveragingKernelDay')*1e-15
        AKs = AKs.transpose((2, 0, 1)).astype('float16')
    else:
        AKs = np.empty((1))
    for z in range(0, 9):
        p_mid[z, :, :] = ps[z]

    # populate mopitt class
    mopitt = satellite_opt(vcd, time, [], tropopause, latitude_center,
                           longitude_center, [], [], uncertainty, np.ones_like(
                               vcd), p_mid, AKs, [], [], [], [],
                           apriori_col, apriori_profile, surface_pressure, apriori_surface, x_col, [], 'MOPITT')
    # interpolation
    if (ctm_models_coordinate is not None):
        print('Currently interpolating ...')
        grid_size = 1.0  # degree
        mopitt = interpolator(
            1, grid_size, mopitt, ctm_models_coordinate, flag_thresh=0.0)
    # return
    return mopitt


def gosat_reader_xch4(fname: str, ctm_models_coordinate=None, read_ak=True) -> satellite_opt:
    '''
       GOSAT L2 reader
       Inputs:
             fname [str]: the name path of the L2 file
             ctm_models_coordinate [dict]: a dictionary containing ctm lat and lon
             read_ak [bool]: true for reading averaging kernels. this must be true for amf_recal 
       Output:
             mopitt_co [satellite_opt]: a dataclass format (see config.py)
    '''
    # say which file is being read
    print("Currently reading: " + fname.split('/')[-1])
    time = _read_nc(fname, 'time')
    time = np.squeeze(np.nanmean(time))
    time = datetime.datetime(
        1970, 1, 1) + datetime.timedelta(seconds=int(time))
    # read lat/lon at centers
    latitude_center = _read_nc(
        fname, 'latitude').astype('float32')
    longitude_center = _read_nc(
        fname, 'longitude').astype('float32')
    # read xch4
    xch4 = _read_nc(fname, 'xch4')
    xch4[np.where((xch4 <= 0) | (np.isinf(xch4)))] = np.nan

    apriori_profile = _read_nc(fname, 'ch4_profile_apriori').transpose()
    apriori_profile[apriori_profile <= 0] = np.nan
    # read quality flag
    quality_flag = _read_nc(fname, 'xch4_quality_flag')
    uncertainty = _read_nc(fname, 'xch4_uncertainty')
    # no need to read tropopause 
    tropopause = np.empty((1))
    # read pressures for AKs
    p_mid = _read_nc(fname, 'pressure_levels')
    p_mid[p_mid<=0] = np.nan
    if read_ak == True:
        AKs = _read_nc(fname, 'xch4_averaging_kernel')
        AKs = AKs.transpose()
        PW  = _read_nc(fname, 'pressure_weight')
        PW = PW.transpose()
        AKs[AKs<=0] = np.nan
        PW[PW<=0] = np.nan
    else:
        AKs = np.empty((1))
        PW  = np.empty((1))

    p_mid = np.transpose(p_mid)
    # populate gosat class
    gosat = satellite_opt(xch4, time, [], tropopause, latitude_center,
                           longitude_center, [], [], uncertainty, 1-quality_flag, p_mid, AKs, [], [], [], [], np.empty((1)), apriori_profile, np.empty((1)), np.empty((1)), xch4, PW, 'GOSAT')
    # since gosat does image the earth, we need to convert the points to gridded maps using filler_gosat.py
    gosat = filler_gosatxch4(1.0, gosat, flag_thresh=0.0)
    # interpolation
    if (ctm_models_coordinate is not None):
        print('Currently interpolating ...')
        grid_size = 1.0  # degree
        gosat = interpolator(
            1, grid_size, gosat, ctm_models_coordinate, flag_thresh=0.0)
    # return
    return gosat

def ssmis_reader_wv(fname: str, ctm_models_coordinate=None) -> satellite_ssmis:
    # say which file is being read
    print("Currently reading: " + fname.split('/')[-1])
    date = fname.split('v7')[-6:-1]
    date = str(date[0])
    time = datetime.datetime(
        int(date[-6:-2]), int(date[-2::]), 1)
    # read lat/lon at centers
    latitude_center = _read_nc(
        fname, 'latitude').astype('float32')
    longitude_center = _read_nc(
        fname, 'longitude').astype('float32')-180.0
    longitude_center,latitude_center = np.meshgrid(longitude_center,latitude_center)
    # read xch4
    pwv = _read_ssmi(fname, 'atmosphere_water_vapor_content').astype('float32')
    pwv[pwv>250.0] = np.nan
    pwv = pwv*0.3
    pwv[np.where((pwv >= 75.0) | (np.isinf(pwv)))] = np.nan

    ssmis = satellite_ssmis(pwv,pwv*0.05,time,latitude_center,longitude_center,False,[],'SSMI')
    # interpolation
    if (ctm_models_coordinate is not None):
        print('Currently interpolating ...')
        grid_size = 0.25  # degree
        ssmis = interpolator_ssmis(
            1, grid_size, ssmis, ctm_models_coordinate)
    # return
    return ssmis

def tropomi_reader(product_dir: str, satellite_product_name: str, ctm_models_coordinate: dict, YYYYMM: str, trop: bool, read_ak=True, num_job=1):
    '''
        reading tropomi data
             product_dir [str]: the folder containing the tropomi data
             satellite_product_name [str]: so far we support:
                                         "NO2"
                                         "HCHO"
             ctm_models_coordinate [dict]: the ctm coordinates
             YYYYMM [int]: the target month and year, e.g., 202005 (May 2020)
             trop [bool]: true for considering the tropospheric region only
             read_ak [bool]: true for reading averaging kernels. this must be true for amf_recal
             num_job [int]: the number of jobs for parallel computation
        Output [tropomi]: the tropomi @dataclass
    '''

    # find L2 files first
    L2_files = sorted(glob.glob(product_dir + "/S5P_*" + "_L2__*___" + str(YYYYMM) + "*.nc"))
    L2_files = _remove_empty_files(L2_files)
    # read the files in parallel
    if satellite_product_name.split('_')[-1] == 'NO2':
        outputs_sat = Parallel(n_jobs=num_job)(delayed(tropomi_reader_no2)(
            L2_files[k], trop, ctm_models_coordinate=ctm_models_coordinate, read_ak=read_ak) for k in range(len(L2_files)))
    elif satellite_product_name.split('_')[-1] == 'HCHO':
        outputs_sat = Parallel(n_jobs=num_job)(delayed(tropomi_reader_hcho)(
            L2_files[k], ctm_models_coordinate=ctm_models_coordinate, read_ak=read_ak) for k in range(len(L2_files)))
    return outputs_sat


def omi_reader(product_dir: str, satellite_product_name: str, ctm_models_coordinate: dict, YYYYMM: str, trop: bool, read_ak=True, num_job=1):
    '''
        reading omi data
             product_dir [str]: the folder containing the tropomi data
             satellite_product_name [str]: so far we support:
                                         "NO2"
                                         "HCHO"
                                         "O3"
             ctm_models_coordinate [dict]: the ctm coordinates
             YYYYMM [int]: the target month and year, e.g., 202005 (May 2020)
             trop [bool]: true for considering the tropospheric region only
             read_ak [bool]: true for reading averaging kernels. this must be true for amf_recal
             num_job [int]: the number of jobs for parallel computation
        Output [tropomi]: the tropomi @dataclass
    '''

    # find L2 files first
    if satellite_product_name.split('_')[-1] != 'O3':
        print(product_dir + "/*" + YYYYMM[0:4] + 'm' + YYYYMM[4::] + "*.nc")
        L2_files = sorted(glob.glob(product_dir + "/*" +
                                    YYYYMM[0:4] + 'm' + YYYYMM[4::] + "*.nc"))
    else:
        print(product_dir + "/*" + YYYYMM[0:4] + 'm' + YYYYMM[4::] + "*.he5")
        L2_files = sorted(glob.glob(product_dir + "/*" +
                                    YYYYMM[0:4] + 'm' + YYYYMM[4::] + "*.he5"))
    L2_files = _remove_empty_files(L2_files)
    # read the files in parallel
    if satellite_product_name.split('_')[-1] == 'NO2':
        outputs_sat = Parallel(n_jobs=num_job)(delayed(omi_reader_no2)(
            L2_files[k], trop, ctm_models_coordinate=ctm_models_coordinate, read_ak=read_ak) for k in range(len(L2_files)))
    elif satellite_product_name.split('_')[-1] == 'HCHO':
        outputs_sat = Parallel(n_jobs=num_job)(delayed(omi_reader_hcho)(
            L2_files[k], ctm_models_coordinate=ctm_models_coordinate, read_ak=read_ak) for k in range(len(L2_files)))
    elif satellite_product_name.split('_')[-1] == 'O3':
        outputs_sat = Parallel(n_jobs=num_job)(delayed(omi_reader_o3)(
            L2_files[k], ctm_models_coordinate=ctm_models_coordinate, read_ak=read_ak) for k in range(len(L2_files)))
    return outputs_sat


def mopitt_reader(product_dir: str, ctm_models_coordinate: dict, YYYYMM: str, read_ak=True, num_job=1):
    '''
        reading mopitt data
             product_dir [str]: the folder containing the tropomi data
             ctm_models_coordinate [dict]: the ctm coordinates
             YYYYMM [int]: the target month and year, e.g., 202005 (May 2020)
             read_ak [bool]: true for reading averaging kernels. this must be true for amf_recal
             num_job [int]: the number of jobs for parallel computation
        Output [mopitt]: the mopitt @dataclass
    '''
    L3_files = sorted(glob.glob(product_dir + "/*" +
                                YYYYMM[0:4] + YYYYMM[4::] + "*.he5"))
    L3_files = _remove_empty_files(L3_files)
    outputs_sat = Parallel(n_jobs=num_job)(delayed(mopitt_reader_co)(
        L3_files[k], ctm_models_coordinate=ctm_models_coordinate, read_ak=read_ak) for k in range(len(L3_files)))
    return outputs_sat


def gosat_reader(product_dir: str, ctm_models_coordinate: dict, YYYYMM: str, read_ak=True, num_job=1):
    '''
        reading mopitt data
             product_dir [str]: the folder containing the tropomi data
             ctm_models_coordinate [dict]: the ctm coordinates
             YYYYMM [int]: the target month and year, e.g., 202005 (May 2020)
             read_ak [bool]: true for reading averaging kernels. this must be true for amf_recal
             num_job [int]: the number of jobs for parallel computation
        Output [gosat]: the mopitt @dataclass
    '''
    L2_files = sorted(glob.glob(product_dir + "/" + YYYYMM[0:4] + "/*" +
                                YYYYMM[0:4] + YYYYMM[4::] + "*.nc"))
    outputs_sat = Parallel(n_jobs=num_job)(delayed(gosat_reader_xch4)(
        L2_files[k], ctm_models_coordinate=ctm_models_coordinate, read_ak=read_ak) for k in range(len(L2_files)))
    return outputs_sat

def ssmis_reader(product_dir: str, ctm_models_coordinate: dict, YYYYMM: str, num_job=1):
    '''
        reading ssmi water vapor column data
             product_dir [str]: the folder containing the tropomi data
             ctm_models_coordinate [dict]: the ctm coordinates
             YYYYMM [int]: the target month and year, e.g., 202005 (May 2020)
             num_job [int]: the number of jobs for parallel computation
        Output [ssmis]: the ssmi @dataclass
    '''
    L3_files = sorted(glob.glob(product_dir + "/*" +
                                YYYYMM[0:4] + YYYYMM[4::] + "*.nc"))
    L3_files = _remove_empty_files(L3_files)
    outputs_sat = Parallel(n_jobs=num_job)(delayed(ssmis_reader_wv)(
        L3_files[k], ctm_models_coordinate=ctm_models_coordinate) for k in range(len(L3_files)))
    return outputs_sat

class readers(object):

    def __init__(self) -> None:
        pass

    def add_satellite_data(self, product_name: str, product_dir: Path):
        '''
            add L2 data
            Input:
                product_name [str]: a string specifying the type of data to read:
                                   TROPOMI_NO2
                                   TROPOMI_HCHO
                                   TROPOMI_CH4
                                   TROPOMI_CO
                                   OMI_NO2
                                   OMI_HCHO
                                   OMI_O3
                                   MOPITT
                                   GOSAT
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
                                "ECCOH"
                product_dir  [Path]: a path object describing the path of CTM files
        '''

        self.ctm_product_dir = product_dir
        self.ctm_product = product_name

    def read_satellite_data(self, YYYYMM: str, read_ak=True, trop=False, num_job=1):
        '''
            read L2 satellite data
            Input:
             YYYYMM [str]: the target month and year, e.g., 202005 (May 2020)
             read_ak [bool]: true for reading averaging kernels. this must be true for amf_recal
             trop[bool]: true for only including the tropospheric region (relevant for NO2 only)
             num_job [int]: the number of jobs for parallel computation
        '''
        satellite = self.satellite_product_name.split('_')[0]
        ctm_models_coordinate = {}
        ctm_models_coordinate["Latitude"] = self.ctm_data[0].latitude
        ctm_models_coordinate["Longitude"] = self.ctm_data[0].longitude
        if satellite == 'TROPOMI':
            self.sat_data = tropomi_reader(self.satellite_product_dir.as_posix(),
                                           self.satellite_product_name, ctm_models_coordinate,
                                           YYYYMM,  trop, read_ak=read_ak, num_job=num_job)
        elif satellite == 'OMI':
            self.sat_data = omi_reader(self.satellite_product_dir.as_posix(),
                                       self.satellite_product_name, ctm_models_coordinate,
                                       YYYYMM,  trop, read_ak=read_ak, num_job=num_job)
        elif satellite == 'MOPITT':
            self.sat_data = mopitt_reader(self.satellite_product_dir.as_posix(),
                                          ctm_models_coordinate,
                                          YYYYMM, read_ak=read_ak, num_job=num_job)
        elif satellite == 'GOSAT':
            self.sat_data = gosat_reader(self.satellite_product_dir.as_posix(),
                                         ctm_models_coordinate,
                                         YYYYMM, read_ak=read_ak, num_job=num_job)
        elif satellite == 'SSMIS':
            self.sat_data = ssmis_reader(self.satellite_product_dir.as_posix(),
                                         ctm_models_coordinate,
                                         YYYYMM, num_job=num_job)
        else:
            raise Exception("the satellite is not supported, come tomorrow!")

    def read_ctm_data(self, YYYYMM: str, gas: str, frequency_opt: str, averaged=False, num_job=1):
        '''
            read ctm data
            Input:
             YYYYMM [str]: the target month and year, e.g., 202005 (May 2020)
             gases_to_be_saved [str]: name of the gas to be loaded. e.g., 'NO2'
             frequency_opt: the frequency of data
                        1 -> hourly 
                        2 -> 3-hourly
                        3 -> daily
             num_job [int]: the number of jobs for parallel computation
        '''
        if self.ctm_product == 'GMI':
            ctm_data = GMI_reader(self.ctm_product_dir.as_posix(), YYYYMM, gas,
                                  frequency_opt=frequency_opt, num_job=num_job)
            if averaged == True:
                # constant variables
                print("Averaging CTM files ...")
                latitude = ctm_data[0].latitude
                longitude = ctm_data[0].longitude
                time = ctm_data[0].time
                ctm_type = 'GMI'
                # averaging over variable things
                gas_profile = []
                pressure_mid = []
                delta_p = []
                for ctm in ctm_data:
                    gas_profile.append(ctm.gas_profile)
                    pressure_mid.append(ctm.pressure_mid)
                    delta_p.append(ctm.delta_p)

                gas_profile = np.nanmean(np.array(gas_profile), axis=0)
                pressure_mid = np.nanmean(np.array(pressure_mid), axis=0)
                delta_p = np.nanmean(np.array(delta_p), axis=0)
                # shape up the ctm class
                self.ctm_data = []
                self.ctm_data.append(ctm_model(latitude, longitude, time, gas_profile,
                                               pressure_mid, [], delta_p, ctm_type, True))
                ctm_data = []
            else:
                self.ctm_data = ctm_data
                ctm_data = []
        if self.ctm_product == 'ECCOH':
            self.ctm_data = ECCOH_reader(
                self.ctm_product_dir.as_posix(), YYYYMM, gas, num_job=num_job)
        if self.ctm_product == 'FREE':
            # Read the control file
            with open('oisatgmi/control_free.yml', 'r') as stream:
                try:
                    ctrl_opts = yaml.safe_load(stream)
                except yaml.YAMLError as exc:
                    raise Exception(exc)
            ctm_type = 'FREE'
            lat1 = ctrl_opts['latll']
            lat2 = ctrl_opts['latur']
            lon1 = ctrl_opts['lonll']
            lon2 = ctrl_opts['lonur']
            gridsize = ctrl_opts['gridsize']
            lon_grid = np.arange(lon1, lon2+gridsize, gridsize)
            lat_grid = np.arange(lat1, lat2+gridsize, gridsize)
            lons_grid, lats_grid = np.meshgrid(lon_grid.astype('float16'), lat_grid.astype('float16'))
            self.ctm_data = []
            time = []
            time.append(datetime.datetime(1989, 1, 16))  # my birthday
            gas_profile = np.zeros(
                (10, np.shape(lats_grid)[0], np.shape(lons_grid)[1]))*np.nan
            delta_p = np.zeros(
                (10, np.shape(lats_grid)[0], np.shape(lons_grid)[1]))*np.nan
            pressure_mid = np.zeros(
                (10, np.shape(lats_grid)[0], np.shape(lons_grid)[1]))*np.nan
            self.ctm_data.append(ctm_model(lats_grid, lons_grid, time, gas_profile,
                                           pressure_mid, [], delta_p, ctm_type, True))


# testing
if __name__ == "__main__":
    reader_obj = readers()
    reader_obj.add_ctm_data('FREE', Path('download_bucket/gmi/'))
    reader_obj.read_ctm_data(
        '201010', 'H2O', frequency_opt='3-hourly', averaged=True)
    reader_obj.add_satellite_data(
        'SSMIS', Path('/media/asouri/Amir_5TB/NASA/SSMIS/'))
    reader_obj.read_satellite_data(
        '201010', read_ak=True, num_job=1)

    latitude = reader_obj.sat_data[0].latitude_center
    longitude = reader_obj.sat_data[0].longitude_center

    output = np.zeros((np.shape(latitude)[0], np.shape(
        latitude)[1], len(reader_obj.sat_data)))
    counter = -1
    for trop in reader_obj.sat_data:
        counter = counter + 1
        if trop is None:
            continue
        output[:, :, counter] = trop.vcd

    #output[output <= 0.0] = np.nan
    moutput = {}
    moutput["vcds"] = output
    moutput["lat"] = latitude
    moutput["lon"] = longitude
    savemat("ssmis.mat", moutput)
