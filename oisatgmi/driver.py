from oisatgmi.reader import readers
from pathlib import Path
from oisatgmi.amf_recal import amf_recal
from oisatgmi.averaging import averaging
from oisatgmi.pwv_cal import pwv_calculator
from oisatgmi.optimal_interpolation import OI
from oisatgmi.report import report
from oisatgmi.ak_conv_mopitt import ak_conv_mopitt
from oisatgmi.ak_conv_gosat import ak_conv_gosat
import numpy as np
from scipy.io import savemat
from numpy import dtype
from netCDF4 import Dataset
import os


class oisatgmi(object):

    def __init__(self) -> None:
        pass

    def read_data(self, ctm_type: str, ctm_path: Path, ctm_gas_name: str, ctm_frequency: str,
                  sat_type: str, sat_path: Path, YYYYMM: str, averaged=False, read_ak=True, trop=False, num_job=1):
        reader_obj = readers()
        reader_obj.add_ctm_data(ctm_type, ctm_path)
        reader_obj.read_ctm_data(YYYYMM, ctm_gas_name,
                                 frequency_opt=ctm_frequency, averaged=averaged, num_job=num_job)
        reader_obj.add_satellite_data(
            sat_type, sat_path)
        reader_obj.read_satellite_data(
            YYYYMM, read_ak=read_ak, trop=trop, num_job=num_job)
        self.reader_obj = reader_obj
        self.gasname = ctm_gas_name[0]
        reader_obj = []

    def recal_amf(self):

        self.reader_obj.sat_data = amf_recal(
            self.reader_obj.ctm_data, self.reader_obj.sat_data)
        
    def cal_pwv(self):

        self.reader_obj.sat_data = pwv_calculator(self.reader_obj.ctm_data, self.reader_obj.sat_data)

    def conv_ak(self,sensor:str):
        if sensor == 'MOPITT':
           self.reader_obj.sat_data = ak_conv_mopitt(
               self.reader_obj.ctm_data, self.reader_obj.sat_data)
        if sensor == 'GOSAT':
           self.reader_obj.sat_data = ak_conv_gosat(
               self.reader_obj.ctm_data, self.reader_obj.sat_data)

    def average(self, startdate: str, enddate: str, gasname=None):
        '''
            average the data
            Input:
                startdate [str]: starting date in YYYY-mm-dd format string
                enddate [str]: ending date in YYYY-mm-dd format string  
        '''
        self.sat_averaged_vcd, self.sat_averaged_error, self.ctm_averaged_vcd, self.aux1, self.aux2 = averaging(
            startdate, enddate, self.reader_obj)
        if gasname == 'O3':
            self.ctm_averaged_vcd = self.ctm_averaged_vcd/(2.69e16*1e-15)

    def oi(self, sensor: str, error_ctm=50.0):
        if sensor != 'GOSAT':
            self.ctm_averaged_vcd_corrected, self.ak_OI, self.increment_OI, self.error_OI = OI(self.ctm_averaged_vcd, self.sat_averaged_vcd,
                        (self.ctm_averaged_vcd*error_ctm/100.0)**2, self.sat_averaged_error**2, regularization_on=True)
        else:
            self.ctm_averaged_vcd_corrected, self.ak_OI, self.increment_OI, self.error_OI = OI(self.aux2, self.aux1,
                        (self.aux2*error_ctm/100.0)**2, self.sat_averaged_error**2, regularization_on=True)
    def reporting(self, fname: str, gasname, folder='report'):

        # pick the right latitude and longitude
        # the right one is the coarsest one so
        if np.size(self.reader_obj.ctm_data[0].latitude)*np.size(self.reader_obj.ctm_data[0].longitude) > \
           np.size(self.reader_obj.sat_data[0].latitude_center)*np.size(self.reader_obj.sat_data[0].longitude_center):

            lat = self.reader_obj.sat_data[0].latitude_center
            lon = self.reader_obj.sat_data[0].longitude_center
        else:
            lat = self.reader_obj.ctm_data[0].latitude
            lon = self.reader_obj.ctm_data[0].longitude

        report(lon, lat, self.ctm_averaged_vcd, self.ctm_averaged_vcd_corrected,
               self.sat_averaged_vcd, self.sat_averaged_error, self.increment_OI, self.ak_OI, self.error_OI, self.aux1, self.aux2, fname, folder, gasname)

    def write_to_nc(self, output_file, output_folder='diag'):
        ''' 
        Write the final results to a netcdf
        ARGS:
            output_file (char): the name of file to be outputted
        '''
        # writing
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        ncfile = Dataset(output_folder + '/' + output_file + '.nc', 'w')

        # create the x and y dimensions.
        ncfile.createDimension('x', np.shape(self.sat_averaged_vcd)[0])
        ncfile.createDimension('y', np.shape(self.sat_averaged_vcd)[1])

        data1 = ncfile.createVariable(
            'sat_averaged_vcd', dtype('float32').char, ('x', 'y'))
        data1[:, :] = self.sat_averaged_vcd

        data2 = ncfile.createVariable(
            'ctm_averaged_vcd_prior', dtype('float32').char, ('x', 'y'))
        data2[:, :] = self.ctm_averaged_vcd

        data3 = ncfile.createVariable(
            'ctm_averaged_vcd_posterior', dtype('float32').char, ('x', 'y'))
        data3[:, :] = self.ctm_averaged_vcd_corrected

        data4 = ncfile.createVariable(
            'sat_averaged_error', dtype('float32').char, ('x', 'y'))
        data4[:, :] = self.sat_averaged_error

        data5 = ncfile.createVariable(
            'ak_OI', dtype('float32').char, ('x', 'y'))
        data5[:, :] = self.ak_OI

        data6 = ncfile.createVariable(
            'error_OI', dtype('float32').char, ('x', 'y'))
        data6[:, :] = self.error_OI

        scaling_factor = self.ctm_averaged_vcd_corrected/self.ctm_averaged_vcd
        scaling_factor[np.where((np.isnan(scaling_factor)) | (np.isinf(scaling_factor)) |
                       (scaling_factor == 0.0))] = 1.0
        data7 = ncfile.createVariable(
            'scaling_factor', dtype('float32').char, ('x', 'y'))
        data7[:, :] = scaling_factor

        data8 = ncfile.createVariable(
            'lon', dtype('float32').char, ('x', 'y'))
        data8[:, :] = self.reader_obj.sat_data[0].longitude_center

        data9 = ncfile.createVariable(
            'lat', dtype('float32').char, ('x', 'y'))
        data9[:, :] = self.reader_obj.sat_data[0].latitude_center

        data10 = ncfile.createVariable(
            'aux1', dtype('float32').char, ('x', 'y'))
        data10[:, :] = self.aux1

        data11 = ncfile.createVariable(
            'aux2', dtype('float32').char, ('x', 'y'))
        data11[:, :] = self.aux2

        ncfile.close()

    def savedaily(self, folder, gasname, date):
        # extract sat data
        if not os.path.exists(folder):
            os.makedirs(folder)
        latitude = self.reader_obj.sat_data[0].latitude_center
        longitude = self.reader_obj.sat_data[0].longitude_center
        vcd_sat = np.zeros((np.shape(latitude)[0], np.shape(
            latitude)[1], len(self.reader_obj.sat_data)))
        vcd_err = np.zeros_like(vcd_sat)
        vcd_ctm = np.zeros_like(vcd_sat)
        time_sat = np.zeros((len(self.reader_obj.sat_data)))
        counter = -1
        for sat in self.reader_obj.sat_data:
            counter = counter + 1
            if sat is None:
                continue
            vcd_sat[:, :, counter] = sat.vcd
            vcd_ctm[:, :, counter] = sat.ctm_vcd
            vcd_err[:, :, counter] = sat.uncertainty
            time_sat[counter] = 10000.0*sat.time.year + 100.0 * \
                sat.time.month + sat.time.day + sat.time.hour/24.0

        sat = {"vcd_sat": vcd_sat, "vcd_ctm": vcd_ctm,
               "vcd_err": vcd_err, "time_sat": time_sat, "lat": latitude, "lon": longitude}
        savemat(folder + "/" + "sat_data_" +
                gasname + "_" + date + ".mat", sat)


# testing
if __name__ == "__main__":

    oisatgmi_obj = oisatgmi()
    oisatgmi_obj.read_data('FREE', Path('/home/asouri/git_repos/mule/eccoh_sample'), 'NO2', 'monthly', 'OMI_NO2',
                           Path('download_bucket/omi_no2/'), '201803',
                           averaged=True, read_ak=False, trop=True, num_job=1)
    oisatgmi_obj.recal_amf()
    #oisatgmi_obj.conv_ak()
    oisatgmi_obj.average('2018-03-01', '2018-04-01')
    oisatgmi_obj.oi(error_ctm=10.0)
    oisatgmi_obj.reporting('NO2_200503_new', 'NO2', folder='report')
    oisatgmi_obj.write_to_nc('NO2_200503_new', 'diag')
