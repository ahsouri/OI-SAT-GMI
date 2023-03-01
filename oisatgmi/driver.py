from reader import readers
from pathlib import Path
from amf_recal import amf_recal
from averaging import averaging
from optimal_interpolation import OI
from report import report
import numpy as np
from scipy.io import savemat


class oisatgmi(object):

    def __init__(self) -> None:
        pass

    def read_data(self, ctm_type: str, ctm_path: Path, ctm_gas_name: list, ctm_frequency: str, sat_type: str, sat_path: Path, YYYYMM: str, read_ak=True, num_job=1):

        reader_obj = readers()
        reader_obj.add_ctm_data(ctm_type, ctm_path)
        reader_obj.read_ctm_data(YYYYMM, ctm_gas_name,
                                 frequency_opt=ctm_frequency)
        reader_obj.add_satellite_data(
            sat_type, sat_path)
        reader_obj.read_satellite_data(
            YYYYMM, read_ak=read_ak, num_job=num_job)
        self.reader_obj = reader_obj
        self.gasname = ctm_gas_name[0]
        reader_obj = []

    def recal_amf(self):

        self.reader_obj.sat_data = amf_recal(
            self.reader_obj.ctm_data, self.reader_obj.sat_data, self.gasname)

    def average(self, startdate: str, enddate: str):
        '''
            average the data
            Input:
                startdate [str]: starting date in YYYY-mm-dd format string
                enddate [str]: ending date in YYYY-mm-dd format string  
        '''
        self.sat_averaged_vcd, self.sat_averaged_error, self.ctm_averaged_vcd, self.new_amf, self.old_amf = averaging(
            startdate, enddate, self.reader_obj)

    def oi(self, error_ctm=200.0):

        self.ctm_averaged_vcd_corrected, self.ak_OI, self.increment_OI, self.error_OI = OI(self.ctm_averaged_vcd, self.sat_averaged_vcd,
                                                                                           (self.ctm_averaged_vcd*error_ctm/100.0)**2, self.sat_averaged_error**2, regularization_on=True)

    def reporting(self, fname: str):

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
               self.sat_averaged_vcd, self.increment_OI, self.ak_OI, self.error_OI,self.new_amf, self.old_amf, fname)


# testing
if __name__ == "__main__":

    oisatgmi_obj = oisatgmi()
    oisatgmi_obj.read_data('GMI', Path('download_bucket/gmi'), ['NO2'], '3-hourly', 'OMI_NO2',
                       Path('download_bucket/omi_no2'),'201905',read_ak=True, num_job=1)
    oisatgmi_obj.recal_amf()
    oisatgmi_obj.average('2019-05-01','2019-06-01')
    oisatgmi_obj.oi()
    oisatgmi_obj.reporting('NO2_201905')
