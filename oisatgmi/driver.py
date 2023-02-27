from reader import readers
from pathlib import Path
from amf_recal import amf_recal
from averaging import averaging
import datetime
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

        self.reader_obj.sat_data, self.reader_obj.ctm_data = amf_recal(
            self.reader_obj.ctm_data, self.reader_obj.sat_data, self.gasname)

    def average(self, startdate: str, enddate: str):
        '''
            average the data
            Input:
                startdate [str]: starting date in YYYY-mm-dd format string
                enddate [str]: ending date in YYYY-mm-dd format string  
        '''
        averaging(startdate, enddate, self.reader_obj)


# testing
if __name__ == "__main__":

    oi_obj = oisatgmi()
    oi_obj.average('2021-10-02', '2021-11-04')
