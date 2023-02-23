from reader import readers
from pathlib import Path
from amf_recal import amf_recal
import datetime
import numpy as np
from scipy.io import savemat

def _daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + datetime.timedelta(n)


class oisatgmi(object):

    def __init__(self) -> None:
        pass

    def read_data(self, ctm_type: str, ctm_path: Path, ctm_gas_name: list, ctm_frequency: str, sat_type: str, sat_path: Path, read_ak=True, num_job=1):

        reader_obj = readers()
        reader_obj.add_ctm_data(ctm_type, ctm_path)
        reader_obj.read_ctm_data(ctm_gas_name, frequency_opt=ctm_frequency)
        reader_obj.add_satellite_data(
            sat_type, sat_path)
        reader_obj.read_satellite_data(read_ak=read_ak, num_job=num_job)
        self.reader_obj = reader_obj
        self.gasname = ctm_gas_name[0]
        reader_obj = []

    def recal_amf(self):

        self.reader_obj.tropomi_data = amf_recal(
            self.reader_obj.ctm_data, self.reader_obj.tropomi_data, self.gasname)

    def average(self, startdate: str, enddate: str, frequency: str):
        '''
            average the data
            Input:
                startdate [str]: starting date in YYYY-mm-dd format string
                enddate [str]: ending date in YYYY-mm-dd format string  
                frequency [str]: 'monthly'
                                 'seasons'
                                 'yearly'
        '''
        # convert dates to datetime
        start_date = datetime.date(int(startdate[0:4]), int(
            startdate[5:7]), int(startdate[8:10]))
        end_date = datetime.date(int(enddate[0:4]), int(
            enddate[5:7]), int(enddate[8:10]))
        list_days = []
        list_months = []
        list_years = []
        for single_date in _daterange(start_date, end_date):
            list_days.append(single_date.day)
            list_months.append(single_date.month)
            list_years.append(single_date.year)

        list_days = np.array(list_days)
        list_months = np.array(list_months)
        list_years = np.array(list_years)

        sat_averaged_vcd = np.zeros((np.shape(self.reader_obj.tropomi_data[0].latitude_center)[0],
                                     np.shape(self.reader_obj.tropomi_data[0].latitude_center)[
            1],
            len(range(np.min(list_months),
                      np.max(list_months)+1)),
            len(range(np.min(list_years), np.max(list_years)+1))))
        sat_samples = np.zeros_like(sat_averaged_vcd)
        sat_averaged_error = np.zeros_like(sat_averaged_vcd)
        ctm_averaged_vcd = np.zeros_like(sat_averaged_vcd)
        for year in range(np.min(list_years), np.max(list_years)+1):
            for month in range(np.min(list_months), np.max(list_months)+1):
                #chosen_days = list_days[((list_months == month) & (list_years == year))]
                sat_chosen_vcd = []
                sat_chosen_error = []
                ctm_chosen_vcd = []
                for sat_data in self.reader_obj.tropomi_data:
                    time_sat = sat_data.time
                    # see if it falls
                    if ((time_sat.year == year) and (time_sat.month == month)):
                        sat_chosen_vcd.append(sat_data.vcd)
                        sat_chosen_error.append(sat_data.uncertainty)
                for ctm_data in self.reader_obj.ctm_data:
                    time_ctm = ctm_data.time_at_sat
                    year_ctm = np.floor(time_ctm/1e4)
                    month_ctm = np.floor((time_ctm-year_ctm*1e4)/100)
                    day_ctm = np.floor(time_ctm - year_ctm*1e4 - month_ctm*1e2)
                    frac_day_ctm = (time_ctm - year_ctm*1e4 -
                                    month_ctm*1e2) - day_ctm
                    time_ctm = datetime.datetime(int(year_ctm), int(
                        month_ctm), int(day_ctm), int(np.floor(frac_day_ctm*24)))
                    # see if it falls
                    if ((time_ctm.year == year) and (time_ctm.month == month)):
                        ctm_chosen_vcd.append(ctm_data.vcd)
                sat_chosen_vcd = np.array(sat_chosen_vcd)
                sat_chosen_vcd[np.isinf(sat_chosen_vcd)] = np.nan
                sat_chosen_error = np.array(sat_chosen_error)
                ctm_chosen_vcd = np.array(ctm_chosen_vcd)
            if np.size(sat_chosen_vcd) != 0:
                sat_averaged_vcd[:, :, month - min(list_months), year - min(
                    list_years)] = np.squeeze(np.nansum(sat_chosen_vcd, axis=0))
                sat_samples[:, :, month - min(list_months), year - min(
                    list_years)] = np.count_nonzero(~np.isnan(sat_chosen_vcd))
                
                moutput = {}
                moutput["sample"] = sat_samples
                moutput["vcd"] = sat_averaged_vcd
                savemat("vcds.mat", moutput)

# testing
if __name__ == "__main__":

    oi_obj = oisatgmi()
    oi_obj.average('2021-10-02', '2021-11-04', 'monthly')
