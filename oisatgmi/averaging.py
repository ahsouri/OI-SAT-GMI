import numpy as np
import datetime
from scipy.io import savemat
from oisatgmi.config import satellite_amf,satellite_opt


def _daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + datetime.timedelta(n)

def error_averager(error_X: np.array):
    error_Y = np.zeros((np.shape(error_X)[1],np.shape(error_X)[2]))*np.nan
    for i in range(0,np.shape(error_X)[1]):
        for j in range(0,np.shape(error_X)[2]):
            temp = []
            for k in range(0,np.shape(error_X)[0]):
                temp.append(error_X[k,i,j])
            temp = np.array(temp)
            temp[np.isinf(temp)]=np.nan
            temp2 = temp[~np.isnan(temp)]
            error_Y[i,j] = np.sum(temp2)/(np.size(temp2)**2)

    error_Y = np.sqrt(error_Y)
    return error_Y

def averaging(startdate: str, enddate: str, reader_obj):
    '''
          average the data
          Input:
              startdate [str]: starting date in YYYY-mm-dd format string
              enddate [str]: ending date in YYYY-mm-dd format string
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

    first_valid_idx = next(i for i, sat_data in enumerate(reader_obj.sat_data)
                          if sat_data is not None)

    sat_averaged_vcd = np.zeros((np.shape(reader_obj.sat_data[first_valid_idx].latitude_center)[0],
                                 np.shape(reader_obj.sat_data[first_valid_idx].latitude_center)[
        1],
        len(range(np.min(list_months),
                  np.max(list_months)+1)),
        len(range(np.min(list_years), np.max(list_years)+1))))
    #sat_samples = np.zeros_like(sat_averaged_vcd)*np.nan
    sat_averaged_error = np.zeros_like(sat_averaged_vcd)*np.nan
    ctm_averaged_vcd = np.zeros_like(sat_averaged_vcd)*np.nan
    sat_aux1 = np.zeros_like(sat_averaged_vcd)*np.nan
    sat_aux2 = np.zeros_like(sat_averaged_vcd)*np.nan
    for year in range(np.min(list_years), np.max(list_years)+1):
        for month in range(np.min(list_months), np.max(list_months)+1):
            sat_chosen_vcd = []
            sat_chosen_aux1 = []
            sat_chosen_aux2 = []
            sat_chosen_error = []
            ctm_chosen_vcd = []
            for sat_data in reader_obj.sat_data:
                if (sat_data is None):
                    continue
                time_sat = sat_data.time
                # see if it falls
                if ((time_sat.year == year) and (time_sat.month == month)):
                    sat_chosen_vcd.append(sat_data.vcd)
                    sat_chosen_error.append(sat_data.uncertainty)
                    ctm_chosen_vcd.append(sat_data.ctm_vcd)
                    if isinstance(sat_data, satellite_amf):
                        sat_chosen_aux1.append(sat_data.new_amf)
                        sat_chosen_aux2.append(sat_data.old_amf)
                    elif isinstance(sat_data, satellite_opt):
                        sat_chosen_aux1.append(sat_data.x_col)
                        sat_chosen_aux2.append(sat_data.ctm_xcol)
                    else: # null
                        sat_chosen_aux1.append(np.nan*sat_data.vcd)
                        sat_chosen_aux2.append(np.nan*sat_data.vcd)
            sat_chosen_vcd = np.array(sat_chosen_vcd)
            sat_chosen_vcd[np.isinf(sat_chosen_vcd)] = np.nan
            sat_chosen_error = np.array(sat_chosen_error)
            ctm_chosen_vcd = np.array(ctm_chosen_vcd)
            sat_chosen_aux1 = np.array(sat_chosen_aux1)
            sat_chosen_aux2 = np.array(sat_chosen_aux2)
        if np.size(sat_chosen_vcd) != 0:
            sat_averaged_vcd[:, :, month - min(list_months), year - min(
                list_years)] = np.squeeze(np.nanmean(sat_chosen_vcd, axis=0))
            sat_averaged_error[:, :, month - min(list_months), year - min(
                list_years)] = error_averager(sat_chosen_error**2)
            ctm_averaged_vcd[:, :, month - min(list_months), year - min(
                list_years)] = np.squeeze(np.nanmean(ctm_chosen_vcd, axis=0))
        if np.size(sat_chosen_aux1) != 0:
            sat_aux1[:, :, month - min(list_months), year - min(
                    list_years)] = np.squeeze(np.nanmean(sat_chosen_aux1, axis=0))
            sat_aux2[:, :, month - min(list_months), year - min(
                    list_years)] = np.squeeze(np.nanmean(sat_chosen_aux2, axis=0))
    # squeeze it
    sat_averaged_vcd = sat_averaged_vcd.squeeze()
    sat_averaged_error = sat_averaged_error.squeeze()
    ctm_averaged_vcd = ctm_averaged_vcd.squeeze()
    sat_aux1 = sat_aux1.squeeze()
    sat_aux2 = sat_aux2.squeeze()
    # average over all data
    if sat_averaged_vcd.ndim == 4:
        sat_averaged_vcd = np.nanmean(np.nanmean(
            sat_averaged_vcd, axis=3).squeeze(), axis=2).squeeze()
        ctm_averaged_vcd = np.nanmean(np.nanmean(
            ctm_averaged_vcd, axis=3).squeeze(), axis=2).squeeze()
        # TODO: we should update this but we never average over several months or years
        sat_averaged_error = np.sqrt(np.nanmean(np.nanmean(
            sat_averaged_error**2, axis=3).squeeze(), axis=2).squeeze())
        sat_aux1 = np.nanmean(np.nanmean(
                sat_aux1, axis=3).squeeze(), axis=2).squeeze()
        sat_aux2 = np.nanmean(np.nanmean(
                sat_aux2, axis=3).squeeze(), axis=2).squeeze()
    if sat_averaged_vcd.ndim == 3:
        sat_averaged_vcd = np.nanmean(sat_averaged_vcd, axis=2).squeeze()
        ctm_averaged_vcd = np.nanmean(ctm_averaged_vcd, axis=2).squeeze()
        # TODO: we should update this but we never average over several months or years
        sat_averaged_error = np.sqrt(np.nanmean(
            sat_averaged_error**2, axis=2).squeeze())
        sat_aux1 = np.nanmean(sat_aux1, axis=2).squeeze()
        sat_aux2 = np.nanmean(sat_aux2, axis=2).squeeze()

    return sat_averaged_vcd, sat_averaged_error, ctm_averaged_vcd, sat_aux1, sat_aux2
