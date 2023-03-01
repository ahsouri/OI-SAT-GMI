import numpy as np
import datetime
from scipy.io import savemat


def _daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + datetime.timedelta(n)


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

    sat_averaged_vcd = np.zeros((np.shape(reader_obj.sat_data[0].latitude_center)[0],
                                 np.shape(reader_obj.sat_data[0].latitude_center)[
        1],
        len(range(np.min(list_months),
                  np.max(list_months)+1)),
        len(range(np.min(list_years), np.max(list_years)+1))))
    #sat_samples = np.zeros_like(sat_averaged_vcd)*np.nan
    sat_averaged_error = np.zeros_like(sat_averaged_vcd)*np.nan
    ctm_averaged_vcd = np.zeros_like(sat_averaged_vcd)*np.nan
    sat_old_amf = np.zeros_like(sat_averaged_vcd)*np.nan
    sat_new_amf = np.zeros_like(sat_averaged_vcd)*np.nan
    for year in range(np.min(list_years), np.max(list_years)+1):
        for month in range(np.min(list_months), np.max(list_months)+1):
            #chosen_days = list_days[((list_months == month) & (list_years == year))]
            sat_chosen_vcd = []
            sat_chosen_new_amf = []
            sat_chosen_old_amf = []
            sat_chosen_error = []
            ctm_chosen_vcd = []
            for sat_data in reader_obj.sat_data:
                time_sat = sat_data.time
                # see if it falls
                if ((time_sat.year == year) and (time_sat.month == month)):
                    sat_chosen_vcd.append(sat_data.vcd)
                    sat_chosen_error.append(sat_data.uncertainty)
                    ctm_chosen_vcd.append(sat_data.ctm_vcd)
                    sat_chosen_new_amf.append(sat_data.new_amf)
                    sat_chosen_old_amf.append(sat_data.old_amf)
            sat_chosen_vcd = np.array(sat_chosen_vcd)
            sat_chosen_vcd[np.isinf(sat_chosen_vcd)] = np.nan
            sat_chosen_error = np.array(sat_chosen_error)
            ctm_chosen_vcd = np.array(ctm_chosen_vcd)
            sat_chosen_new_amf = np.array(sat_chosen_new_amf)
            sat_chosen_old_amf = np.array(sat_chosen_old_amf)
        if np.size(sat_chosen_vcd) != 0:
            sat_averaged_vcd[:, :, month - min(list_months), year - min(
                list_years)] = np.squeeze(np.nanmean(sat_chosen_vcd, axis=0))
            sat_averaged_error[:, :, month - min(list_months), year - min(
                list_years)] = np.sqrt(np.squeeze(np.nanmean(sat_chosen_error**2, axis=0)))
            ctm_averaged_vcd[:, :, month - min(list_months), year - min(
                list_years)] = np.squeeze(np.nanmean(ctm_chosen_vcd, axis=0))
        if np.size(sat_new_amf) != 0:
            sat_old_amf[:, :, month - min(list_months), year - min(
                list_years)] = np.squeeze(np.nanmean(sat_chosen_old_amf, axis=0))
            sat_new_amf[:, :, month - min(list_months), year - min(
                list_years)] = np.squeeze(np.nanmean(sat_chosen_new_amf, axis=0))

    # squeeze it
    sat_averaged_vcd = sat_averaged_vcd.squeeze()
    sat_averaged_error = sat_averaged_error.squeeze()
    ctm_averaged_vcd = ctm_averaged_vcd.squeeze()
    sat_old_amf = sat_old_amf.squeeze()
    sat_new_amf = sat_new_amf.squeeze()
    # average over all data
    if sat_averaged_vcd.ndim == 4:
        sat_averaged_vcd = np.nanmean(np.nanmean(
            sat_averaged_vcd, axis=3).squeeze(), axis=2).squeeze()
        ctm_averaged_vcd = np.nanmean(np.nanmean(
            ctm_averaged_vcd, axis=3).squeeze(), axis=2).squeeze()
        sat_old_amf = np.nanmean(np.nanmean(
            sat_old_amf, axis=3).squeeze(), axis=2).squeeze()
        sat_new_amf = np.nanmean(np.nanmean(
            sat_new_amf, axis=3).squeeze(), axis=2).squeeze()
        sat_averaged_error = np.sqrt(np.nanmean(np.nanmean(
            sat_averaged_error**2, axis=3).squeeze(), axis=2).squeeze())
    if sat_averaged_vcd.ndim == 3:
        sat_averaged_vcd = np.nanmean(sat_averaged_vcd, axis=2).squeeze()
        ctm_averaged_vcd = np.nanmean(ctm_averaged_vcd, axis=2).squeeze()
        sat_old_amf = np.nanmean(sat_old_amf, axis=2).squeeze()
        sat_new_amf = np.nanmean(sat_new_amf, axis=2).squeeze()
        sat_averaged_error = np.sqrt(np.nanmean(
            sat_averaged_error**2, axis=2).squeeze())
    return sat_averaged_vcd, sat_averaged_error, ctm_averaged_vcd, sat_new_amf, sat_old_amf
