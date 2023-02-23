import numpy as np
from scipy import interpolate
from pathlib import Path
import datetime
import glob


def amf_recal(ctm_data: list, sat_data: list, gas_name: str):

    # list the time in ctm_data
    time_ctm = []
    for ctm_granule in ctm_data:
        time_temp = ctm_granule.time
        for n in range(len(time_temp)):
            time_temp2 = time_temp[n].year*10000 + time_temp[n].month*100 +\
                time_temp[n].day + time_temp[n].hour/24.0 + \
                time_temp[n].minute/60.0/24.0 + time_temp[n].second/3600.0/24.0
            time_ctm.append(time_temp2)
    time_ctm = np.array(time_ctm)
    # loop over the satellite list
    counter = 0
    for L2_granule in sat_data:
        time_sat = L2_granule.time
        time_sat = time_sat.year*10000 + time_sat.month*100 +\
            time_sat.day + time_sat.hour/24.0 + time_sat.minute / \
            60.0/24.0 + time_sat.second/3600.0/24.0
        # find the closest day
        closest_index = np.argmin(np.abs(time_sat - time_ctm))
        # find the closest hour (this only works for 3-hourly frequency)
        closest_index_day = int(np.floor(closest_index/8.0))
        closest_index_hour = int(closest_index % 8)
        # take the profile and pressure from the right ctm data
        ctm_mid_pressure = ctm_data[closest_index_day].pressure_mid[closest_index_hour, :, :, :].squeeze(
        )
        ctm_profile = ctm_data[closest_index_day].gas_profile[gas_name][closest_index_hour, :, :, :].squeeze(
        )
        ctm_deltap = ctm_data[closest_index_day].delta_p[closest_index_hour, :, :, :].squeeze(
        )

        # interpolate vertical grid
        Mair = 28.97e-3
        g = 9.80665
        N_A = 6.02214076e23
        new_amf = np.zeros_like(L2_granule.vcd)*np.nan
        for i in range(0, np.shape(L2_granule.vcd)[0]):
            for j in range(0, np.shape(L2_granule.vcd)[1]):
                if np.isnan(L2_granule.vcd[i, j]):
                    continue
                ctm_profile_tmp = ctm_profile[:, i, j].squeeze()
                ctm_deltap_tmp = ctm_deltap[:, i, j].squeeze()
                ctm_partial_column = (ctm_profile_tmp*ctm_deltap_tmp/g/Mair*N_A*1e-4*1e-15)*100.0
                ctm_mid_pressure_tmp = ctm_mid_pressure[:, i, j].squeeze()
                # interpolate
                f = interpolate.interp1d(
                    np.log(L2_granule.pressure_mid[:,i,j].squeeze()), 
                    L2_granule.scattering_weights[:,i,j].squeeze(), fill_value="extrapolate")
                interpolated_SW = f(np.log(ctm_mid_pressure_tmp))
                # remove above tropopause SWs
                if np.size(L2_granule.tropopause) != 1:
                    interpolated_SW[ctm_mid_pressure_tmp <
                                    L2_granule.tropopause[i, j]] = np.nan
                    ctm_partial_column[ctm_mid_pressure_tmp <
                                    L2_granule.tropopause[i, j]] = np.nan
                # calculate model SCD
                model_SCD = np.nansum(interpolated_SW*ctm_partial_column)
                # calculate model VCD
                model_VCD = np.nansum(ctm_partial_column)
                # calculate model AMF
                model_AMF = model_SCD/model_VCD
                # new amf
                new_amf[i, j] = model_AMF
        # updating the sat data
        sat_data[counter].vcd = sat_data[counter].scd/new_amf

    return sat_data
