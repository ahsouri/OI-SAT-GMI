import numpy as np
from scipy import interpolate
from oisatgmi.interpolator import _upscaler
from scipy.spatial import Delaunay

def amf_recal(ctm_data: list, sat_data: list):
    print('AMF Recal begins...')
    # list the time in ctm_data
    time_ctm = []
    time_ctm_hour_only = []
    time_ctm_datetype = []
    for ctm_granule in ctm_data:
        time_temp = ctm_granule.time
        for n in range(len(time_temp)):
            time_temp2 = time_temp[n].year*10000 + time_temp[n].month*100 +\
                time_temp[n].day + time_temp[n].hour/24.0 + \
                time_temp[n].minute/60.0/24.0 + time_temp[n].second/3600.0/24.0
            time_ctm.append(time_temp2)
            # I do this to save only hour in case of monthly-avereaged CTMs:
            time_temp2 =  time_temp[n].hour/24.0 + \
                time_temp[n].minute/60.0/24.0 + time_temp[n].second/3600.0/24.0
            time_ctm_hour_only.append(time_temp2)
        time_ctm_datetype.append(ctm_granule.time)

    time_ctm = np.array(time_ctm)
    time_ctm_hour_only = np.array(time_ctm_hour_only)
    # define the triangulation if we need to interpolate ctm_grid to sat_grid
    # because ctm_grid < sat_grid
    points = np.zeros((np.size(ctm_data[0].latitude), 2))
    points[:, 0] = ctm_data[0].longitude.flatten()
    points[:, 1] = ctm_data[0].latitude.flatten()
    tri = Delaunay(points)
    # loop over the satellite list
    counter = 0
    for L2_granule in sat_data:
        if (L2_granule is None):
            counter += 1
            continue
        time_sat_datetime = L2_granule.time
        time_sat = time_sat_datetime.year*10000 + time_sat_datetime.month*100 +\
            time_sat_datetime.day + time_sat_datetime.hour/24.0 + time_sat_datetime.minute / \
            60.0/24.0 + time_sat_datetime.second/3600.0/24.0
        time_sat_hourl_only = time_sat_datetime.hour/24.0 + time_sat_datetime.minute / \
            60.0/24.0 + time_sat_datetime.second/3600.0/24.0
        # find the closest day
        if ctm_data[0].averaged == False:
           closest_index = np.argmin(np.abs(time_sat - time_ctm))
           # find the closest hour (this only works for 3-hourly frequency)
           closest_index_day = int(np.floor(closest_index/8.0))
           closest_index_hour = int(closest_index % 8)
        else:
           closest_index = np.argmin(np.abs(time_sat_hourl_only - time_ctm_hour_only))
           # find the closest hour 
           closest_index_hour = int(closest_index)
           closest_index_day = int(0)

        print("The closest GMI file used for the L2 at " + str(L2_granule.time) +
              " is at " + str(time_ctm_datetype[closest_index_day][closest_index_hour]))        
        # take the profile and pressure from the right ctm data
        Mair = 28.97e-3
        g = 9.80665
        N_A = 6.02214076e23
        ctm_mid_pressure = ctm_data[closest_index_day].pressure_mid[closest_index_hour, :, :, :].squeeze(
        )
        ctm_profile = ctm_data[closest_index_day].gas_profile[closest_index_hour, :, :, :].squeeze(
        )
        ctm_deltap = ctm_data[closest_index_day].delta_p[closest_index_hour, :, :, :].squeeze(
        )
        ctm_partial_column = ctm_deltap*ctm_profile/g/Mair*N_A*1e-4*1e-15*100.0*1e-9
        # see if we need to upscale the ctm fields
        if L2_granule.ctm_upscaled_needed == True:
            ctm_mid_pressure_new = np.zeros((np.shape(ctm_mid_pressure)[0],
                                             np.shape(L2_granule.longitude_center)[0], np.shape(
                                                 L2_granule.longitude_center)[1],
                                             ))*np.nan
            ctm_partial_new = np.zeros_like(ctm_mid_pressure_new)*np.nan
            sat_coordinate = {}
            sat_coordinate["Longitude"] = L2_granule.longitude_center
            sat_coordinate["Latitude"] = L2_granule.latitude_center
            for z in range(0, np.shape(ctm_mid_pressure)[0]):
                _, _, ctm_mid_pressure_new[z, :, :], _ = _upscaler(ctm_data[0].longitude, ctm_data[0].latitude,
                                                                   ctm_mid_pressure[z, :, :], sat_coordinate, 0.6, 0.8, tri=tri)
                _, _, ctm_partial_new[z, :, :], _ = _upscaler(ctm_data[0].longitude, ctm_data[0].latitude,
                                                              ctm_deltap[z, :, :]*ctm_profile[z, :, :]/g/Mair*N_A*1e-4*1e-15*100.0*1e-9, sat_coordinate, 0.6, 0.8, tri=tri)
            ctm_mid_pressure = ctm_mid_pressure_new
            ctm_partial_column = ctm_partial_new
        # interpolate vertical grid
        # check if AMF recal is even possible
        if (np.size(L2_granule.scattering_weights) == 1):
            print(
                'no scattering weights were found, recalculation is not possible..just grabbing vcds')
            if np.size(L2_granule.tropopause) != 1:
                for z in range(0, np.shape(ctm_profile)[0]):
                    ctm_partial_column_tmp = ctm_partial_column[z, :, :].squeeze(
                    )
                    ctm_partial_column_tmp[ctm_mid_pressure[z, :, :] <
                                           L2_granule.tropopause] = np.nan
                    ctm_partial_column[z, :, :] = ctm_partial_column_tmp
            # calculate model VCD
            model_VCD = np.nansum(ctm_partial_column, axis=0)
            model_VCD[np.isnan(L2_granule.vcd)] = np.nan
            sat_data[counter].ctm_vcd = model_VCD
            sat_data[counter].ctm_time_at_sat = time_ctm[closest_index]
            sat_data[counter].old_amf = np.empty((1))
            sat_data[counter].new_amf = np.empty((1))
            counter += 1
            if counter == len(sat_data):  # skip the rest
                return sat_data
            continue
        new_amf = np.zeros_like(L2_granule.vcd)*np.nan
        model_VCD = np.zeros_like(L2_granule.vcd)*np.nan
        for i in range(0, np.shape(L2_granule.vcd)[0]):
            for j in range(0, np.shape(L2_granule.vcd)[1]):
                if np.isnan(L2_granule.vcd[i, j]):
                    continue
                ctm_partial_column_tmp = ctm_partial_column[:, i, j].squeeze()
                ctm_mid_pressure_tmp = ctm_mid_pressure[:, i, j].squeeze()
                # interpolate
                f = interpolate.interp1d(
                    np.log(L2_granule.pressure_mid[:, i, j].squeeze()),
                    L2_granule.scattering_weights[:, i, j].squeeze(), fill_value="extrapolate")
                interpolated_SW = f(np.log(ctm_mid_pressure_tmp))
                # remove bad values
                interpolated_SW[np.isinf(interpolated_SW)] = 0.0
                # remove above tropopause SWs
                if np.size(L2_granule.tropopause) != 1:
                    interpolated_SW[ctm_mid_pressure_tmp <
                                    L2_granule.tropopause[i, j]] = np.nan
                    ctm_partial_column_tmp[ctm_mid_pressure_tmp <
                                           L2_granule.tropopause[i, j]] = np.nan
                # calculate model SCD
                model_SCD = np.nansum(interpolated_SW*ctm_partial_column_tmp)
                # calculate model VCD
                model_VCD[i, j] = np.nansum(ctm_partial_column_tmp)
                # calculate model AMF
                model_AMF = model_SCD/model_VCD[i, j]
                # new amf
                new_amf[i, j] = model_AMF
        # updating the sat data
        sat_data[counter].old_amf = sat_data[counter].scd/sat_data[counter].vcd
        new_amf[np.isnan(sat_data[counter].vcd)] = np.nan
        sat_data[counter].new_amf = new_amf
        sat_data[counter].vcd = sat_data[counter].scd/new_amf
        model_VCD[np.isnan(L2_granule.vcd)] = np.nan
        model_VCD[np.isinf(L2_granule.vcd)] = np.nan
        sat_data[counter].ctm_vcd = model_VCD
        sat_data[counter].ctm_time_at_sat = time_ctm[closest_index]

        counter += 1

    return sat_data
