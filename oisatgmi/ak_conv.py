import numpy as np
from scipy import interpolate
from oisatgmi.interpolator import _upscaler
from scipy.spatial import Delaunay
from scipy.io import savemat


def ak_conv(ctm_data: list, sat_data: list):
    print('Averaging Kernel Conv begins...')
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
            time_temp2 = time_temp[n].hour/24.0 + \
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
        # currently we only use monthly ECCOH so it's not worth going into hours or mins
        time_sat = time_sat_datetime.year*10000 + time_sat_datetime.month*100 +\
            time_sat_datetime.day
        # find the closest day
        if ctm_data[0].averaged == False:
            closest_index = np.argmin(np.abs(time_sat - time_ctm))
            # find the closest hour (this only works for monthly frequency)
            closest_index_day = int(np.floor(closest_index))
        else:
            closest_index = int(0)
            closest_index_day = int(0)

        print("The closest GMI file used for the L2 at " + str(L2_granule.time) +
              " is at " + str(time_ctm_datetype[closest_index_day]))
        # take the profile and pressure from the right ctm data
        Mair = 28.97e-3
        g = 9.80665
        N_A = 6.02214076e23
        if ctm_data[0].ctmtype == "ECCOH":
           ctm_mid_pressure = ctm_data[closest_index_day].pressure_mid[ :, :, :].squeeze(
           )
           ctm_profile = ctm_data[closest_index_day].gas_profile[ :, :, :].squeeze(
           )
           ctm_deltap = ctm_data[closest_index_day].delta_p[ :, :, :].squeeze(
           )
           ctm_partial_column = ctm_deltap*ctm_profile/g/Mair*N_A*1e-4*1e-15*100.0*1e-9
        elif ctm_data[0].ctmtype == "GMI":
           ctm_mid_pressure = np.nanmean(ctm_data[closest_index_day].pressure_mid[:, :, :, :], axis=0).squeeze(
           )
           ctm_profile = np.nanmean(ctm_data[closest_index_day].gas_profile[:, :, :, :], axis=0).squeeze(
           )
           ctm_deltap = np.nanmean(ctm_data[closest_index_day].delta_p[:, :, :, :], axis=0).squeeze(
           )
           ctm_partial_column = ctm_deltap*ctm_profile/g/Mair*N_A*1e-4*1e-15*100.0*1e-9
        # see if we need to upscale the ctm fields
        if L2_granule.ctm_upscaled_needed == True:
            ctm_mid_pressure_new = np.zeros((np.shape(ctm_mid_pressure)[0],
                                             np.shape(L2_granule.longitude_center)[0], np.shape(
                                                 L2_granule.longitude_center)[1],
                                             ))*np.nan
            ctm_profile_new = np.zeros_like(ctm_mid_pressure_new)*np.nan
            ctm_partial_column_new = np.zeros_like(ctm_mid_pressure_new)*np.nan
            sat_coordinate = {}
            sat_coordinate["Longitude"] = L2_granule.longitude_center
            sat_coordinate["Latitude"] = L2_granule.latitude_center
            # TODO define 0.6
            for z in range(0, np.shape(ctm_mid_pressure)[0]):
                _, _, ctm_mid_pressure_new[z, :, :], _ = _upscaler(ctm_data[0].longitude, ctm_data[0].latitude,
                                                                   ctm_mid_pressure[z, :, :], sat_coordinate, 0.6, 1.0, tri=tri)
                _, _, ctm_profile_new[z, :, :], _ = _upscaler(ctm_data[0].longitude, ctm_data[0].latitude,
                                                              ctm_profile[z, :, :], sat_coordinate, 0.6, 1.0, tri=tri)
                _, _, ctm_partial_column_new[z, :, :], _ = _upscaler(ctm_data[0].longitude, ctm_data[0].latitude,
                                                                     ctm_partial_column[z, :, :], sat_coordinate, 0.6, 1.0, tri=tri)
            ctm_mid_pressure = ctm_mid_pressure_new
            ctm_profile = ctm_profile_new
            ctm_partial_column = ctm_partial_column_new
            ctm_profile_new = []
            ctm_mid_pressure_new = []
            ctm_partial_column_new = []

        model_VCD_after = np.zeros_like(L2_granule.vcd)*np.nan
        for i in range(0, np.shape(L2_granule.vcd)[0]):
            for j in range(0, np.shape(L2_granule.vcd)[1]):
                if np.isnan(L2_granule.vcd[i, j]):
                    continue
                ctm_profile_tmp = ctm_profile[:, i, j].squeeze()
                ctm_mid_pressure_tmp = ctm_mid_pressure[:, i, j].squeeze()
                # interpolate the prior profiles
                f = interpolate.interp1d(
                    np.log(ctm_mid_pressure_tmp),
                    ctm_profile_tmp, fill_value=np.nan, bounds_error=False)
                interpolated_ctm_profile = f(
                    np.log(L2_granule.pressure_mid[:, i, j].squeeze()))
                # after applying AKs
                L2_granule.averaging_kernels[0, i, j] = np.nan
                model_VCD_after[i, j] = L2_granule.aprior_column[i, j] +\
                    np.nansum(L2_granule.averaging_kernels[:, i, j].squeeze(
                    )*(np.log10(interpolated_ctm_profile)-np.log10(L2_granule.apriori_profile[:, i, j].squeeze())))
                #model_VCD_after[i,j] = np.nansum(ctm_partial_tmp)

        # updating the ctm data
        model_VCD_after[np.isnan(L2_granule.vcd)] = np.nan
        model_VCD_after[np.isinf(L2_granule.vcd)] = np.nan
        sat_data[counter].ctm_vcd = model_VCD_after
        sat_data[counter].ctm_time_at_sat = time_ctm[closest_index]

        counter += 1

    return sat_data
