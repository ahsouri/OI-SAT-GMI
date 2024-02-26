import numpy as np
from oisatgmi.interpolator import _upscaler
from scipy.spatial import Delaunay
from scipy.io import savemat


def pwv_calculator(ctm_data: list, sat_data: list):
    print('PWV begins...')
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

        print("The closest CTM file used for the L2 at " + str(L2_granule.time) +
              " is at " + str(time_ctm_datetype[closest_index_day]))
        # take the profile and pressure from the right ctm data
        Mair = 28.97e-3
        g = 9.80665
        N_A = 6.02214076e23
        if (ctm_data[0].ctmtype == "ECCOH") or (ctm_data[0].ctmtype == "FREE"):
           ctm_deltap = ctm_data[closest_index_day].delta_p[ :, :, :].squeeze(
           )
           ctm_profile = ctm_data[closest_index_day].gas_profile[ :, :, :].squeeze(
           )
           ctm_partial_column = ctm_deltap*ctm_profile/g/1000.0
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
            ctm_partial_column_new = np.zeros((np.shape(ctm_deltap)[0],
                                             np.shape(L2_granule.longitude_center)[0], np.shape(
                                                 L2_granule.longitude_center)[1],
                                             ))*np.nan
            sat_coordinate = {}
            sat_coordinate["Longitude"] = L2_granule.longitude_center
            sat_coordinate["Latitude"] = L2_granule.latitude_center
            size_grid_sat_lon = np.abs(sat_coordinate["Longitude"][0, 0]-sat_coordinate["Longitude"][0, 1])
            size_grid_sat_lat = np.abs(sat_coordinate["Latitude"][0, 0] - sat_coordinate["Latitude"][1, 0])
            threshold_sat = np.sqrt(size_grid_sat_lon**2 + size_grid_sat_lat**2)
            ctm_longitude = ctm_data[0].longitude
            ctm_latitude = ctm_data[0].latitude
            size_grid_model_lon = np.abs(ctm_longitude[0, 0]-ctm_longitude[0, 1])
            size_grid_model_lat = np.abs(ctm_latitude[0, 0] - ctm_latitude[1, 0])
            gridsize_ctm = np.sqrt(size_grid_model_lon**2 + size_grid_model_lat**2)
            for z in range(0, np.shape(ctm_mid_pressure)[0]):
                _, _, ctm_partial_column_new[z, :, :], _ = _upscaler(ctm_data[0].longitude, ctm_data[0].latitude,
                                                                     ctm_partial_column[z, :, :], sat_coordinate, gridsize_ctm, threshold_sat, tri=tri)
            ctm_partial_column = ctm_partial_column_new
            ctm_partial_column_new = []

        model_PWV = np.nansum(ctm_partial_column/1000.0,axis=0).squeeze()
        model_PWV[np.isnan(L2_granule.vcd)] = np.nan
        model_PWV[np.isinf(L2_granule.vcd)] = np.nan
        sat_data[counter].ctm_vcd = model_PWV

        counter += 1

    return sat_data
