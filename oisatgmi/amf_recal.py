import numpy as np
from scipy import interpolate
from scipy.spatial import Delaunay
from oisatgmi.interpolator import _upscaler


def _flatten_time(time_obj):
    """Convert datetime object to float representation."""
    return (
        time_obj.year * 10000 +
        time_obj.month * 100 +
        time_obj.day +
        time_obj.hour / 24.0 +
        time_obj.minute / 60.0 / 24.0 +
        time_obj.second / 3600.0 / 24.0
    )

def _hour_only_time(time_obj):
    """Convert datetime object to hour fraction of the day."""
    return (
        time_obj.hour / 24.0 +
        time_obj.minute / 60.0 / 24.0 +
        time_obj.second / 3600.0 / 24.0
    )

def _find_closest_indices(time_sat, time_sat_hour_only, ctm_data, time_ctm, time_ctm_hour_only):
    """Find closest time indices for CTM data."""
    averaged = ctm_data[0].averaged
    if not averaged:
        closest_index = np.argmin(np.abs(time_sat - time_ctm))
        closest_index_day = int(np.floor(closest_index / 8.0))
        closest_index_hour = int(closest_index % 8)
    else:
        closest_index = np.argmin(np.abs(time_sat_hour_only - time_ctm_hour_only))
        closest_index_hour = int(closest_index)
        closest_index_day = 0
    return closest_index, closest_index_day, closest_index_hour

def _get_ctm_fields(ctm_data, closest_index_day, closest_index_hour):
    """Extract CTM profiles and pressure fields."""
    if ctm_data[0].ctmtype == "FREE":
        mid_pressure = ctm_data[closest_index_day].pressure_mid.squeeze()
        profile = ctm_data[closest_index_day].gas_profile.squeeze()
        deltap = ctm_data[closest_index_day].delta_p.squeeze()
    else:
        mid_pressure = ctm_data[closest_index_day].pressure_mid[closest_index_hour].squeeze()
        profile = ctm_data[closest_index_day].gas_profile[closest_index_hour].squeeze()
        deltap = ctm_data[closest_index_day].delta_p[closest_index_hour].squeeze()
    return mid_pressure, profile, deltap

def _calculate_ctm_partial_column(deltap, profile):
    """Calculate CTM partial column."""
    Mair = 28.97e-3
    g = 9.80665
    N_A = 6.02214076e23
    return deltap * profile / g / Mair * N_A * 1e-4 * 1e-15 * 100.0 * 1e-9

def _upscale_ctm_fields(ctm_mid_pressure, ctm_profile, deltap, ctm_data, L2_granule, tri):
    """Upscale CTM fields to match satellite grid."""
    sat_coord = {
        "Longitude": L2_granule.longitude_center,
        "Latitude": L2_granule.latitude_center
    }
    size_grid_sat_lon = np.abs(sat_coord["Longitude"][0, 0] - sat_coord["Longitude"][0, 1])
    size_grid_sat_lat = np.abs(sat_coord["Latitude"][0, 0] - sat_coord["Latitude"][1, 0])
    threshold_sat = np.sqrt(size_grid_sat_lon ** 2 + size_grid_sat_lat ** 2)
    ctm_longitude = ctm_data[0].longitude
    ctm_latitude = ctm_data[0].latitude
    size_grid_model_lon = np.abs(ctm_longitude[0, 0] - ctm_longitude[0, 1])
    size_grid_model_lat = np.abs(ctm_latitude[0, 0] - ctm_latitude[1, 0])
    gridsize_ctm = np.sqrt(size_grid_model_lon ** 2 + size_grid_model_lat ** 2)
    layers = np.shape(ctm_mid_pressure)[0]
    new_shape = (layers, *np.shape(sat_coord["Longitude"]))
    ctm_mid_pressure_new = np.full(new_shape, np.nan)
    ctm_partial_new = np.full(new_shape, np.nan)
    for z in range(layers):
        _, _, ctm_mid_pressure_new[z], _ = _upscaler(
            ctm_longitude, ctm_latitude, ctm_mid_pressure[z], sat_coord, gridsize_ctm, threshold_sat, tri=tri
        )
        _, _, ctm_partial_new[z], _ = _upscaler(
            ctm_longitude, ctm_latitude, _calculate_ctm_partial_column(deltap[z], ctm_profile[z]), sat_coord, gridsize_ctm, threshold_sat, tri=tri
        )
    return ctm_mid_pressure_new, ctm_partial_new

def _apply_tropopause_mask(ctm_partial_column, ctm_mid_pressure, tropopause):
    """Mask partial column below tropopause."""
    layers = np.shape(ctm_partial_column)[0]
    for z in range(layers):
        mask = ctm_mid_pressure[z] < tropopause
        ctm_partial_column[z][mask] = np.nan
    return ctm_partial_column

def _vertical_interp_and_amf(L2_granule, ctm_mid_pressure, ctm_partial_column):
    """Perform vertical interpolation, AMF recalculation, VCD computation."""
    new_amf = np.full_like(L2_granule.vcd, np.nan)
    model_VCD = np.full_like(L2_granule.vcd, np.nan)
    for i in range(L2_granule.vcd.shape[0]):
        for j in range(L2_granule.vcd.shape[1]):
            if np.isnan(L2_granule.vcd[i, j]):
                continue
            ctm_partial_column_tmp = ctm_partial_column[:, i, j]
            ctm_mid_pressure_tmp = ctm_mid_pressure[:, i, j]
            # Interpolate scattering weights to CTM vertical grid (log pressure)
            f = interpolate.interp1d(
                np.log(L2_granule.pressure_mid[:, i, j].squeeze()),
                L2_granule.scattering_weights[:, i, j].squeeze(), fill_value="extrapolate"
            )
            interpolated_SW = f(np.log(ctm_mid_pressure_tmp))
            interpolated_SW[np.isinf(interpolated_SW)] = 0.0
            # Mask below tropopause if needed
            if np.size(L2_granule.tropopause) != 1:
                mask = ctm_mid_pressure_tmp < L2_granule.tropopause[i, j]
                interpolated_SW[mask] = np.nan
                ctm_partial_column_tmp[mask] = np.nan
            model_SCD = np.nansum(interpolated_SW * ctm_partial_column_tmp)
            model_VCD[i, j] = np.nansum(ctm_partial_column_tmp)
            model_AMF = model_SCD / model_VCD[i, j] if model_VCD[i, j] != 0 else np.nan
            new_amf[i, j] = model_AMF
    return new_amf, model_VCD

def amf_recal(ctm_data: list, sat_data: list):
    print('AMF Recal begins...')
    # Flatten CTM times
    time_ctm, time_ctm_hour_only, time_ctm_datetype = [], [], []
    for ctm_granule in ctm_data:
        times = ctm_granule.time
        time_ctm.extend([_flatten_time(t) for t in times])
        time_ctm_hour_only.extend([_hour_only_time(t) for t in times])
        time_ctm_datetype.append(ctm_granule.time)
    time_ctm = np.array(time_ctm)
    time_ctm_hour_only = np.array(time_ctm_hour_only)
    # Create triangulation for upscaling
    lon_flat = ctm_data[0].longitude.flatten()
    lat_flat = ctm_data[0].latitude.flatten()
    #tri = Delaunay(np.column_stack((lon_flat, lat_flat)))
    tri = None # upscaling is now done using cKDtree method
    # Main satellite loop
    for counter, L2_granule in enumerate(sat_data):
        if L2_granule is None:
            continue
        # Compute satellite times
        time_sat_datetime = L2_granule.time
        time_sat = _flatten_time(time_sat_datetime)
        time_sat_hour_only = _hour_only_time(time_sat_datetime)
        # Find closest CTM time
        closest_index, closest_index_day, closest_index_hour = _find_closest_indices(
            time_sat, time_sat_hour_only, ctm_data, time_ctm, time_ctm_hour_only
        )
        print(f"The closest GMI file used for the L2 at {L2_granule.time} is at {time_ctm_datetype[closest_index_day][closest_index_hour]}")
        # Extract CTM fields
        ctm_mid_pressure, ctm_profile, ctm_deltap = _get_ctm_fields(ctm_data, closest_index_day, closest_index_hour)
        ctm_partial_column = _calculate_ctm_partial_column(ctm_deltap, ctm_profile)
        # Upscale if needed
        if L2_granule.ctm_upscaled_needed == True:
            print("Upscaling of the model is needed.")
            ctm_mid_pressure, ctm_partial_column = _upscale_ctm_fields(
                ctm_mid_pressure, ctm_profile, ctm_deltap, ctm_data, L2_granule, tri
            )
        # Vertical grid interpolation and AMF recalculation
        if np.size(L2_granule.scattering_weights) == 1:
            print('No scattering weights found, recalculation is not possible..just grabbing VCDs')
            if np.size(L2_granule.tropopause) != 1:
                ctm_partial_column = _apply_tropopause_mask(ctm_partial_column, ctm_mid_pressure, L2_granule.tropopause)
            # Calculate model VCD
            model_VCD = np.nansum(ctm_partial_column, axis=0)
            model_VCD[np.isnan(L2_granule.vcd)] = np.nan
            L2_granule.ctm_vcd = model_VCD
            L2_granule.ctm_time_at_sat = time_ctm[closest_index]
            L2_granule.old_amf = np.empty((1))
            L2_granule.new_amf = np.empty((1))
            continue
        # AMF recalculation
        new_amf, model_VCD = _vertical_interp_and_amf(L2_granule, ctm_mid_pressure, ctm_partial_column)
        # Update satellite data object
        L2_granule.old_amf = getattr(L2_granule, 'amf', None)
        new_amf[np.isnan(L2_granule.vcd)] = np.nan
        L2_granule.new_amf = new_amf
        # VCD correction
        L2_granule.vcd = (L2_granule.amf * L2_granule.vcd) / new_amf
        model_VCD[np.isnan(L2_granule.vcd)] = np.nan
        model_VCD[np.isinf(L2_granule.vcd)] = np.nan
        L2_granule.ctm_vcd = model_VCD
        L2_granule.ctm_time_at_sat = time_ctm[closest_index]

    return sat_data
