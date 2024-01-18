import numpy as np
from oisatgmi.config import satellite_opt
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.interpolate import RBFInterpolator
from scipy import signal
from scipy.interpolate.interpnd import _ndim_coords_from_arrays
from scipy.spatial import cKDTree


def _interpolosis(interpol_func, Z: np.array, X: np.array, Y: np.array, interpolator_type: int, dists: np.array, threshold: float) -> np.array:
    # to make the interpolator() shorter
    if interpolator_type == 1:
        interpolator = LinearNDInterpolator(
            interpol_func, (Z).flatten(), fill_value=np.nan)
        ZZ = interpolator((X, Y))
        ZZ[dists > threshold*10.0] = np.nan
    elif interpolator_type == 2:
        interpolator = NearestNDInterpolator(interpol_func, (Z).flatten())
        ZZ = interpolator((X, Y))
        ZZ[dists > threshold*10.0] = np.nan
    elif interpolator_type == 3:
        interpolator = RBFInterpolator(
            interpol_func, (Z).flatten(), neighbors=5)
        XX = np.stack([X.ravel(), Y.ravel()], -1)
        ZZ = interpolator(XX)
        ZZ = ZZ.reshape(np.shape(X))
        ZZ[dists > threshold*3.0] = np.nan
    else:
        raise Exception(
            "other type of interpolation methods has not been implemented yet")
    return ZZ

def filler_gosatxch4(grid_size: float, sat_data, flag_thresh=0.75):
    '''
        The interpolator function
        Input:
            grid_size [float]: the size of grids defined by the user
            sat_data  [satellite_amf or satellite_opt]: a dataclass for satellite data
            flag_thresh [float]: the quality flag threshold
    '''
    # creating the delaunay triangulation on satellite coordinates
    # get the center lat/lon
    sat_center_lat = sat_data.latitude_center
    sat_center_lon = sat_data.longitude_center
    # mask bad data
    mask = sat_data.quality_flag > flag_thresh
    mask = np.multiply(mask, 1.0).squeeze()
    mask_for_interpolation = mask
    mask[mask != 1.0] = np.nan
    # mask clouds
    # define the triangulation
    points = np.zeros((np.size(sat_center_lat), 2))
    points[:, 0] = sat_center_lon.flatten()
    points[:, 1] = sat_center_lat.flatten()
    # if the points are not unique or weird, the convex hull can't be formed,
    # at this point, we can just skip this L2 file
    try:
        tri = Delaunay(points)
    except:
        return None
    # define the grid
    lat_ctm_min = -90.0
    lat_ctm_max =  90.0
    lon_ctm_min = -180.0
    lon_ctm_max =  180.0
    lon_grid = np.arange(lon_ctm_min, lon_ctm_max+grid_size, grid_size)
    lat_grid = np.arange(lat_ctm_min, lat_ctm_max+grid_size, grid_size)
    lons_grid, lats_grid = np.meshgrid(lon_grid.astype('float16'), lat_grid.astype('float16'))
    # calculate distance to remove too-far estimates
    tree = cKDTree(points)
    grid = np.zeros((2, np.shape(lons_grid)[0], np.shape(lons_grid)[1]))
    grid[0, :, :] = lons_grid
    grid[1, :, :] = lats_grid
    xi = _ndim_coords_from_arrays(tuple(grid), ndim=points.shape[1])
    dists, _ = tree.query(xi)

    latitude_center = lats_grid
    longitude_center = lons_grid
    # interpolate 2Ds fields
    print('....................... vcd')
    vcd =  _interpolosis(
        tri, sat_data.x_col*mask, lons_grid, lats_grid, 1, dists, grid_size)
    xch4 =  _interpolosis(
        tri, sat_data.x_col*mask, lons_grid, lats_grid, 1, dists, grid_size)

    print('....................... tropopause')
    tropopause = np.empty((1))

    print('....................... quality flag')
    quality_flag =  _interpolosis(
        tri, mask_for_interpolation, lons_grid, lats_grid, 2, dists, grid_size)

    print('....................... error')
    uncertainty = _interpolosis(
        tri, sat_data.uncertainty**2*mask, lons_grid, lats_grid, 1, dists, grid_size)
    uncertainty = np.sqrt(uncertainty)

    # interpolate 3Ds fields for optimal estimation algorithms (averaging kernels; e.g., MOPITT)
    if isinstance(sat_data, satellite_opt):
        averaging_kernels = np.zeros((np.shape(sat_data.pressure_mid)[0], np.shape(latitude_center)[0],
                                      np.shape(latitude_center)[1]))
        for z in range(0, np.shape(sat_data.pressure_mid)[0]):
            print('....................... AKs [' + str(z+1) +
                  '/' + str(np.shape(sat_data.pressure_mid)[0]) + ']')
            averaging_kernels[z, :, :] = _interpolosis(tri, sat_data.averaging_kernels[z, :].squeeze()
                                                                          * mask, lons_grid, lats_grid, 1, dists, grid_size)
        pressure_mid = np.zeros((np.shape(sat_data.pressure_mid)[0], np.shape(latitude_center)[0],
                                 np.shape(latitude_center)[1]))
        for z in range(0, np.shape(sat_data.pressure_mid)[0]):
            print('....................... pmids [' + str(z+1) +
                  '/' + str(np.shape(sat_data.pressure_mid)[0]) + ']')
            pressure_mid[z, :, :] = _interpolosis(tri, sat_data.pressure_mid[z, :].squeeze()
                                                                      * mask, lons_grid, lats_grid, 1, dists, grid_size)
        apriori_profile = np.zeros((np.shape(sat_data.pressure_mid)[0], np.shape(latitude_center)[0],
                                    np.shape(latitude_center)[1]))
        for z in range(0, np.shape(sat_data.pressure_mid)[0]):
            print('....................... apriori profile [' + str(z+1) +
                  '/' + str(np.shape(sat_data.pressure_mid)[0]) + ']')
            apriori_profile[z, :, :] = _interpolosis(tri, sat_data.apriori_profile[z, :].squeeze()
                                                                         * mask, lons_grid, lats_grid, 1, dists, grid_size)
        pressure_weights = np.zeros((np.shape(sat_data.pressure_mid)[0], np.shape(latitude_center)[0],
                                    np.shape(latitude_center)[1]))
        for z in range(0, np.shape(sat_data.pressure_mid)[0]):
            print('....................... pressure weights [' + str(z+1) +
                  '/' + str(np.shape(sat_data.pressure_mid)[0]) + ']')
            pressure_weights[z, :, :] = _interpolosis(tri, sat_data.pressure_weight[z, :].squeeze()
                                                                         * mask, lons_grid, lats_grid, 1, dists, grid_size)

    if isinstance(sat_data, satellite_opt):
        interpolated_sat = satellite_opt(vcd, sat_data.time, [], tropopause, latitude_center, longitude_center, [
        ], [], uncertainty, quality_flag, pressure_mid, averaging_kernels, [], [], [], [], 
        [], apriori_profile, [], [], xch4, pressure_weights)
    return interpolated_sat