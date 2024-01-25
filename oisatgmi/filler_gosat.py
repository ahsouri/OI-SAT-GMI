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
        ZZ[dists > threshold] = np.nan
    elif interpolator_type == 2:
        interpolator = NearestNDInterpolator(interpol_func, (Z).flatten())
        ZZ = interpolator((X, Y))
        ZZ[dists > threshold] = np.nan
    elif interpolator_type == 3:
        interpolator = RBFInterpolator(
            interpol_func, (Z).flatten(), neighbors=5)
        XX = np.stack([X.ravel(), Y.ravel()], -1)
        ZZ = interpolator(XX)
        ZZ = ZZ.reshape(np.shape(X))
        ZZ[dists > threshold] = np.nan
    else:
        raise Exception(
            "other type of interpolation methods has not been implemented yet")
    return ZZ

def _boxfilter(size_kernel_x, size_kernel_y) -> np.array:

    return np.ones((int(size_kernel_x), int(size_kernel_y)))/(size_kernel_x*size_kernel_y)


def _upscaler(X: np.array, Y: np.array, Z: np.array, ctm_models_coordinate: dict, grid_size: float, threshold: float, tri=None):
    '''
        upscaler function
        Input:
            X [2D]: x coordinates of the input (Z)
            Y [2D]: y coordinates of the input (Z)
            Z [2D]: Z values
            ctm_models_coordinate [dic]: a dictionary containing lat and lon of the model
            grid_size [float]: the size of grids defined by the user
            threshold [float]: any points with distance above this will be masked
    '''
    ctm_latitude = ctm_models_coordinate['Latitude']
    ctm_longitude = ctm_models_coordinate['Longitude']
    size_grid_model_lon = np.abs(ctm_longitude[0, 0]-ctm_longitude[0, 1])
    size_grid_model_lat = np.abs(ctm_latitude[0, 0] - ctm_latitude[1, 0])

    if (size_grid_model_lon >= grid_size) or (size_grid_model_lat >= grid_size):
        # upscaling is needed
        size_kernel_x = np.floor(size_grid_model_lon/grid_size)
        size_kernel_y = np.floor(size_grid_model_lat/grid_size)
        if size_kernel_x == 0 : size_kernel_x = 1
        if size_kernel_y == 0 : size_kernel_y = 1
        kernel = _boxfilter(size_kernel_y, size_kernel_x)
        Z = signal.convolve2d(Z, kernel, boundary='symm', mode='same')
        # define the triangulation
        points = np.zeros((np.size(X), 2))
        points[:, 0] = X.flatten()
        points[:, 1] = Y.flatten()
        if (tri is None):
            tri = Delaunay(points)
        # remove to far estimates
        tree = cKDTree(points)
        grid = np.zeros((2, np.shape(ctm_latitude)[
                        0], np.shape(ctm_latitude)[1]))
        grid[0, :, :] = ctm_longitude
        grid[1, :, :] = ctm_latitude
        xi = _ndim_coords_from_arrays(tuple(grid), ndim=points.shape[1])
        dists, _ = tree.query(xi)
        # interpolate
        Z = _interpolosis(tri, Z, ctm_longitude,
                          ctm_latitude, 1, dists, threshold)
        upscaled_ctm_needed = False
        return ctm_longitude, ctm_latitude, Z, upscaled_ctm_needed
    else:
        # upscaling is not needed but the model needs to match with sat
        upscaled_ctm_needed = True
        return X, Y, Z, upscaled_ctm_needed

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
    # fake a ctm for upscaling (1x1)
    lon_grid = np.arange(lon_ctm_min, lon_ctm_max+1.0, 0.1)
    lat_grid = np.arange(lat_ctm_min, lat_ctm_max+1.0, 0.1)
    lons_grid_ctm, lats_grid_ctm = np.meshgrid(lon_grid.astype('float16'), lat_grid.astype('float16'))
    ctm_models_coordinate = {}
    ctm_models_coordinate["Latitude"] = lats_grid_ctm
    ctm_models_coordinate["Longitude"] = lons_grid_ctm
    threshold_ctm = 1.0
    # calculate distance to remove too-far estimates
    tree = cKDTree(points)
    grid = np.zeros((2, np.shape(lons_grid)[0], np.shape(lons_grid)[1]))
    grid[0, :, :] = lons_grid
    grid[1, :, :] = lats_grid
    xi = _ndim_coords_from_arrays(tuple(grid), ndim=points.shape[1])
    dists, _ = tree.query(xi)

    print('Gridding GOSAT point data into 2D images...')
    # interpolate 2Ds fields
    print('....................... vcd')
    longitude_center,latitude_center,vcd,_ =  _upscaler(lons_grid, lats_grid,_interpolosis(
        tri, sat_data.x_col*mask, lons_grid, lats_grid, 1, dists, grid_size),
        ctm_models_coordinate, grid_size, threshold_ctm)
    _,_,xch4,_ =  _upscaler(lons_grid, lats_grid,_interpolosis(
        tri, sat_data.x_col*mask, lons_grid, lats_grid, 1, dists, grid_size),
        ctm_models_coordinate, grid_size, threshold_ctm)

    print('....................... tropopause')
    tropopause = np.empty((1))

    print('....................... quality flag')
    _,_,quality_flag,_ =  _upscaler(lons_grid, lats_grid,_interpolosis(
        tri, mask_for_interpolation, lons_grid, lats_grid, 2, dists, grid_size),
        ctm_models_coordinate, grid_size, threshold_ctm)

    print('....................... error')
    _,_,uncertainty,_ = _upscaler(lons_grid, lats_grid,_interpolosis(
        tri, sat_data.uncertainty**2*mask, lons_grid, lats_grid, 1, dists, grid_size),
        ctm_models_coordinate, grid_size, threshold_ctm)
    uncertainty = np.sqrt(uncertainty)

    # interpolate 3Ds fields for optimal estimation algorithms (averaging kernels; e.g., MOPITT)
    if isinstance(sat_data, satellite_opt):
        averaging_kernels = np.zeros((np.shape(sat_data.pressure_mid)[0], np.shape(latitude_center)[0],
                                      np.shape(latitude_center)[1]))
        for z in range(0, np.shape(sat_data.pressure_mid)[0]):
            print('....................... AKs [' + str(z+1) +
                  '/' + str(np.shape(sat_data.pressure_mid)[0]) + ']')
            _,_,averaging_kernels[z, :, :],_ = _upscaler(lons_grid, lats_grid,_interpolosis(tri, sat_data.averaging_kernels[z, :].squeeze()
                                                                          * mask, lons_grid, lats_grid, 1, dists, grid_size),
                                                                          ctm_models_coordinate, grid_size, threshold_ctm)
        pressure_mid = np.zeros((np.shape(sat_data.pressure_mid)[0], np.shape(latitude_center)[0],
                                 np.shape(latitude_center)[1]))
        for z in range(0, np.shape(sat_data.pressure_mid)[0]):
            print('....................... pmids [' + str(z+1) +
                  '/' + str(np.shape(sat_data.pressure_mid)[0]) + ']')
            _,_,pressure_mid[z, :, :],_ = _upscaler(lons_grid, lats_grid,_interpolosis(tri, sat_data.pressure_mid[z, :].squeeze()
                                                                      * mask, lons_grid, lats_grid, 1, dists, grid_size),
                                                                      ctm_models_coordinate, grid_size, threshold_ctm)
        apriori_profile = np.zeros((np.shape(sat_data.pressure_mid)[0], np.shape(latitude_center)[0],
                                    np.shape(latitude_center)[1]))
        for z in range(0, np.shape(sat_data.pressure_mid)[0]):
            print('....................... apriori profile [' + str(z+1) +
                  '/' + str(np.shape(sat_data.pressure_mid)[0]) + ']')
            _,_,apriori_profile[z, :, :],_ = _upscaler(lons_grid, lats_grid,_interpolosis(tri, sat_data.apriori_profile[z, :].squeeze()
                                                                         * mask, lons_grid, lats_grid, 1, dists, grid_size),
                                                                         ctm_models_coordinate, grid_size, threshold_ctm)
        pressure_weights = np.zeros((np.shape(sat_data.pressure_mid)[0], np.shape(latitude_center)[0],
                                    np.shape(latitude_center)[1]))
        for z in range(0, np.shape(sat_data.pressure_mid)[0]):
            print('....................... pressure weights [' + str(z+1) +
                  '/' + str(np.shape(sat_data.pressure_mid)[0]) + ']')
            _,_,pressure_weights[z, :, :],_ = _upscaler(lons_grid, lats_grid,_interpolosis(tri, sat_data.pressure_weight[z, :].squeeze()
                                                                         * mask, lons_grid, lats_grid, 1, dists, grid_size),
                                                                         ctm_models_coordinate, grid_size, threshold_ctm)
    if isinstance(sat_data, satellite_opt):
        interpolated_sat = satellite_opt(vcd, sat_data.time, [], tropopause, latitude_center, longitude_center, [
        ], [], uncertainty, quality_flag, pressure_mid, averaging_kernels, [], [], [], [], 
        np.empty((1)), apriori_profile, np.empty((1)), np.empty((1)), xch4, pressure_weights, 'GOSAT')
    return interpolated_sat
