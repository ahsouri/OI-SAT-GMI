import numpy as np
from oisatgmi.config import satellite_ssmis
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.interpolate import RBFInterpolator
from scipy import signal
from scipy.interpolate.interpnd import _ndim_coords_from_arrays
from scipy.spatial import cKDTree

# this is much more compact than the regular intrtpolator because it's straightforward to do it for SSMIS

def _interpolosis(interpol_func, Z: np.array, X: np.array, Y: np.array, interpolator_type: int, dists: np.array, threshold: float) -> np.array:
    # to make the interpolator() shorter
    if interpolator_type == 1:
        interpolator = LinearNDInterpolator(
            interpol_func, (Z).flatten(), fill_value=np.nan)
        ZZ = interpolator((X, Y))
        ZZ[dists > threshold*3.0] = np.nan
    elif interpolator_type == 2:
        interpolator = NearestNDInterpolator(interpol_func, (Z).flatten())
        ZZ = interpolator((X, Y))
        ZZ[dists > threshold*3.0] = np.nan
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

    if (size_grid_model_lon > grid_size) or (size_grid_model_lat > grid_size):
        # upscaling is needed
        size_kernel_x = np.floor(size_grid_model_lon/grid_size)
        size_kernel_y = np.floor(size_grid_model_lat/grid_size)
        if size_kernel_x == 0 : size_kernel_x = 1
        if size_kernel_y == 0 : size_kernel_y = 1
        kernel = _boxfilter(size_kernel_y, size_kernel_x)
        # TO DO, in case of large jumps from small pixels to large ones
        # you may exagerate the NaN boxes which seems to be a problem for
        # satelite observations like GOSAT having too many gaps so I need 
        # to figure out a better solution for this later
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


def interpolator_ssmis(interpolator_type: int, grid_size: float, sat_data, ctm_models_coordinate: dict):
    '''
        The interpolator function
        Input:
            interpolator_type [int]: an index specifying the type of interpolator
                    1 > Bilinear interpolation (recommended)
                    2 > Nearest neighbour
                    3 > RBF (thin_plate_spline)
                    4 > Poppy (not implemented yet)
            grid_size [float]: the size of grids defined by the user
            sat_data  [satellite_amf or satellite_opt]: a dataclass for satellite data
            ctm_models_coordinate [dic]: a dictionary containing lat and lon of the model
            flag_thresh [float]: the quality flag threshold
    '''
    # creating the delaunay triangulation on satellite coordinates
    # model lat and lon
    ctm_latitude = ctm_models_coordinate['Latitude']
    ctm_longitude = ctm_models_coordinate['Longitude']
    size_grid_model_lon = np.abs(ctm_longitude[0, 0]-ctm_longitude[0, 1])
    size_grid_model_lat = np.abs(ctm_latitude[0, 0] - ctm_latitude[1, 0])
    threshold_ctm = np.sqrt(size_grid_model_lon**2 + size_grid_model_lat**2)
    # get the center lat/lon
    sat_center_lat = sat_data.latitude_center
    sat_center_lon = sat_data.longitude_center
    # mask bad data
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
    lat_ctm_min = np.min(ctm_models_coordinate['Latitude'].flatten())
    lat_ctm_max = np.max(ctm_models_coordinate['Latitude'].flatten())
    lon_ctm_min = np.min(ctm_models_coordinate['Longitude'].flatten())
    lon_ctm_max = np.max(ctm_models_coordinate['Longitude'].flatten())

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
    # if RBF is chosen
    if interpolator_type == 3:
        tri = points
    # interpolate 2Ds fields
    print('....................... vcd')
    upscaled_X, upscaled_Y, vcd, upscaled_ctm_needed = _upscaler(lons_grid, lats_grid, _interpolosis(
        tri, sat_data.vcd, lons_grid, lats_grid, interpolator_type, dists, grid_size),
        ctm_models_coordinate, grid_size, threshold_ctm)

    print('....................... error')
    _, _, error, _ = _upscaler(lons_grid, lats_grid, _interpolosis(
        tri, sat_data.uncertainty, lons_grid, lats_grid, interpolator_type, dists, grid_size),
        ctm_models_coordinate, grid_size, threshold_ctm)
    
    latitude_center = upscaled_Y
    longitude_center = upscaled_X
 

    interpolated_sat = satellite_ssmis(vcd,error,sat_data.time,latitude_center,longitude_center,upscaled_ctm_needed,[],'SSMIS')

    return interpolated_sat
