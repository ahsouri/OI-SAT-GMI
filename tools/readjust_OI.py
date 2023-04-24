import numpy as np
from pathlib import Path
import sys
import os
import glob
from netCDF4 import Dataset
import warnings
from oisatgmi.optimal_interpolation import OI
from numpy import dtype

warnings.filterwarnings("ignore", category=RuntimeWarning)


def _read_nc(filename, var):
    # reading nc files without a group
    nc_f = filename
    nc_fid = Dataset(nc_f, 'r')
    out = np.array(nc_fid.variables[var])
    nc_fid.close()
    return np.squeeze(out)

# inputs
oi_diag_folder = Path(str(sys.argv[1]))
oi_diag_files = sorted(
    glob.glob(oi_diag_folder.as_posix() + "/*.nc"))
ext_files_folder = Path(str(sys.argv[2]))
if not os.path.exists(ext_files_folder.as_posix()):
    os.makedirs(ext_files_folder.as_posix())


for fname in oi_diag_files:
    # read the OI diag files
    print("Now processing " + fname)
    lat = _read_nc(fname, 'lat')
    lon = _read_nc(fname, 'lon')
    old_amf = _read_nc(fname, 'old_amf')
    new_amf = _read_nc(fname, 'new_amf')

    sat_averaged_vcd = _read_nc(fname, 'sat_averaged_vcd')
    sat_averaged_error = _read_nc(fname, 'sat_averaged_error')
    ctm_averaged_vcd = _read_nc(fname, 'ctm_averaged_vcd_prior')
    error_ctm = 50.0
    fbasename = os.path.basename(fname)

    ctm_averaged_vcd_corrected, ak_OI, increment_OI, error_OI = OI(ctm_averaged_vcd, sat_averaged_vcd
                                                                 (ctm_averaged_vcd*error_ctm/100.0)**2, 
                                                                 sat_averaged_error**2, regularization_on=True)
    
    
    # Write the final results to a netcdf
    ncfile = Dataset(ext_files_folder + '/' + fbasename + '.nc', 'w')

    # create the x and y dimensions.
    ncfile.createDimension('x', np.shape(sat_averaged_vcd)[0])
    ncfile.createDimension('y', np.shape(sat_averaged_vcd)[1])

    data1 = ncfile.createVariable(
            'sat_averaged_vcd', dtype('float32').char, ('x', 'y'))
    data1[:, :] = sat_averaged_vcd

    data2 = ncfile.createVariable(
            'ctm_averaged_vcd_prior', dtype('float32').char, ('x', 'y'))
    data2[:, :] = ctm_averaged_vcd

    data3 = ncfile.createVariable(
            'ctm_averaged_vcd_posterior', dtype('float32').char, ('x', 'y'))
    data3[:, :] = ctm_averaged_vcd_corrected

    data4 = ncfile.createVariable(
            'sat_averaged_error', dtype('float32').char, ('x', 'y'))
    data4[:, :] = sat_averaged_error

    data5 = ncfile.createVariable(
            'ak_OI', dtype('float32').char, ('x', 'y'))
    data5[:, :] = ak_OI

    data6 = ncfile.createVariable(
            'error_OI', dtype('float32').char, ('x', 'y'))
    data6[:, :] = error_OI

    scaling_factor = ctm_averaged_vcd_corrected/ctm_averaged_vcd
    scaling_factor[np.where((np.isnan(scaling_factor)) | (np.isinf(scaling_factor)) |
                       (scaling_factor == 0.0))] = 1.0
    data7 = ncfile.createVariable(
            'scaling_factor', dtype('float32').char, ('x', 'y'))
    data7[:, :] = scaling_factor

    data8 = ncfile.createVariable(
            'lon', dtype('float32').char, ('x', 'y'))
    data8[:, :] = lon

    data9 = ncfile.createVariable(
            'lat', dtype('float32').char, ('x', 'y'))
    data9[:, :] = lat

    data10 = ncfile.createVariable(
            'old_amf', dtype('float32').char, ('x', 'y'))
    data10[:, :] = old_amf

    data11 = ncfile.createVariable(
            'new_amf', dtype('float32').char, ('x', 'y'))
    data11[:, :] = new_amf

    ncfile.close()
