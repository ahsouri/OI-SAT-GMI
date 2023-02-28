import os.path
import glob
import numpy as np
from fpdf import FPDF
from test_plotter import test_plotter

def report(lon: np.ndarray, lat: np.ndarray, ctm_vcd_before: np.ndarray, ctm_vcd_after: np.ndarray, 
            sat_vcd: np.ndarray, increment: np.ndarray, averaging_kernel: np.ndarray, error_OI: np.ndarray, fname):
    '''
    '''
    test_plotter(lon,lat,ctm_vcd_before,'temp/ctm_vcd_before_' + fname + '.png','CTM VCD (prior)',1)
    test_plotter(lon,lat,ctm_vcd_after,'temp/ctm_vcd_after_' + fname+ '.png','CTM VCD (posterior)',1)
    test_plotter(lon,lat,sat_vcd,'temp/sat_vcd_' + fname+ '.png', 'Satellite Observation (Y)',1)
    test_plotter(lon,lat,increment,'temp/increment_' + fname+ '.png', 'Increment',1)
    test_plotter(lon,lat,averaging_kernel,'temp/ak_' + fname+ '.png', 'Averaging Kernels',2)
    test_plotter(lon,lat,error_OI,'temp/error_' + fname+ '.png', 'OI estimate error',1)



    