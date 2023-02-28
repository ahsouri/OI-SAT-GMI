import numpy as np
from scipy.io import savemat
from kneed import KneeLocator

def OI(Xa: np.array, Y: np.array, Sa: np.array, So: np.array, regularization_on = True):
    '''
    Optimal interpolation between two variables looking at the exact quantity
            under such condition, K = ones
            Xb = Xa + SaK^T(KSaK^T + So)^-1 * (Y-K*Xa)
    '''
    print('Optimal interpolation begins...')
    # negative values are not meaninful from the CTM perspective so:
    Y[Y<0] = 0.0
    if regularization_on == True:   
        scaling_factors = np.arange(0.1,10,0.1)
        scaling_factors = list(scaling_factors)
    else:
        scaling_factors = []
        scaling_factors.append(1.0)

    averaging_kernel_mean = []
    kalman_gain = []
    Sb = []
    averaging_kernel = []
    for reg in scaling_factors:                            
        kalman_gain_tmp = (Sa*(Sa+So*float(reg))**(-1))
        kalman_gain.append(kalman_gain_tmp)
        Sb_tmp = (np.ones_like(kalman_gain_tmp)-kalman_gain_tmp)*Sa
        Sb.append(Sb_tmp)
        AK = np.ones_like(Sb_tmp)-(Sb_tmp)/(Sa)
        AK[AK==0] = 0.0
        averaging_kernel.append(AK)
        averaging_kernel_mean.append(np.nanmean(AK.flatten()))

    if regularization_on == True:
        averaging_kernel_mean = np.array(averaging_kernel_mean)
        kneedle = KneeLocator(np.array(scaling_factors), averaging_kernel_mean, direction='decreasing')
        knee_index = np.argwhere(np.array(scaling_factors) == kneedle.knee)
    else:
        knee_index = []
        knee_index.append(0)

    kalman_gain = kalman_gain[int(knee_index[0])]
    averaging_kernel = averaging_kernel[int(knee_index[0])]
    Sb = Sb[int(knee_index[0])]
    increment = kalman_gain*(Y-Xa)
    Xb = Xa + increment

    #moutput = {}
    #moutput["kalman_gain"] = kalman_gain
    #moutput["increment"] = increment
    #moutput["Xb"] = Xb
    #moutput["Sb"] = Sb
    #moutput["Sa"] = Sa
    #moutput["Xa"] = Xa
    #moutput["So"] = So
    #moutput["Y"] = Y
    #moutput["averaging_kernel"] = averaging_kernel
    #savemat("oi_diag.mat", moutput)
    return Xb,averaging_kernel,increment,np.sqrt(Sb)