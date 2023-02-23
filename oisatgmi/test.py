from driver import oisatgmi
from pathlib import Path
import numpy as np
from scipy.io import savemat

oisatgmi_obj = oisatgmi()
oisatgmi_obj.read_data('GMI', Path('download_bucket/gmi/'), ['NO2'], '3-hourly', 'TROPOMI_NO2',
                       Path('download_bucket/trop_no2/subset'))



oisatgmi_obj.recal_amf()
oisatgmi_obj.average('2020-06-01','2020-07-01','monthly')



exit()
latitude = oisatgmi_obj.reader_obj.tropomi_data[0].latitude_center
longitude = oisatgmi_obj.reader_obj.tropomi_data[0].longitude_center

output = np.zeros((np.shape(latitude)[0], np.shape(
        latitude)[1], len(oisatgmi_obj.reader_obj.tropomi_data)))
counter = -1
for trop in oisatgmi_obj.reader_obj.tropomi_data:
    counter = counter + 1
    output[:, :, counter] = trop.vcd

    output[output <= 0.0] = np.nan

moutput = {}
moutput["vcds_old"] = output
moutput["lat"] = latitude
moutput["lon"] = longitude

output = np.zeros((np.shape(latitude)[0], np.shape(
        latitude)[1], len(oisatgmi_obj.reader_obj.tropomi_data)))
counter = -1
for trop in oisatgmi_obj.reader_obj.tropomi_data:
    counter = counter + 1
    output[:, :, counter] = trop.vcd

    output[output <= 0.0] = np.nan

moutput["vcds_new"] = output
savemat("vcds.mat", moutput)