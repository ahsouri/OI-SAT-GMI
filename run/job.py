import yaml
from oisatgmi import oisatgmi
from pathlib import Path
import sys

# Read the control file
with open('run/control.yml', 'r') as stream:
    try:
        ctrl_opts = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise Exception(exc)

ctm_name = ctrl_opts['ctm_name']
ctm_dir = ctrl_opts['ctm_dir']
ctm_freq = ctrl_opts['ctm_freq']
gas = ctrl_opts['gas']
sensor = ctrl_opts['sensor']
read_AK = ctrl_opts['read_AK']
troposphere_only = ctrl_opts['troposphere_only']
sat_dir = ctrl_opts['sat_dir']
output_pdf_dir = ctrl_opts['output_pdf_dir']
num_job = ctrl_opts['num_job']

year = sys.argv[1]
month = sys.argv[2]

oisatgmi_obj = oisatgmi()
oisatgmi_obj.read_data(ctm_name, Path(ctm_dir), [gas], ctm_freq, sensor+'_'+gas,
                       Path(sat_dir), str(year) + f"{month:02}", read_ak=read_AK,
                       trop=troposphere_only, num_job=int(num_job))
oisatgmi_obj.recal_amf()
oisatgmi_obj.average('2019-05-01', '2019-06-01')
oisatgmi_obj.oi()
oisatgmi_obj.reporting('NO2_201905')
oisatgmi_obj.write_to_nc('NO2_201905')
