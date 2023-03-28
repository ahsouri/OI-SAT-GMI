import yaml
from oisatgmi import oisatgmi
from pathlib import Path
import sys

# Read the control file
with open('./control.yml', 'r') as stream:
    try:
        ctrl_opts = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise Exception(exc)

ctm_name = ctrl_opts['ctm_name']
ctm_dir = ctrl_opts['ctm_dir']
ctm_freq = ctrl_opts['ctm_freq']
ctm_avg = ctrl_opts['ctm_avg']
gas = ctrl_opts['gas']
sensor = ctrl_opts['sensor']
read_AK = ctrl_opts['read_AK']
troposphere_only = ctrl_opts['troposphere_only']
sat_dir = ctrl_opts['sat_dir']
output_pdf_dir = ctrl_opts['output_pdf_dir']
output_nc_dir = ctrl_opts['output_nc_dir']
num_job = ctrl_opts['num_job']
error_ctm = ctrl_opts['ctm_error']

year = int(sys.argv[1])
month = int(sys.argv[2])

oisatgmi_obj = oisatgmi()
oisatgmi_obj.read_data(ctm_name, Path(ctm_dir), gas, ctm_freq, sensor+'_'+gas,
                       Path(sat_dir), str(year) + f"{month:02}", averaged=ctm_avg, read_ak=read_AK,
                       trop=troposphere_only, num_job=int(num_job))
oisatgmi_obj.recal_amf()
if month != 12:
    oisatgmi_obj.average(str(
        year) + '-' + f"{month:02}" + '-01', str(year) + '-' + f"{month+1:02}" + '-01')
else:
    oisatgmi_obj.average(
        str(year) + '-' + f"{month:02}" + '-01', str(year+1) + '-' + "01" + '-01')
oisatgmi_obj.oi(error_ctm=error_ctm)
oisatgmi_obj.reporting(gas + '_' + str(year) + f"{month:02}", gas, output_pdf_dir)
oisatgmi_obj.write_to_nc(gas + '_' + str(year) + f"{month:02}", output_nc_dir)
