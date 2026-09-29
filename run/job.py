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
if ctm_name == "FREE":
    read_AK = "False"
else:
    read_AK = ctrl_opts['read_AK']
mcip_dir = ctrl_opts['mcip_dir']
troposphere_only = ctrl_opts['troposphere_only']
sat_dir = ctrl_opts['sat_dir']
output_pdf_dir = ctrl_opts['output_pdf_dir']
output_nc_dir = ctrl_opts['output_nc_dir']
num_job = ctrl_opts['num_job']
error_ctm = ctrl_opts['ctm_error']
save_daily = ctrl_opts['save_daily']
split_month = ctrl_opts['split_month']

year = int(sys.argv[1])
month = int(sys.argv[2])
if sensor == "TEMPO":
   hour = int(sys.argv[3])

# if TEMPO is selected
if sensor == "TEMPO":
   try:
      oisatgmi_obj = oisatgmi()
      YYYYMM = str(year) + f"{month:02}"
      oisatgmi_obj.read_data(
        ctm_name,
        Path(ctm_dir),
        gas,
        ctm_freq,
        sensor + '_' + gas,
        Path(sat_dir),
        YYYYMM,
        averaging=ctm_avg,
        read_ak=read_AK,
        trop=troposphere_only,
        num_job=int(num_job),
        mcip_dir=Path(mcip_dir),
        tempo_hour=hour
      )
      oisatgmi_obj.recal_amf(num_job=int(num_job))
      if save_daily:
         oisatgmi_obj.savedaily(
            './diag_mat/',
            gas,
            YYYYMM + '_' + str(hour) + 'UTC'
         )
      # ---------------------------------------------------------
      # Define beginning of next month
      # ---------------------------------------------------------
      if month != 12:
         next_year = year
         next_month = month + 1
      else:
         next_year = year + 1
         next_month = 1

      next_month_date = (
        str(next_year) + '-' +
        f"{next_month:02}" + '-01'
      )
      # =========================================================
      # SPLIT-MONTH MODE
      # =========================================================
      if split_month:
         periods = [
            (
                str(year) + '-' + f"{month:02}" + '-01',
                str(year) + '-' + f"{month:02}" + '-16',
                YYYYMM + '1'
            ),
            (
                str(year) + '-' + f"{month:02}" + '-16',
                next_month_date,
                YYYYMM + '2'
            )
         ]
      # =========================================================
      # NORMAL FULL-MONTH MODE
      # =========================================================
      else:
         periods = [
            (
                str(year) + '-' + f"{month:02}" + '-01',
                next_month_date,
                YYYYMM
            )
         ]
      # ---------------------------------------------------------
      # Process each averaging period
      # ---------------------------------------------------------
      for start_date, end_date, period_name in periods:

         print(
            f"Processing averaging period: "
            f"{start_date} to {end_date} "
            f"as {period_name}"
         )
         oisatgmi_obj.average(
            start_date,
            end_date,
            gasname=gas
         )
         oisatgmi_obj.bias_correct(
            sensor,
            gas
         )
         oisatgmi_obj.oi(
            sensor,
            error_ctm=error_ctm,
            regularization=False
         )
         output_name = (
            gas + '_' +
            period_name + '_' +
            f"{hour:02}" + '_UTC'
         )
         oisatgmi_obj.reporting(
            output_name,
            gas,
            output_pdf_dir
         )

         oisatgmi_obj.write_to_nc(
            output_name,
            output_nc_dir
         )

   except Exception as e:
     print(f"Error processing hour {hour}: {e}")
   exit() # we should skip the rest

oisatgmi_obj = oisatgmi()
oisatgmi_obj.read_data(ctm_name, Path(ctm_dir), gas, ctm_freq, sensor+'_'+gas,
                       Path(sat_dir), str(year) + f"{month:02}", averaging=ctm_avg, read_ak=read_AK,
                       trop=troposphere_only, num_job=int(num_job), mcip_dir=Path(mcip_dir))
if sensor == "MOPITT":
   oisatgmi_obj.conv_ak(sensor)
elif sensor == "GOSAT":
   oisatgmi_obj.conv_ak(sensor)
elif sensor == "SSMIS":
   oisatgmi_obj.cal_pwv()
else:
   oisatgmi_obj.recal_amf()

if save_daily:
   oisatgmi_obj.savedaily(output_nc_dir, gas,
                           str(year) + '_' + f"{month:02}")
if month != 12:
    oisatgmi_obj.average(str(
        year) + '-' + f"{month:02}" + '-01', str(year) + '-' + f"{month+1:02}" + '-01', gasname=gas)
else:
    oisatgmi_obj.average(
        str(year) + '-' + f"{month:02}" + '-01', str(year+1) + '-' + "01" + '-01', gasname=gas)
oisatgmi_obj.bias_correct(sensor,gas)
oisatgmi_obj.oi(sensor, error_ctm=error_ctm)
oisatgmi_obj.reporting(gas + '_' + str(year) + f"{month:02}", gas, output_pdf_dir)
oisatgmi_obj.write_to_nc(gas + '_' + str(year) + f"{month:02}", output_nc_dir)
