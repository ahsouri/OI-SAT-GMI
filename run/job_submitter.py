import yaml
import os
import datetime
import numpy as np


def _daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + datetime.timedelta(n)


# Read the control file
with open('./control.yml', 'r') as stream:
    try:
        ctrl_opts = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise Exception(exc)

startdate = ctrl_opts['start_date']
enddate = ctrl_opts['end_date']
num_job = ctrl_opts['num_job']
python_bin = ctrl_opts['python_bin']

# convert dates to datetime
start_date = datetime.date(int(startdate[0:4]), int(
    startdate[5:7]),1)
end_date = datetime.date(int(enddate[0:4]), int(
    enddate[5:7]),26)
list_months = []
list_years = []
for single_date in _daterange(start_date, end_date):
    list_months.append(single_date.month)
    list_years.append(single_date.year)


list_months = np.array(list_months)
list_years = np.array(list_years)

# submit jobs per month per year (12 jobs per year)
if not os.path.exists('./jobs'):
    os.makedirs('./jobs')
for year in range(np.min(list_years), np.max(list_years)+1):
    for month in range(np.min(list_months), np.max(list_months)+1):
        # slurm command
        # Opening a file
        file = open('./jobs/' + 'job_' + str(year) +
                    '_' + str(month) + '.j', 'w')
        slurm_cmd = '#!/bin/bash \n'
        slurm_cmd += '#SBATCH -J oi_gmi \n'
        slurm_cmd += '#SBATCH --account=s1043 \n'
        slurm_cmd += '#SBATCH --ntasks=1 \n'
        slurm_cmd += '#SBATCH --cpus-per-task=' + str(int(num_job)) + ' \n'
        slurm_cmd += '#SBATCH --mem=170G \n'
        slurm_cmd += '#SBATCH -t 12:00:00 \n'
        slurm_cmd += '#SBATCH -o oi_gmi-%j.out \n'
        slurm_cmd += '#SBATCH -e oi_gmi-%j.err \n'
        slurm_cmd += python_bin + ' ./job.py ' + str(year) + ' ' + str(month)
        file.writelines(slurm_cmd)
        file.close()
        os.system('sbatch ' + './jobs/' + 'job_' + str(year) +
                    '_' + str(month) + '.j')
