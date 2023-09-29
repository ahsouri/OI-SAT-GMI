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
#hard coding for now
num_job = 24
python_bin = ctrl_opts['python_bin']
debug_on = ctrl_opts['debug']

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
os.system('rm -rf temp/*.png')

for year in range(np.min(list_years), np.max(list_years)+1):
    for month in range(np.min(list_months), np.max(list_months)+1):
        # slurm command
        # Opening a file
        # slurm command --> changed them to PBS job script
        # Opening a file
        file = open('./jobs/' + 'job_' + str(year) +
                    '_' + str(month) + '.j', 'w')
        slurm_cmd = '#!/bin/bash \n'
        slurm_cmd += '#PBS -l select=6:ncpus=4:mpiprocs=4:model=ivy  \n'
        slurm_cmd += '#PBS -l walltime=3:00:00  \n'
        slurm_cmd += '#PBS -N oi_gmi  \n'
        slurm_cmd += '#PBS -j oe  \n'
        slurm_cmd += '#PBS -m abe  \n'
        slurm_cmd += '#PBS -o oi_gmi.out  \n'
        slurm_cmd += '#PBS -e oi_gmi.err  \n'
        slurm_cmd += '#PBS -W group_list=s1395 \n'
        slurm_cmd += 'cd $PBS_O_WORKDIR  \n'
        if debug_on:
            slurm_cmd += '#PBS -q devel  \n'
        slurm_cmd += python_bin + ' ./job.py ' + str(year) + ' ' + str(month)
        file.writelines(slurm_cmd)
        file.close()
        os.system('qsub ' + './jobs/' + 'job_' + str(year) +
                    '_' + str(month) + '.j')
