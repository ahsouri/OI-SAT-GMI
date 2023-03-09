#!bin/bash 
#SBATCH -J oi_gmi 
#SBATCH --constraint="[cas|sky]" 
#SBATCH --account=s1043 
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=24 
#SBATCH --mem=100G 
#SBATCH -t 2:00:00 
#SBATCH -o oi_gmi-%j.out 
#SBATCH -r oi_gmi-%j.err 
/home/asouri/anaconda3/bin/python3.9job.py 2019 5