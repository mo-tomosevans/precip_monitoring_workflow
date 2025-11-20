#!/bin/bash -l
#SBATCH --mem=25G
#SBATCH --ntasks=2
#SBATCH --time=280
#SBATCH --export=NONE
set -x; module load scitools 
module use /data/users/cfver/METBuild/modules
module load MET_Stable

run_metplus.py ${CYLC_SUITE_DEF_PATH}/app/${ROSE_TASK_APP}/run_series_analysis/bin/SeriesAnalysisMETplus_precip.ltg

##run_metplus.py SeriesAnalysisMETplus_precip.ltg