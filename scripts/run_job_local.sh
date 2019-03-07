#!/bin/bash
usage() {
  echo " Usage : $0 [-l running-locally] [-f force-overwrite]"
}

# Flags and default values
LOCAL=false
FORCE_OVERWRITE=false
while getopts fhl flag; do
  case $flag in
    l) LOCAL=true ;;
    f) FORCE_OVERWRITE=true ;;
    h) usage; exit;;
    *) usage; exit;;
    ?) usage; exit;;
  esac
done

# Project specific values
out_dir=$SCRATCH/projects/augusta/jobs
python_file=src/train/fgan_gumbel_distr.py  # (will be called from job repository)

# Assume running this from the script directory
job_dir=$PWD/.jobs
config_dir=$PWD/.configs

# naming group
read -t 10 -p "Enter group name, if necessary (10 secs) > " group_name
if [ ! -z "$group_name" ]; then
  # if group_name is defined
  out_dir=$out_dir/$group_name
fi

# Launch loop
for config_file in $config_dir/*.conf; do
  config_name=$(basename $config_file)
  timestamp=$(gdate +%s%3N)
  job_file=$job_dir/${job_name}.job
  job_out_dir=$out_dir/$timestamp

  # copy current version of code
  echo Launching job with $config_file..
  mkdir -p $job_out_dir
  mkdir -p $(dirname ${job_file})
  cp -r $PWD/src $job_out_dir
  cp $config_file $job_out_dir

  job_python_file=$job_out_dir/$python_file
  job_config_file=$job_out_dir/${config_name}
  python_args="-c $job_config_file"

  # add stuff to the run config
  echo "log-dir: $job_out_dir" >> $job_config_file

  python $job_python_file $python_args
done