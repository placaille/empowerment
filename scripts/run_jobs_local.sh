#!/bin/bash
usage() {
  echo " Usage : $0 [-l running-locally] [-f force-overwrite] [-g <name-of-group>]"
}

# Flags and default values
LOCAL=false
FORCE_OVERWRITE=false
while getopts g:h flag; do
  case $flag in
    g) group_name=$OPTARG ;;
    h) usage; exit;;
    *) usage; exit;;
    ?) usage; exit;;
  esac
done

# Project specific values
out_dir=$SCRATCH/projects/augusta/jobs
tensorboard_dir=$SCRATCH/projects/augusta/tensorboard
python_file=src/train/fgan.py  # (will be called from job repository)

# Assume running this from the script directory
job_dir=$PWD/jobs
config_dir=$PWD/configs

if [ ! -z "$group_name" ]; then
  # if group_name is defined
  out_dir=$out_dir/$group_name
  tensorboard_dir=$tensorboard_dir/$group_name
fi

# Launch loop
for config_file in $config_dir/*.yml; do
  config_name=$(basename $config_file)
  timestamp=$(gdate +%s%3N) || timestamp=$(date +%s%3N)
  config_id="${config_name%.*}"
  unique_id=${config_id}-${timestamp}

  job_file=$job_dir/${unique_id}.job
  job_out_dir=$out_dir/$unique_id
  job_tensorboard_dir=$tensorboard_dir/$unique_id

  # copy current version of code
  echo Launching job $group_name/${config_name}..
  mkdir -p $job_out_dir
  mkdir -p $(dirname ${job_file})
  cp -r $PWD/src $job_out_dir
  cp $config_file $job_out_dir

  job_python_file=$job_out_dir/$python_file
  job_config_file=$job_out_dir/${config_name}
  python_args="-c $job_config_file"

  # add stuff to the run config
  echo "log-dir: $job_out_dir" >> $job_config_file
  echo "tensorboard-dir: $job_tensorboard_dir" >> $job_config_file

  python $job_python_file $python_args
done
