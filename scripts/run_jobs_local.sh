#!/bin/bash
usage() {
  echo " Usage : $0 [-l running-locally] [-f force-overwrite] [-g <name-of-group>]"
}

# Flags and default values
LOCAL=false
FORCE_OVERWRITE=false
while getopts g:h flag; do
  case $flag in
    g) ARG_GROUP_NAME=$OPTARG ;;
    h) usage; exit;;
    *) usage; exit;;
    ?) usage; exit;;
  esac
done

# Project specific values
out_dir=$SCRATCH/projects/augusta/jobs
tensorboard_dir=$SCRATCH/projects/augusta/tensorboard
python_file=src/train/fgan_gumbel_distr.py  # (will be called from job repository)

# Assume running this from the script directory
job_dir=$PWD/.jobs
config_dir=$PWD/.configs

# naming group
if [ -z "$ARG_GROUP_NAME" ]; then
  read -t 10 -p "Enter group name, if necessary (10 secs) > " group_name
else
  group_name=$ARG_GROUP_NAME
fi
if [ ! -z "$group_name" ]; then
  # if group_name is defined
  out_dir=$out_dir/$group_name
  tensorboard_dir=$tensorboard_dir/$group_name
fi

# Launch loop
for config_file in $config_dir/*.conf; do
  config_name=$(basename $config_file)
  timestamp=$(gdate +%s%3N) || timestamp=$(date +%s%3N)
  job_file=$job_dir/${job_name}.job
  job_out_dir=$out_dir/$timestamp
  job_tensorboard_dir=$tensorboard_dir/$timestamp

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
