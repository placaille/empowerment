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

# Job specific values
job_defaults=(test)
read -p "Please enter the job names (default: test) > " -a job_names
job_names=${job_names:=${job_defaults}}

# Project specific values
out_dir=$SCRATCH/projects/augusta/jobs
other_cmds="source activate augusta"
python_file=src/mine_repro_distr.py  # (will be called from job repository)

# Assume running this from the script directory
job_dir=$PWD/.jobs
config_dir=$PWD/.configs

# Launch loop
for job_name in ${job_names[@]}; do
  job_file=$job_dir/${job_name}.job
  config_file=$config_dir/${job_name}.conf
  job_out_dir=$out_dir/$job_name

  # test if config file exists
  if [ ! -f "${config_file}" ]; then
    echo Config file $config_file doesn\'t exist, skipping job $job_name.
    continue
  fi

  # test if job file already in job directory (overwrite possible)
  if [ -d "${job_out_dir}" ]; then
    if [ -f "${job_out_dir}/${job_name}.job" ]; then
      if [ $FORCE_OVERWRITE = true ]; then
            echo $job_out_dir already existed and a copy was moved to $job_out_dir"_backup".
            echo WARNING: Backups are not backed up and could be overwritten.
            rm -rf $job_out_dir"_backup"
            mkdir -p $job_out_dir"_backup"
            mv $job_out_dir/* $job_out_dir"_backup"
      else
        echo Job $job_name already has ${job_out_dir}/${job_name}.job, skipping it. Relaunch with '-f' to overwrite.
        continue
      fi
    else
      echo Job $job_name has ${job_out_dir}, but no ${job_name}.job, deleting and relaunching.
      echo INFO: This is an indication sbatch wasn\'t successful.
      rm -rf $job_out_dir
    fi
  fi

  # copy current version of code
  echo Launching job $job_name..
  mkdir -p $job_out_dir
  mkdir -p $(dirname ${job_file})
  cp -r $PWD/src $job_out_dir
  cp $config_file $job_out_dir

  job_python_file=$job_out_dir/$python_file
  job_config_file=$job_out_dir/${job_name}.conf
  python_args="-c $job_config_file"

  # add stuff to the run config
  echo "log-dir: $job_out_dir" >> $job_config_file

  # set sbatch settings
  echo "#!/bin/bash
#SBATCH --job-name=$job_name.job
#SBATCH --output=$job_out_dir/slurm-%j.out
#SBATCH --error=$job_out_dir/slurm-%j.err
#SBATCH --nodelist=kepler2
#SBATCH --exclude=kepler4
#SBATCH --time=3:00:00
#SBATCH --mem=10G
$other_cmds
python $job_python_file $python_args" > $job_file

  if sbatch $job_file; then
    cp $job_file $job_out_dir
  fi
done
