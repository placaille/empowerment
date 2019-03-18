#!/bin/bash
usage() {
  echo " Usage : $0 [-g <name-of-group>]"
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
python_file=src/train/fgan_gumbel_distr.py  # (will be called from job repository)

# Assume running this from the script directory
job_dir=$PWD/.jobs
config_dir=$PWD/.configs

if [ ! -z "$group_name" ]; then
  # if group_name is defined
  out_dir=$out_dir/$group_name
  tensorboard_dir=$tensorboard_dir/$group_name
fi

# Launch loop
for config_file in $config_dir/*.conf; do
  config_name=$(basename $config_file)
  timestamp=$(date +%s%3N)
  config_id="${config_name%.*}"
  unique_id=${config_id}-${timestamp}

  job_file=$job_dir/${unique_id}.job
  job_out_dir=$out_dir/$unique_id
  job_tensorboard_dir=$tensorboard_dir/$unique_id
  job_tmp_dir=\$TMPDIR/$unique_id

  # copy current version of code
  echo Launching job $group_name/${unique_id}..
  mkdir -p $job_out_dir/out
  mkdir -p $(dirname ${job_file})
  cp -r $PWD/src $job_out_dir
  cp $config_file $job_out_dir

  job_python_file=$job_tmp_dir/$python_file
  job_config_file=$job_out_dir/${config_name}
  python_args="-c $job_config_file"

  # other cmds to add to .job
  other_cmds="
# check if job already exists, if so, delete and relaunch (terminated job)
# slurm_files=($job_out_dir/slurm*)
# if [ -e \${slurm_files[0]} ]; then
#   echo job relaunched, deleting previous results..
#   echo \${slurm_files[0]}
#   rm -r $job_out_dir/out/*
#   rm -r $job_tensorboard_dir/*
# fi

mkdir -p $job_tmp_dir
cp -r $job_out_dir/src $job_tmp_dir
source activate augusta
"

  # add stuff to the run config
  echo "log-dir: $job_out_dir/out" >> $job_config_file
  echo "tensorboard-dir: $job_tensorboard_dir" >> $job_config_file

  # set sbatch settings
  echo "#!/bin/bash
#SBATCH --job-name=$config_name.job
#SBATCH --output=$job_out_dir/slurm-%j.out
#SBATCH --error=$job_out_dir/slurm-%j.err
#SBATCH --time=2:00:00
#SBATCH --mem=4G
#SBATCH -c 1
#SBATCH --qos=low
#SBATCH --requeue
#SBATCH --exclude=leto52,leto20,leto12
$other_cmds
python -u $job_python_file $python_args" > $job_file

  # sbatch $job_file
  if sbatch $job_file; then
    cp $job_file $job_out_dir
    rm $config_file
  fi
  sleep 1
done
