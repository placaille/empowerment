#!/usr/bin bash
# originally taken from
# https://gist.github.com/taylorpaul/250ee3ed2524e8c28ee7c58ed656a5b9

usage() {
»       echo " Usage : $0 [-l <log_dir>] [-f forward-home]"
}

LOG_DIR=""
FWD_HOME=false
while getopts l:f flag; do
  case $flag in
    f) FWD_HOME=true ;;
    l) LOG_DIR=$OPTARG ;;
    *) usage; exit;;
    ?) usage; exit;;
  esac
done

source activate tensorboard

let ipnport=($UID-6025)%65274
echo ipnport=$ipnport

ipnip=$(hostname -i)
echo ipnip=$ipnip

if [ "$FWD_HOME" = true ] ; then
  echo Forwarding port $ipnport to home:12900
  ssh -fNR :12900:localhost:${ipnport} home
fi

# launch tensorboard
tensorboard --logdir="${LOG_DIR}" --port=$ipnport
