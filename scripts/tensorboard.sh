#!/bin/bash

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
if [ $(($ipnport)) -le "0" ] ; then
  ipnport=6006
fi
echo ipnport=$ipnport

ipnip=$(hostname -i)
echo ipnip=$ipnip

if [ "$FWD_HOME" = true ] ; then
  echo Forwarding port $ipnport to home:12900
  ssh -fNR :12900:localhost:${ipnport} home
fi

# launch tensorboard
tensorboard --logdir="${LOG_DIR}" --port=$ipnport
