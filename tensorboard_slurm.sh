#!/usr/bin/env bash
# originally taken from
# https://gist.github.com/taylorpaul/250ee3ed2524e8c28ee7c58ed656a5b9

usage() {
»       echo " Usage : $0 [-l <log_dir>]"
}

LOG_DIR=""
while getopts l: flag; do
  case $flag in
    l) LOG_DIR=$OPTARG ;;
    *) usage; exit;;
    ?) usage; exit;;
  esac
done

source activate var-info-max

let ipnport=($UID-6025)%65274
echo ipnport=$ipnport

ipnip=$(hostname -i)
echo ipnip=$ipnip

tensorboard --logdir="${LOG_DIR}" --port=$ipnport
