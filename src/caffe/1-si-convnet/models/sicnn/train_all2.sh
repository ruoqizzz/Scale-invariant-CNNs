#!/bin/bash

TOOLS=../../build/tools
TIMESTAMP=$(date +"%b%d-%R")
MODEL=$(echo $1 | tr "[:upper:]" "[:lower:]") #farabet,sicnn,cnn

SOLVER=${MODEL}_solver_mnist-sc_split2.prototxt
LOG=log/${MODEL}_mnist-sc_split2_${TIMESTAMP}
echo "Running $SOLVER log to $LOG"

time $TOOLS/caffe train --solver=${SOLVER} |& tee -a $LOG

