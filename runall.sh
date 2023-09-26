#!/bin/bash
for dir in */; do
  cd $dir
  sbatch runscript
  cd ..
done
