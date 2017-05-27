#!/usr/bin/env sh
export PYTHONPATH=./
MODEL_NAME=face_gan_deploy
if [ ! -d "./model_cache/$MODEL_NAME" ]; then
  echo "mkdir ./model_cache/$MODEL_NAME" 
  mkdir -p ./model_cache/$MODEL_NAME
fi
../build/tools/caffe.bin train \
    -solver ./face_solver.prototxt \
    -gpu 3 \
    -weights ./model_cache/face_gan/face_gan_iter_280000.caffemodel \
    2>&1| tee ./model_cache/$MODEL_NAME/log.txt
