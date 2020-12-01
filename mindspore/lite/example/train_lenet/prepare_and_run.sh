#!/bin/bash

display_usage() {
	echo -e "\nUsage: prepare_and_run.sh dataset_path [mindspore_docker] [release.tar.gz]\n"
	}

if [ -n "$1" ]; then
  MNIST_DATA_PATH=$1
else
  echo "MNIST Dataset directory path was not provided"
  display_usage
  exit 0
fi

if [ -n "$2" ]; then
  DOCKER=$2
else
  DOCKER=""
  #echo "MindSpore docker was not provided"
  #display_usage
  #exit 0
fi

if [ -n "$3" ]; then
  TARBALL=$3
else 
  if [ -f ../../../../output/mindspore-lite-*-runtime-arm64-cpu-train.tar.gz ]; then
    TARBALL="../../../../output/mindspore-lite-*-runtime-arm64-cpu-train.tar.gz"
  else
    echo "release.tar.gz was not found"
    display_usage
    exit 0
  fi
fi


# Prepare the model
cd model/
rm -f *.ms
./prepare_model.sh $DOCKER 
cd -

# Copy the .ms model to the package folder
rm -rf package
mkdir -p package/model
cp model/*.ms package/model

# Copy the running script to the package
cp scripts/train.sh package/
cp scripts/eval.sh package/

# Copy the shared MindSpore ToD library
tar -xzvf ${TARBALL} --wildcards --no-anchored libmindspore-lite.so
tar -xzvf ${TARBALL} --wildcards --no-anchored include
mv mindspore-*/lib package/
mkdir msl
mv mindspore-*/* msl/
rm -rf mindspore-*

# Copy the dataset to the package
cp -r ${MNIST_DATA_PATH} package/dataset

# Compile program
make TARGET=arm64
 
# Copy the executable to the package
mv bin package/

# Push the folder to the device
adb push package /data/local/tmp/

echo "Training on Device"
adb shell < scripts/run_train.sh

echo
echo "Load trained model and evaluate accuracy"
adb shell < scripts/run_eval.sh
echo

#rm -rf src/*.o package model/__pycache__ model/*.ms

#./prepare_and_run.sh /opt/share/dataset/mnist mindspore_dev:5
