#!/bin/bash

display_usage()
{
  echo -e "\nUsage: prepare_and_run.sh -D dataset_path [-d mindspore_docker] [-r release.tar.gz] [-t arm64|x86]\n"
}

checkopts()
{
  TARGET="arm64"
  DOCKER=""
  PLACES_DATA_PATH=""
  while getopts 'D:d:r:t:' opt
  do
    OPTARG=$(echo ${OPTARG} | tr '[A-Z]' '[a-z]')
    case "${opt}" in
      D)
        PLACES_DATA_PATH=$OPTARG
        ;;
      d)
        DOCKER=$OPTARG
        ;;
      t)
        if [ "$OPTARG" == "arm64" ] || [ "$OPTARG" == "x86" ]; then
          TARGET=$OPTARG
        else
          echo "No such target " $OPTARG
          display_usage
          exit 1
        fi
        ;;
      r)
        TARBALL=$OPTARG
        ;;
      *)
        echo "Unknown option ${opt}!"
        display_usage
        exit 1
    esac
  done
}
checkopts "$@"
if [ "$PLACES_DATA_PATH" == "" ]; then
  echo "Places Dataset directory path was not provided"
  display_usage
  exit 1
fi

if [ "$TARBALL" == "" ]; then
  file=$(ls ../../../../output/mindspore-lite-*-runtime-${TARGET}-cpu-train.tar.gz)
  if [ -f ${file} ]; then
    TARBALL=${file}
  else
    echo "release.tar.gz was not found"
    display_usage
    exit 1
  fi
fi

# Prepare the model
cd model/ || exit 1
rm -f *.ms
./prepare_model.sh $DOCKER || exit 1
cd ../

# Copy the .ms model to the package folder

PACKAGE=package-${TARGET}

rm -rf ${PACKAGE}
mkdir -p ${PACKAGE}/model
cp model/*.ms ${PACKAGE}/model

# Copy the running script to the package
cp scripts/*.sh ${PACKAGE}/

# Copy the shared MindSpore ToD library
tar -xzf ${TARBALL} --wildcards --no-anchored libmindspore-lite.so
tar -xzf ${TARBALL} --wildcards --no-anchored include
mv mindspore-*/lib ${PACKAGE}/
rm -rf msl
mkdir msl
mv mindspore-*/* msl/
rm -rf mindspore-*

# Convert the dataset into the package
./prepare_dataset.sh ${PLACES_DATA_PATH}
cp -r dataset ${PACKAGE}

echo "==========Compiling============"
make TARGET=${TARGET}
 
# Copy the executable to the package
mv bin ${PACKAGE}/ || exit 1

if [ "${TARGET}" == "arm64" ]; then
  echo "=======Pushing to device======="
  adb push ${PACKAGE} /data/local/tmp/

  echo "==Evaluating Untrained Model==="
  adb shell "cd /data/local/tmp/package-arm64 && /system/bin/sh eval_untrained.sh"

  echo "========Training on Device====="
  adb shell "cd /data/local/tmp/package-arm64 && /system/bin/sh train.sh"

  echo
  echo "===Evaluating trained Model====="
  adb shell "cd /data/local/tmp/package-arm64 && /system/bin/sh eval.sh"
  echo
else
  cd ${PACKAGE} || exit 1
  echo "==Evaluating Untrained Model==="
  ./eval_untrained.sh

  echo "======Training Locally========="
  ./train.sh

  echo "===Evaluating trained Model====="
  ./eval.sh
  cd ..
fi

