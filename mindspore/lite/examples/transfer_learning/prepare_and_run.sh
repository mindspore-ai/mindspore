#!/bin/bash

display_usage()
{
  echo -e "\nUsage: prepare_and_run.sh -D dataset_path [-d mindspore_docker] [-r release.tar.gz] [-t arm64|x86] [-i device_id] [-o]\n"
}

checkopts()
{
  TARGET="arm64"
  DOCKER=""
  PLACES_DATA_PATH=""
  ENABLEFP16=""
  DEVICE_ID=""
  while getopts 'D:d:r:t:i:o' opt
  do
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
      o)
        ENABLEFP16="-o"
        ;;
      i)
        DEVICE_ID=$OPTARG
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
  if [ "${TARGET}" == "arm64" ]; then
    file=$(ls ../../../../output/mindspore-lite-*-android-aarch64.tar.gz)
  else
    file=$(ls ../../../../output/mindspore-lite-*-linux-x64.tar.gz)
  fi
  if [[ ${file} != "" ]] && [[ -f ${file} ]]; then
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
cp model/*.ms ${PACKAGE}/model || exit 1

# Copy the running script to the package
cp scripts/*.sh ${PACKAGE}/

# Copy the shared MindSpore ToD library
tar -xzf ${TARBALL}
mv mindspore-*/runtime/lib ${PACKAGE}/
mv mindspore-*/runtime/third_party/libjpeg-turbo/lib/* ${PACKAGE}/lib/
if [ "${TARGET}" == "arm64" ]; then
  tar -xzf ${TARBALL} --wildcards --no-anchored hiai_ddk
  mv mindspore-*/runtime/third_party/hiai_ddk/lib/* ${PACKAGE}/lib/
fi

rm -rf msl
mv mindspore-* msl/

# Convert the dataset into the package
./prepare_dataset.sh ${PLACES_DATA_PATH} || exit 1
cp -r dataset ${PACKAGE}

echo "==========Compiling============"
make clean
make TARGET=${TARGET}
 
# Copy the executable to the package
mv bin ${PACKAGE}/ || exit 1

if [ "${TARGET}" == "arm64" ]; then
  cp ${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so ${PACKAGE}/lib/ || exit 1

  if [ "${DEVICE_ID}" == "" ]; then
    echo "=======Pushing to device======="
    adb push ${PACKAGE} /data/local/tmp/

    echo "==Evaluating Untrained Model==="
    adb shell "cd /data/local/tmp/package-arm64 && /system/bin/sh eval_untrained.sh $ENABLEFP16"

    echo "========Training on Device====="
    adb shell "cd /data/local/tmp/package-arm64 && /system/bin/sh train.sh $ENABLEFP16"

    echo "===Evaluating trained Model====="
    adb shell "cd /data/local/tmp/package-arm64 && /system/bin/sh eval.sh $ENABLEFP16"
  else
    echo "=======Pushing to device======="
    adb -s ${DEVICE_ID} push ${PACKAGE} /data/local/tmp/

    echo "==Evaluating Untrained Model==="
    adb -s ${DEVICE_ID} shell "cd /data/local/tmp/package-arm64 && /system/bin/sh eval_untrained.sh $ENABLEFP16"

    echo "========Training on Device====="
    adb -s ${DEVICE_ID} shell "cd /data/local/tmp/package-arm64 && /system/bin/sh train.sh $ENABLEFP16"

    echo "===Evaluating trained Model====="
    adb -s ${DEVICE_ID} shell "cd /data/local/tmp/package-arm64 && /system/bin/sh eval.sh $ENABLEFP16"
  fi
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

