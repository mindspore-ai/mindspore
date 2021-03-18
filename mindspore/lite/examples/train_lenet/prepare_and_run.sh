#!/bin/bash

display_usage()
{
  echo -e "\nUsage: prepare_and_run.sh -D dataset_path [-d mindspore_docker] [-r release.tar.gz] [-t arm64|x86]\n"
}

checkopts()
{
  TARGET="arm64"
  DOCKER=""
  MNIST_DATA_PATH=""
  while getopts 'D:d:r:t:' opt
  do
    case "${opt}" in
      D)
        MNIST_DATA_PATH=$OPTARG
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
if [ "$MNIST_DATA_PATH" == "" ]; then
  echo "MNIST Dataset directory path was not provided"
  display_usage
  exit 1
fi

if [ "$TARBALL" == "" ]; then
  if [ "${TARGET}" == "arm64" ]; then
    file=$(ls ../../../../output/mindspore-lite-*-train-android-aarch64.tar.gz)
  else
    file=$(ls ../../../../output/mindspore-lite-*-train-linux-x64.tar.gz)
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
mv mindspore-*/train/lib ${PACKAGE}/
mv mindspore-*/train/minddata/lib/* ${PACKAGE}/lib/
mv mindspore-*/train/minddata/third_party/libjpeg-turbo/lib/* ${PACKAGE}/lib/
if [ "${TARGET}" == "arm64" ]; then
  tar -xzf ${TARBALL} --wildcards --no-anchored hiai_ddk
  mv mindspore-*/train/third_party/hiai_ddk/lib/* ${PACKAGE}/lib/
fi

rm -rf msl
mv mindspore-* msl/

# Copy the dataset to the package
cp -r $MNIST_DATA_PATH ${PACKAGE}/dataset || exit 1

echo "==========Compiling============"
make TARGET=${TARGET}
 
# Copy the executable to the package
mv bin ${PACKAGE}/ || exit 1

if [ "${TARGET}" == "arm64" ]; then
  cp ${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so ${PACKAGE}/lib/ || exit 1

  echo "=======Pushing to device======="
  adb push ${PACKAGE} /data/local/tmp/

  echo "========Training on Device====="
  adb shell "cd /data/local/tmp/package-arm64 && /system/bin/sh train.sh"

  echo
  echo "===Evaluating trained Model====="
  adb shell "cd /data/local/tmp/package-arm64 && /system/bin/sh eval.sh"
  echo
else
  cd ${PACKAGE} || exit 1
  echo "======Training Locally========="
  ./train.sh

  echo "===Evaluating trained Model====="
  ./eval.sh
  cd ..
fi

