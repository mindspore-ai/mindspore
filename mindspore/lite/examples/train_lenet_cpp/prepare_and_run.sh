#!/bin/bash

display_usage()
{
  echo -e "\nUsage: prepare_and_run.sh -D dataset_path [-d mindspore_docker] [-r release.tar.gz] [-t arm64|x86] [-o] [-b virtual_batch] [-m mindir] [-e epochs_to_train] [-i device_id]\n"
}

checkopts()
{
  TARGET="arm64"
  DOCKER=""
  MNIST_DATA_PATH=""
  ENABLEFP16=""
  VIRTUAL_BATCH=-1
  MINDIR_FILE=""
  EPOCHS="-e 5"
  DEVICE_ID=""
  while getopts 'D:d:e:m:r:t:i:ob:' opt
  do
    case "${opt}" in
      D)
        MNIST_DATA_PATH=$OPTARG
        ;;
      d)
        DOCKER=$OPTARG
        ;;
      e)
        EPOCHS="-e $OPTARG"
        ;;
      m)
        MINDIR_FILE=$OPTARG
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
      b)
        VIRTUAL_BATCH=$OPTARG
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
if [ "$MNIST_DATA_PATH" == "" ]; then
  echo "MNIST Dataset directory path was not provided"
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
if [[ "${VIRTUAL_BATCH}" == "-1" ]]; then
  BATCH=32
else
  BATCH=1
fi

EXPORT=""
if [ "$MINDIR_FILE" != "" ]; then
  cp -f $MINDIR_FILE model/lenet_tod.mindir
  EXPORT="DONT_EXPORT"
fi

cd model/ || exit 1
rm -f *.ms
EXPORT=$EXPORT ./prepare_model.sh $BATCH $DOCKER || exit 1
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
mv mindspore-*/runtime/third_party/glog/* ${PACKAGE}/lib/
cd mindspore-*
if [[ "${TARGET}" == "arm64" ]] && [[ -d "runtime/third_party/hiai_ddk/lib" ]]; then
  mv runtime/third_party/hiai_ddk/lib/* ../${PACKAGE}/lib/
fi

cd ..
rm -rf msl
mv mindspore-* msl/
rm -rf msl/tools/
rm ${PACKAGE}/lib/*.a

# Copy the dataset to the package
cp -r $MNIST_DATA_PATH ${PACKAGE}/dataset || exit 1
cp scripts/*.dat ${PACKAGE}/dataset

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

    echo "========Training on Device====="
    adb shell "cd /data/local/tmp/package-arm64 && /system/bin/sh train.sh ${EPOCHS} ${ENABLEFP16} -b ${VIRTUAL_BATCH}"
    echo

    echo "===Evaluating trained Model====="
    adb shell "cd /data/local/tmp/package-arm64 && /system/bin/sh eval.sh ${ENABLEFP16}"
    echo

    echo "====Running Inference Model====="
    adb shell "cd /data/local/tmp/package-arm64 && /system/bin/sh infer.sh"
    echo
  else
    echo "=======Pushing to device======="
    adb -s ${DEVICE_ID} push ${PACKAGE} /data/local/tmp/

    echo "========Training on Device====="
    adb -s ${DEVICE_ID} shell "cd /data/local/tmp/package-arm64 && /system/bin/sh train.sh ${EPOCHS} ${ENABLEFP16} -b ${VIRTUAL_BATCH}"
    echo

    echo "===Evaluating trained Model====="
    adb -s ${DEVICE_ID} shell "cd /data/local/tmp/package-arm64 && /system/bin/sh eval.sh ${ENABLEFP16}"
    echo

    echo "====Running Inference Model====="
    adb -s ${DEVICE_ID} shell "cd /data/local/tmp/package-arm64 && /system/bin/sh infer.sh"
    echo
  fi
else
  cd ${PACKAGE} || exit 1
  echo "======Training Locally========="
  ./train.sh ${EPOCHS}
  if [ "$?" != "0" ]; then
    echo "Training Model failed"
    exit 1
  fi

  echo "===Evaluating trained Model====="
  ./eval.sh
  if [ "$?" != "0" ]; then
    echo "Eval Model failed"
    exit 1
  fi
  
  echo "====Running Inference Model====="
  ./infer.sh
  if [ "$?" != "0" ]; then
    echo "Running Inference Model failed"
    exit 1
  fi
  
  cd ..
fi

