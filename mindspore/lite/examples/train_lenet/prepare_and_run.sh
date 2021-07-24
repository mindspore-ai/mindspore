#!/bin/bash

display_usage()
{
  echo -e "\nUsage: prepare_and_run.sh -D dataset_path [-d mindspore_docker] [-r release.tar.gz] [-t arm64|x86] [-q] [-o] [-b virtual_batch]\n"
}

checkopts()
{
  TARGET="arm64"
  DOCKER=""
  MNIST_DATA_PATH=""
  QUANTIZE=""
  ENABLEFP16=false
  VIRTUAL_BATCH=-1
  while getopts 'D:d:r:t:qob:' opt
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
      q)
        QUANTIZE="QUANTIZE"
        ;;
      o)
        ENABLEFP16=true
        ;; 
      b)
        VIRTUAL_BATCH=$OPTARG
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


cd model/ || exit 1
rm -f *.ms
QUANTIZE=${QUANTIZE} ./prepare_model.sh $BATCH $DOCKER || exit 1
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
cd mindspore-*
if [[ "${TARGET}" == "arm64" ]] && [[ -d "runtime/third_party/hiai_ddk/lib" ]]; then
  mv runtime/third_party/hiai_ddk/lib/* ../${PACKAGE}/lib/
fi

cd ../
rm -rf msl
mv mindspore-* msl/

# Copy the dataset to the package
cp -r $MNIST_DATA_PATH ${PACKAGE}/dataset || exit 1

echo "==========Compiling============"
make clean
make TARGET=${TARGET}
 
# Copy the executable to the package
mv bin ${PACKAGE}/ || exit 1

if [ "${TARGET}" == "arm64" ]; then
  cp ${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so ${PACKAGE}/lib/ || exit 1

  echo "=======Pushing to device======="
  adb push ${PACKAGE} /data/local/tmp/

  echo "========Training on Device====="
  if "$ENABLEFP16"; then
    echo "Training fp16.."
    adb shell "cd /data/local/tmp/package-arm64 && /system/bin/sh train.sh -o -b ${VIRTUAL_BATCH}"
  else
    adb shell "cd /data/local/tmp/package-arm64 && /system/bin/sh train.sh -b ${VIRTUAL_BATCH}"
  fi   

  echo
  echo "===Evaluating trained Model====="
  if "$ENABLEFP16"; then
    echo "Evaluating fp16 Model.."
    adb shell "cd /data/local/tmp/package-arm64 && /system/bin/sh eval.sh -o"
  else
    adb shell "cd /data/local/tmp/package-arm64 && /system/bin/sh eval.sh"
  fi
  echo
else
  cd ${PACKAGE} || exit 1
  echo "======Training Locally========="
  ./train.sh

  echo "===Evaluating trained Model====="
  ./eval.sh
  
  cd ..
fi

