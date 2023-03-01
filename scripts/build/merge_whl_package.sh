#!/bin/bash
# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

set -e
BASEPATH=$(cd "$(dirname $0)"; pwd)
MINDSPORE_ROOT_PATH=`realpath $BASEPATH/../../`
BASE_PACKAGE_UNZIP_DIR=./0
PACKAGE_FILE_NAME=`basename $1`

counter=0
for whl in "$@"; do
  echo "Unzip $whl ..."
  unzip -q $whl -d $counter
  ((++counter))
done

MAX_GPU_VERSION=0
declare -A GPU_VERSION_MAP
for ((i=1;i<$counter;i=$i+1))
do
  echo "Rename $i dirname to mindspore ..."
  mv ./$i/mindspore.py* "./$i/mindspore"
  echo "Copy $i plugin files to 0 ..."
  if [ -d "./$i/mindspore/lib/plugin" ]; then
    \cp -rf ./$i/mindspore/lib/plugin/* $BASE_PACKAGE_UNZIP_DIR/mindspore/lib/plugin
  fi;
  if [ -f "./$i/mindspore/lib/libmpi_collective.so" ]; then
    \cp -rf ./$i/mindspore/lib/libmpi_collective.so $BASE_PACKAGE_UNZIP_DIR/mindspore/lib/
  fi;
  if [ -f "./$i/mindspore/lib/libmpi_adapter.so" ]; then
    \cp -rf ./$i/mindspore/lib/libmpi_adapter.so $BASE_PACKAGE_UNZIP_DIR/mindspore/lib/
  fi;
  if [ -f "./$i/mindspore/lib/libmindspore.so" ]; then
    \cp -rf ./$i/mindspore/lib/libmindspore.so $BASE_PACKAGE_UNZIP_DIR/mindspore/lib/
  fi;
  if [ -f "./$i/mindspore/lib/libmindspore_shared_lib.so" ]; then
    \cp -rf ./$i/mindspore/lib/libmindspore_shared_lib.so $BASE_PACKAGE_UNZIP_DIR/mindspore/lib/
  fi;
  CUR_GPU_VERSION=`find "./$i/mindspore/lib/plugin" -name 'gpu*' -exec sh -c 'echo ${0##*gpu}' {} \;`
  if [ -n "$CUR_GPU_VERSION" ]; then
    GPU_VERSION_MAP[$CUR_GPU_VERSION]=$i
  else
    rm -rf $i
  fi;
done

for key in $(for x in "${!GPU_VERSION_MAP[@]}"; do echo $x; done | sort)
do
  i=${GPU_VERSION_MAP[$key]}
  CUR_GPU_VERSION=$key
  CUDA_OPS_FILE=`basename ./$i/mindspore/lib/plugin/gpu$CUR_GPU_VERSION/libcuda_ops.so*`
  if [ "`echo "$CUR_GPU_VERSION > $MAX_GPU_VERSION" | bc`" -eq 1 ]; then
    if [ "`echo "$CUR_GPU_VERSION > $MAX_GPU_VERSION" | bc`" -eq 1 ]; then
      MAX_GPU_VERSION=$CUR_GPU_VERSION
    fi;
    if [ ! -d "$BASE_PACKAGE_UNZIP_DIR/mindspore/lib/plugin/gpu" ]; then
      mkdir -p $BASE_PACKAGE_UNZIP_DIR/mindspore/lib/plugin/gpu
    fi;
    \cp -rf ./$i/mindspore/lib/plugin/gpu$CUR_GPU_VERSION/$CUDA_OPS_FILE \
      $BASE_PACKAGE_UNZIP_DIR/mindspore/lib/plugin/gpu/$CUDA_OPS_FILE
  fi;
  rm -f $BASE_PACKAGE_UNZIP_DIR/mindspore/lib/plugin/gpu$CUR_GPU_VERSION/libcuda_ops.so*
  rm -rf $i
done


export COMMIT_ID=`cat $BASE_PACKAGE_UNZIP_DIR/mindspore/.commit_id | awk '{print $3}' | sed $'s/\'//g'`
VERSION=`cat $BASE_PACKAGE_UNZIP_DIR/mindspore/version.py | awk '{print $3}' | sed $'s/\'//g'`
echo -n "$VERSION" > $MINDSPORE_ROOT_PATH/version.txt

echo "Delete useless file ..."
rm -f $BASE_PACKAGE_UNZIP_DIR/mindspore/version.py
rm -f $BASE_PACKAGE_UNZIP_DIR/mindspore/default_config.py
rm -f $BASE_PACKAGE_UNZIP_DIR/mindspore/.commit_id
rm -f $BASE_PACKAGE_UNZIP_DIR/mindspore/lib/libakg.so

echo "Repacking new wheel package ..."
PACKAGE_WORK_DIR=$MINDSPORE_ROOT_PATH/build
if [ -d "$PACKAGE_WORK_DIR/package" ]; then
  rm -rf $PACKAGE_WORK_DIR/package
fi
mkdir -p $MINDSPORE_ROOT_PATH/mindspore/python/mindspore
mkdir -p $PACKAGE_WORK_DIR
PACKAGE_WORK_DIR=`realpath $PACKAGE_WORK_DIR`
mv $BASE_PACKAGE_UNZIP_DIR $PACKAGE_WORK_DIR/package
export MS_PACKAGE_NAME="mindspore"
export BACKEND_POLICY="ms"
export BUILD_PATH=$PACKAGE_WORK_DIR
cd $BUILD_PATH/package
python $MINDSPORE_ROOT_PATH/setup.py bdist_wheel
if [ -d "$MINDSPORE_ROOT_PATH/output" ]; then
  rm -rf $MINDSPORE_ROOT_PATH/output
fi
mkdir -p $MINDSPORE_ROOT_PATH/output
mv dist/*.whl $MINDSPORE_ROOT_PATH/output/$PACKAGE_FILE_NAME
cd -
cd $MINDSPORE_ROOT_PATH/output/
echo "$(sha256sum $PACKAGE_FILE_NAME)" > $PACKAGE_FILE_NAME.sha256
cd -
