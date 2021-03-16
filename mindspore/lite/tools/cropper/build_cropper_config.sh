#!/bin/bash

CURRENT_PATH=$(pwd)
MINDSPORE_HOME="${CURRENT_PATH}/../../../.."
echo "MINDSPORE_HOME path is ${MINDSPORE_HOME}"
CROPPER_OUTPUT_DIR=${MINDSPORE_HOME}/mindspore/lite/build/tools/cropper
mkdir -p ${CROPPER_OUTPUT_DIR}
MAPPING_OUTPUT_FILE_NAME_TMP=${CROPPER_OUTPUT_DIR}/cropper_mapping_cpu_tmp.cfg
MAPPING_OUTPUT_FILE_NAME=${CROPPER_OUTPUT_DIR}/cropper_mapping_cpu.cfg
echo "MAPPING_OUTPUT_FILE_NAME is ${MAPPING_OUTPUT_FILE_NAME}"
[ -n "${MAPPING_OUTPUT_FILE_NAME_TMP}" ] && rm -f ${MAPPING_OUTPUT_FILE_NAME_TMP}
[ -n "${MAPPING_OUTPUT_FILE_NAME}" ] && rm -f ${MAPPING_OUTPUT_FILE_NAME}
ops_list=()
DEFINE_STR="-DENABLE_ANDROID -DENABLE_ARM -DENABLE_ARM64 -DENABLE_NEON -DNO_DLIB -DUSE_ANDROID_LOG -DANDROID"
# get the flatbuffers path
if [ ${MSLIBS_CACHE_PATH} ]; then
  FLATBUFFERS_LIST=()
  while IFS='' read -r line; do FLATBUFFERS_LIST+=("$line"); done < <(ls -d ${MSLIBS_CACHE_PATH}/flatbuffers_*/include)
  FLATBUFFERS=${FLATBUFFERS_LIST[0]}
  echo "FLATBUFFERS path is ${FLATBUFFERS}"
else
  FLATBUFFERS=$(ls -d ${MINDSPORE_HOME}/mindspore/lite/build/.mslib/flatbuffers_*/include)
  echo "FLATBUFFERS path is ${FLATBUFFERS}"
fi

HEADER_LOCATION="-I${MINDSPORE_HOME}
-I${MINDSPORE_HOME}/mindspore/core
-I${MINDSPORE_HOME}/mindspore/core/ir
-I${MINDSPORE_HOME}/mindspore/core/mindrt/include
-I${MINDSPORE_HOME}/mindspore/core/mindrt/src
-I${MINDSPORE_HOME}/mindspore/core/mindrt/
-I${MINDSPORE_HOME}/mindspore/ccsrc
-I${MINDSPORE_HOME}/mindspore/lite
-I${MINDSPORE_HOME}/mindspore/lite/src
-I${MINDSPORE_HOME}/mindspore/lite/src/runtime/kernel/arm
-I${MINDSPORE_HOME}/third_party
-I${MINDSPORE_HOME}/mindspore/lite/build
-I${MINDSPORE_HOME}/cmake/../third_party/securec/include
-I${FLATBUFFERS}
-I${MINDSPORE_HOME}/mindspore/lite/build/schema
-I${MINDSPORE_HOME}/mindspore/lite/build/schema/inner
-I${MINDSPORE_HOME}/mindspore/lite/src/../nnacl
-I${MINDSPORE_HOME}/mindspore/lite/src/../nnacl/optimize"

REMOVE_LISTS_STR=""
getDeep() {
  map_files=$(gcc -MM ${2} ${DEFINE_STR} ${HEADER_LOCATION})
  # first is *.o second is *.cc
  array_deep=()
  while IFS='' read -r line; do array_deep+=("$line"); done < <(echo ${map_files} | awk -F '\' '{for(i=3;i<=NF;i++){print $i}}' | grep -E 'src/runtime|nnacl' | egrep -v ${REMOVE_LISTS_STR})
  # shellcheck disable=SC2068
  for array_deep_file in ${array_deep[@]}; do
    # only add existing files
    if [[ -e ${array_deep_file%h*}cc ]]; then
      file_split=$(echo ${array_deep_file} | awk -F '/' '{print $NF}')
      echo "${1},${3},${file_split%h*}cc.o" >>${MAPPING_OUTPUT_FILE_NAME_TMP}
    fi
    if [[ -e ${array_deep_file%h*}c ]]; then
      file_split=$(echo ${array_deep_file} | awk -F '/' '{print $NF}')
      echo "${1},${3},${file_split%h*}c.o" >>${MAPPING_OUTPUT_FILE_NAME_TMP}
    fi
  done
}
getOpsFile() {
  echo "start get operator mapping file $3"
  # shellcheck disable=SC2068
  for type in ${ops_list[@]}; do
    # get mapping
    ret=$(egrep -r -l "$1${type}," $2)
    array=("${ret}")
    # shellcheck disable=SC2068
    for file in ${array[@]}; do
      # delete \n
      out_file=$(echo ${file} | awk -F '/' '{print $NF}')
      # concat schemaType + fileType + fileName append to files
      echo "${type},${3},${out_file}.o" >>${MAPPING_OUTPUT_FILE_NAME_TMP}
      map_files=$(gcc -MM ${file} ${DEFINE_STR} ${HEADER_LOCATION})
      # first is *.o second is *.cc
      array_file=()
      while IFS='' read -r line; do array_file+=("$line"); done < <(echo ${map_files} | awk -F '\' '{for(i=3;i<=NF;i++){print $i}}' | grep -E 'src/runtime|nnacl' | egrep -v ${REMOVE_LISTS_STR})
      # shellcheck disable=SC2068
      for array_file in ${array_file[@]}; do
        # only add existing files
        if [[ -e ${array_file%h*}cc ]]; then
          getDeep ${type} ${array_file%h*}cc ${3} &
          array_file_split=$(echo ${array_file} | awk -F '/' '{print $NF}')
          echo "${type},${3},${array_file_split%h*}cc.o" >>${MAPPING_OUTPUT_FILE_NAME_TMP}
        fi
        if [[ -e ${array_file%h*}c ]]; then
          getDeep ${type} ${array_file%h*}c ${3} &
          array_file_split=$(echo ${array_file} | awk -F '/' '{print $NF}')
          echo "${type},${3},${array_file_split%h*}c.o" >>${MAPPING_OUTPUT_FILE_NAME_TMP}
        fi
      done
    done
  done
}
getCommonFile() {
  echo "start get common files"
  include_h=()
  while IFS='' read -r line; do include_h+=("$line"); done < <(ls ${MINDSPORE_HOME}/mindspore/lite/include/*.h)
  src_files_h=()
  while IFS='' read -r line; do src_files_h+=("$line"); done < <(ls ${MINDSPORE_HOME}/mindspore/lite/src/*.h)
  common_files_h=()
  while IFS='' read -r line; do common_files_h+=("$line"); done < <(ls ${MINDSPORE_HOME}/mindspore/lite/src/common/*.h)
  runtime_files_h=()
  while IFS='' read -r line; do runtime_files_h+=("$line"); done < <(ls ${MINDSPORE_HOME}/mindspore/lite/src/runtime/*.h)
  others_files_h=(
    "${MINDSPORE_HOME}"/mindspore/lite/src/populate/populate_register.h
    "${MINDSPORE_HOME}"/mindspore/lite/src/runtime/infer_manager.h
    "${MINDSPORE_HOME}"/mindspore/lite/nnacl/infer/infer_register.h
    "${MINDSPORE_HOME}"/mindspore/lite/nnacl/nnacl_utils.h
    "${MINDSPORE_HOME}"/mindspore/lite/nnacl/pack.h
    "${MINDSPORE_HOME}"/mindspore/lite/src/runtime/kernel/arm/fp16/common_fp16.h
  )
  all_files_h=("${include_h[@]}" "${src_files_h[@]}" "${common_files_h[@]}" "${runtime_files_h[@]}" "${others_files_h[@]}")

  # concat regx
  REMOVE_LISTS_STR="${all_files_h[0]}"
  # shellcheck disable=SC2068
  for val in ${all_files_h[@]:1}; do
    REMOVE_LISTS_STR="$REMOVE_LISTS_STR|$val"
  done

  cxx_api_files=()
  while IFS='' read -r line; do cxx_api_files+=("$line"); done < <(ls ${MINDSPORE_HOME}/mindspore/lite/src/cxx_api/graph/*.cc)
  while IFS='' read -r line; do cxx_api_files+=("$line"); done < <(ls ${MINDSPORE_HOME}/mindspore/lite/src/cxx_api/model/*.cc)
  while IFS='' read -r line; do cxx_api_files+=("$line"); done < <(ls ${MINDSPORE_HOME}/mindspore/lite/src/cxx_api/tensor/*.cc)
  while IFS='' read -r line; do cxx_api_files+=("$line"); done < <(ls ${MINDSPORE_HOME}/mindspore/lite/src/cxx_api/*.cc)
  mindrt_files=()
  while IFS='' read -r line; do mindrt_files+=("$line"); done < <(ls ${MINDSPORE_HOME}/mindspore/core/mindrt/src/*.cc)
  while IFS='' read -r line; do mindrt_files+=("$line"); done < <(ls ${MINDSPORE_HOME}/mindspore/core/mindrt/src/async/*.cc)
  while IFS='' read -r line; do mindrt_files+=("$line"); done < <(ls ${MINDSPORE_HOME}/mindspore/core/mindrt/src/actor/*.cc)
  src_files=()
  while IFS='' read -r line; do src_files+=("$line"); done < <(ls ${MINDSPORE_HOME}/mindspore/lite/src/*.cc)
  common_files=()
  while IFS='' read -r line; do common_files+=("$line"); done < <(ls ${MINDSPORE_HOME}/mindspore/lite/src/common/*.cc)
  runtime_files_cc=()
  while IFS='' read -r line; do runtime_files_cc+=("$line"); done < <(ls ${MINDSPORE_HOME}/mindspore/lite/src/runtime/*.cc)
  runtime_files_c=()
  while IFS='' read -r line; do runtime_files_c+=("$line"); done < <(ls ${MINDSPORE_HOME}/mindspore/lite/src/runtime/*.c)
  # sava all assembly files
  assembly_files=()
  while IFS='' read -r line; do assembly_files+=("$line"); done < <(ls ${MINDSPORE_HOME}/mindspore/lite/nnacl/assembly/*/*.S)
  others_files_c=(
    "${MINDSPORE_HOME}"/mindspore/lite/nnacl/nnacl_utils.c
    "${MINDSPORE_HOME}"/mindspore/lite/src/runtime/kernel/arm/fp16/common_fp16.cc
    "${MINDSPORE_HOME}"/mindspore/lite/src/runtime/infer_manager.cc
    "${MINDSPORE_HOME}"/mindspore/lite/nnacl/infer/infer_register.c
    "${MINDSPORE_HOME}"/mindspore/core/utils/status.cc
  )
  all_files=("${src_files[@]}" "${common_files[@]}" "${runtime_files_cc[@]}"
    "${runtime_files_c[@]}" "${others_files_c[@]}" "${assembly_files[@]}" "${mindrt_files[@]}"
    "${cxx_api_files[@]}"
  )
  # shellcheck disable=SC2068
  for file in ${all_files[@]}; do
    map_files=$(gcc -MM ${file} ${DEFINE_STR} ${HEADER_LOCATION})
    # first is *.o second is *.cc
    # shellcheck disable=SC2207
    array_runtime=($(echo ${map_files} | awk -F '\' '{for(i=3;i<=NF;i++){print $i}}' | grep -v "flatbuffers" | egrep -v ${REMOVE_LISTS_STR}))
    # only add existing files
    for array_runtime_file in "${array_runtime[@]}"; do
      if [[ -e ${array_runtime_file%h*}cc && ! ${all_files[*]} =~ ${array_runtime_file%h*}cc ]]; then
        all_files=("${all_files[@]}" "${array_runtime_file%h*}cc")
      fi
      if [[ -e ${array_runtime_file%h*}c && ! ${all_files[*]} =~ ${array_runtime_file%h*}c ]]; then
        all_files=("${all_files[@]}" "${array_runtime_file%h*}c")
      fi
    done
  done
  # shellcheck disable=SC2068
  for file in ${all_files[@]}; do
    file=$(echo ${file} | awk -F '/' '{print $NF}')
    echo "CommonFile,common,${file}.o" >>${MAPPING_OUTPUT_FILE_NAME_TMP} &
  done
  wait
}

# automatically generate operator list
generateOpsList() {
  echo "start generate operator list"
  ops_list=()
  while IFS='' read -r line; do ops_list+=("$line"); done < <(grep -Rn "^table" "${MINDSPORE_HOME}/mindspore/lite/schema/ops.fbs" | awk -F ' ' '{print $2}')
  ops_num=$((${#ops_list[@]}))
  echo "ops nums:${ops_num}"
}
echo "Start getting all file associations."
generateOpsList
getCommonFile
# get src/ops
getOpsFile "Registry\(schema::PrimitiveType_" "${MINDSPORE_HOME}/mindspore/lite/src/ops" "prototype" &
getOpsFile "REG_INFER\(.*?, PrimType_" "${MINDSPORE_HOME}/mindspore/lite/nnacl/infer" "prototype" &
getOpsFile "REG_KERNEL\(.*?, kNumberTypeFloat32, PrimitiveType_" "${MINDSPORE_HOME}/mindspore/lite/src/runtime/kernel/arm" "kNumberTypeFloat32" &
getOpsFile "REG_KERNEL\(.*?, kNumberTypeFloat16, PrimitiveType_" "${MINDSPORE_HOME}/mindspore/lite/src/runtime/kernel/arm" "kNumberTypeFloat16" &
getOpsFile "REG_KERNEL\(.*?, kNumberTypeInt8, PrimitiveType_" "${MINDSPORE_HOME}/mindspore/lite/src/runtime/kernel/arm" "kNumberTypeInt8" &
getOpsFile "REG_KERNEL\(.*?, kNumberTypeInt32, PrimitiveType_" "${MINDSPORE_HOME}/mindspore/lite/src/runtime/kernel/arm" "kNumberTypeInt32" &
wait
echo "remove duplicate files"
# remove duplicate files
sort ${MAPPING_OUTPUT_FILE_NAME_TMP} | uniq >${MAPPING_OUTPUT_FILE_NAME}
# modify file permissions to read-only
[ -n "${MAPPING_OUTPUT_FILE_NAME_TMP}" ] && rm -f ${MAPPING_OUTPUT_FILE_NAME_TMP}
chmod 444 ${MAPPING_OUTPUT_FILE_NAME}
echo "Complete all tasks."
