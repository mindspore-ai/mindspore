#!/bin/bash

CURRENT_PATH=$(pwd)
MINDSPORE_HOME="${CURRENT_PATH}/../../../.."
echo "MINDSPORE_HOME path is ${MINDSPORE_HOME}"
CROPPER_OUTPUT_DIR=${MINDSPORE_HOME}/mindspore/lite/build/tools/cropper
mkdir -p ${CROPPER_OUTPUT_DIR}
MAPPING_OUTPUT_FILE_NAME_TMP=${CROPPER_OUTPUT_DIR}/cropper_mapping_tmp.cfg
CPU_MAPPING_OUTPUT_FILE=${CROPPER_OUTPUT_DIR}/cropper_mapping_cpu.cfg
GPU_MAPPING_OUTPUT_FILE=${CROPPER_OUTPUT_DIR}/cropper_mapping_gpu.cfg
NPU_MAPPING_OUTPUT_FILE=${CROPPER_OUTPUT_DIR}/cropper_mapping_npu.cfg
[ -n "${MAPPING_OUTPUT_FILE_NAME_TMP}" ] && rm -f ${MAPPING_OUTPUT_FILE_NAME_TMP}
[ -n "${CPU_MAPPING_OUTPUT_FILE}" ] && rm -f ${CPU_MAPPING_OUTPUT_FILE}
[ -n "${GPU_MAPPING_OUTPUT_FILE}" ] && rm -f ${GPU_MAPPING_OUTPUT_FILE}
[ -n "${NPU_MAPPING_OUTPUT_FILE}" ] && rm -f ${NPU_MAPPING_OUTPUT_FILE}
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
-I${MINDSPORE_HOME}/mindspore/ccsrc/backend/kernel_compiler/cpu"

REMOVE_LISTS_STR=""
getDeep() {
  map_files=$(gcc -MM ${2} ${DEFINE_STR} ${HEADER_LOCATION})
  # first is *.o second is *.cc
  array_deep=()
  while IFS='' read -r line; do array_deep+=("$line"); done < <(echo ${map_files} | awk -F '\' '{for(i=3;i<=NF;i++){print $i}}' | egrep -v 'flatbuffers|build' | egrep -v ${REMOVE_LISTS_STR})
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
      while IFS='' read -r line; do array_file+=("$line"); done < <(echo ${map_files} | awk -F '\' '{for(i=3;i<=NF;i++){print $i}}' | egrep -v 'flatbuffers|build' | egrep -v ${REMOVE_LISTS_STR})
      # shellcheck disable=SC2068
      for array_file in ${array_file[@]}; do
        # only add existing files
        if [[ -e ${array_file%h*}cc ]]; then
          getDeep ${type} ${array_file%h*}cc ${3} &
          getDeep ${type} ${array_file} ${3} &
          array_file_split=$(echo ${array_file} | awk -F '/' '{print $NF}')
          echo "${type},${3},${array_file_split%h*}cc.o" >>${MAPPING_OUTPUT_FILE_NAME_TMP}
        fi
        if [[ -e ${array_file%h*}c ]]; then
          getDeep ${type} ${array_file%h*}c ${3} &
          getDeep ${type} ${array_file} ${3} &
          array_file_split=$(echo ${array_file} | awk -F '/' '{print $NF}')
          echo "${type},${3},${array_file_split%h*}c.o" >>${MAPPING_OUTPUT_FILE_NAME_TMP}
        fi
      done
    done
  done
}

getCommonFile() {
  echo "start get common files"
  cd "${MINDSPORE_HOME}" || exit 1
  include_h=()
  while IFS='' read -r line; do include_h+=("$line"); done < <(ls mindspore/lite/include/*.h)
  src_files_h=()
  while IFS='' read -r line; do src_files_h+=("$line"); done < <(ls mindspore/lite/src/*.h)
  common_files_h=()
  while IFS='' read -r line; do common_files_h+=("$line"); done < <(ls mindspore/lite/src/common/*.h)
  runtime_files_h=()
  while IFS='' read -r line; do runtime_files_h+=("$line"); done < <(ls mindspore/lite/src/runtime/*.h)
  others_files_h=(
    mindspore/lite/src/populate/populate_register.h
    mindspore/lite/src/runtime/infer_manager.h
    mindspore/ccsrc/backend/kernel_compiler/cpu/nnacl/infer/infer_register.h
    mindspore/ccsrc/backend/kernel_compiler/cpu/nnacl/nnacl_utils.h
    mindspore/ccsrc/backend/kernel_compiler/cpu/nnacl/pack.h
    mindspore/lite/src/runtime/kernel/arm/fp16/common_fp16.h
    mindspore/lite/src/ops/populate/populate_register.h
    mindspore/ccsrc/backend/kernel_compiler/cpu/nnacl/op_base.h
    mindspore/core/ir/dtype/type_id.h
    mindspore/core/utils/overload.h
    mindspore/lite/tools/common/option.h
    mindspore/ccsrc/backend/kernel_compiler/cpu/nnacl/intrinsics/ms_simd_instructions.h
    mindspore/ccsrc/backend/kernel_compiler/cpu/nnacl/intrinsics/ms_simd_instructions_fp16.h
    mindspore/ccsrc/backend/kernel_compiler/cpu/nnacl/infer/infer.h
    mindspore/ccsrc/backend/kernel_compiler/cpu/nnacl/tensor_c.h
    mindspore/ccsrc/backend/kernel_compiler/cpu/nnacl/infer/common_infer.h
    mindspore/ccsrc/backend/kernel_compiler/cpu/nnacl/errorcode.h
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
  while IFS='' read -r line; do assembly_files+=("$line"); done < <(ls ${MINDSPORE_HOME}/mindspore/ccsrc/backend/kernel_compiler/cpu/nnacl/assembly/*/*.S)
  others_files_c=(
    "${MINDSPORE_HOME}"/mindspore/ccsrc/backend/kernel_compiler/cpu/nnacl/nnacl_utils.c
    "${MINDSPORE_HOME}"/mindspore/lite/src/runtime/kernel/arm/fp16/common_fp16.cc
    "${MINDSPORE_HOME}"/mindspore/lite/src/runtime/infer_manager.cc
    "${MINDSPORE_HOME}"/mindspore/ccsrc/backend/kernel_compiler/cpu/nnacl/infer/infer_register.c
    "${MINDSPORE_HOME}"/mindspore/core/utils/status.cc
    "${MINDSPORE_HOME}"/mindspore/ccsrc/backend/kernel_compiler/cpu/nnacl/infer/common_infer.c
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
    echo "CommonFile,common,${file}.o" >>${MAPPING_OUTPUT_FILE_NAME_TMP}
  done
}

# The x86 platform cannot search based on header files, so manually search for the first layer.
# opencl & ddk
getOpsFileWithNoDeepSearch() {
  echo "start get gpu/npu operator mapping file $3"
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

      local ret=$(egrep -r *.h\" ${file} | awk -F '\"' '{print $2}')
      local ret_h=$(egrep -r *.h\" ${file%cc*}h | awk -F '\"' '{print $2}')
      local depend_file=("${ret}" "${ret_h}")
      for array_file in ${depend_file[@]}; do
        # only add existing files
        if [[ -e ${MINDSPORE_HOME}/mindspore/lite/${array_file%h*}cc ]]; then
          array_file_split=$(echo ${array_file} | awk -F '/' '{print $NF}')
          echo "${type},${3},${array_file_split%h*}cc.o" >>${MAPPING_OUTPUT_FILE_NAME_TMP}
        fi
        if [[ -e ${MINDSPORE_HOME}/mindspore/lite/${array_file%h*}c ]]; then
          array_file_split=$(echo ${array_file} | awk -F '/' '{print $NF}')
          echo "${type},${3},${array_file_split%h*}c.o" >>${MAPPING_OUTPUT_FILE_NAME_TMP}
        fi
      done
    done
  done
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
getOpsFile "REG_POPULATE\(PrimitiveType_" "${MINDSPORE_HOME}/mindspore/lite/src/ops/populate" "prototype" &
getOpsFile "REG_INFER\(.*?, PrimType_" "${MINDSPORE_HOME}/mindspore/ccsrc/backend/kernel_compiler/cpu/nnacl/infer" "prototype" &
# support for cpu
getOpsFile "REG_KERNEL\(.*?, kNumberTypeFloat32, PrimitiveType_" "${MINDSPORE_HOME}/mindspore/lite/src/runtime/kernel/arm" "kNumberTypeFloat32" &
getOpsFile "REG_KERNEL\(.*?, kNumberTypeFloat16, PrimitiveType_" "${MINDSPORE_HOME}/mindspore/lite/src/runtime/kernel/arm" "kNumberTypeFloat16" &
getOpsFile "REG_KERNEL\(.*?, kNumberTypeInt8, PrimitiveType_" "${MINDSPORE_HOME}/mindspore/lite/src/runtime/kernel/arm" "kNumberTypeInt8" &
getOpsFile "REG_KERNEL\(.*?, kNumberTypeInt32, PrimitiveType_" "${MINDSPORE_HOME}/mindspore/lite/src/runtime/kernel/arm" "kNumberTypeInt32" &
getOpsFile "REG_KERNEL\(.*?, kNumberTypeBool, PrimitiveType_" "${MINDSPORE_HOME}/mindspore/lite/src/runtime/kernel/arm" "kNumberTypeInt32" &
wait
sleep 1
# remove duplicate files
sort ${MAPPING_OUTPUT_FILE_NAME_TMP} | egrep -v "*.h.o" | uniq >${CPU_MAPPING_OUTPUT_FILE}
chmod 444 ${CPU_MAPPING_OUTPUT_FILE}

# support for gpu
opencl_files=()
while IFS='' read -r line; do opencl_files+=("$line"); done < <(ls ${MINDSPORE_HOME}/mindspore/lite/src/runtime/kernel/opencl/*.cc)
while IFS='' read -r line; do opencl_files+=("$line"); done < <(ls ${MINDSPORE_HOME}/mindspore/lite/src/runtime/gpu/*.cc)
while IFS='' read -r line; do opencl_files+=("$line"); done < <(ls ${MINDSPORE_HOME}/mindspore/lite/src/runtime/gpu/opencl/*.cc)
opencl_others_files=(
  "${MINDSPORE_HOME}/mindspore/lite/src/runtime/kernel/opencl/kernel/fusion_eltwise.cc"
  "${MINDSPORE_HOME}/mindspore/lite/src/runtime/kernel/opencl/kernel/to_format.cc"
)
opencl_files=("${opencl_files[@]}" "${opencl_others_files[@]}")
# shellcheck disable=SC2068
for file in ${opencl_files[@]}; do
  file=$(echo ${file} | awk -F '/' '{print $NF}')
  echo "CommonFile,common,${file}.o" >>${MAPPING_OUTPUT_FILE_NAME_TMP}
done

getOpsFileWithNoDeepSearch "REG_KERNEL\(.*?, kNumberTypeFloat32, PrimitiveType_" "${MINDSPORE_HOME}/mindspore/lite/src/runtime/kernel/opencl/kernel" "kNumberTypeFloat32" &
getOpsFileWithNoDeepSearch "REG_KERNEL\(.*?, kNumberTypeFloat16, PrimitiveType_" "${MINDSPORE_HOME}/mindspore/lite/src/runtime/kernel/opencl/kernel" "kNumberTypeFloat16" &
getOpsFileWithNoDeepSearch "REG_KERNEL\(.*?, kNumberTypeInt8, PrimitiveType_" "${MINDSPORE_HOME}/mindspore/lite/src/runtime/kernel/opencl/kernel" "kNumberTypeInt8" &
getOpsFileWithNoDeepSearch "REG_KERNEL\(.*?, kNumberTypeInt32, PrimitiveType_" "${MINDSPORE_HOME}/mindspore/lite/src/runtime/kernel/opencl/kernel" "kNumberTypeInt32" &
getOpsFileWithNoDeepSearch "REG_KERNEL\(.*?, kNumberTypeBool, PrimitiveType_" "${MINDSPORE_HOME}/mindspore/lite/src/runtime/kernel/opencl/kernel" "kNumberTypeInt32" &
sleep 1
wait
sort ${MAPPING_OUTPUT_FILE_NAME_TMP} | egrep -v "*.h.o" | uniq >${GPU_MAPPING_OUTPUT_FILE}
chmod 444 ${GPU_MAPPING_OUTPUT_FILE}

# support for npu
npu_files=()
while IFS='' read -r line; do npu_files+=("$line"); done < <(ls ${MINDSPORE_HOME}/mindspore/lite/src/runtime/agent/npu/*.cc)
while IFS='' read -r line; do npu_files+=("$line"); done < <(ls ${MINDSPORE_HOME}/mindspore/lite/src/runtime/agent/npu/optimizer/*.cc)
npu_other_files=(
  "${MINDSPORE_HOME}/mindspore/lite/src/runtime/kernel/arm/fp32/transpose_fp32.cc"
)
# shellcheck disable=SC2068
for file in ${npu_other_files[@]}; do
  getDeep CommonFile ${file} common
done
opencl_files=("${opencl_files[@]}" "${npu_other_files[@]}")
# shellcheck disable=SC2068
for file in ${npu_files[@]}; do
  file=$(echo ${file} | awk -F '/' '{print $NF}')
  echo "CommonFile,common,${file}.o" >>${MAPPING_OUTPUT_FILE_NAME_TMP}
done

getOpsFileWithNoDeepSearch "REG_KERNEL\(.*?, kNumberTypeFloat32, PrimitiveType_" "${MINDSPORE_HOME}/mindspore/lite/src/runtime/kernel/npu" "kNumberTypeFloat32" &
getOpsFileWithNoDeepSearch "REG_KERNEL\(.*?, kNumberTypeFloat16, PrimitiveType_" "${MINDSPORE_HOME}/mindspore/lite/src/runtime/kernel/npu" "kNumberTypeFloat16" &
getOpsFileWithNoDeepSearch "REG_KERNEL\(.*?, kNumberTypeInt8, PrimitiveType_" "${MINDSPORE_HOME}/mindspore/lite/src/runtime/kernel/npu" "kNumberTypeInt8" &
getOpsFileWithNoDeepSearch "REG_KERNEL\(.*?, kNumberTypeInt32, PrimitiveType_" "${MINDSPORE_HOME}/mindspore/lite/src/runtime/kernel/npu" "kNumberTypeInt32" &
getOpsFileWithNoDeepSearch "REG_KERNEL\(.*?, kNumberTypeBool, PrimitiveType_" "${MINDSPORE_HOME}/mindspore/lite/src/runtime/kernel/npu" "kNumberTypeInt32" &
wait
sleep 1
sort ${MAPPING_OUTPUT_FILE_NAME_TMP} | egrep -v "*.h.o" | uniq >${NPU_MAPPING_OUTPUT_FILE}
chmod 444 ${NPU_MAPPING_OUTPUT_FILE}

# modify file permissions to read-only
[ -n "${MAPPING_OUTPUT_FILE_NAME_TMP}" ] && rm -f ${MAPPING_OUTPUT_FILE_NAME_TMP}
echo "Complete all tasks."
