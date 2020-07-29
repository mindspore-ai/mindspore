#!/usr/bin/env bash

set -e

CUR_DIR=$(cd "$(dirname $0)"; pwd)
BASE_DIR=${CUR_DIR}/../../

usage()
{
  echo "Usage:"
  echo "bash build.sh [-d] [-a arm64|arm32] [-j[n]] [-m] [-f] [-g] [-c] [-s] [-o]"
  echo ""
  echo "Options:"
  echo "    -d Enable Debug"
  echo "    -c Enable compile converter, default off"
  echo "    -m Enable Incremental compilation"
  echo "    -a Select ARM platform, default off"
  echo "    -j[n] Set the threads when building, default: -j8"
  echo "    -f Compile fp16 ops"
  echo "    -g Enable gpu compile"
  echo "    -s Support train"
  echo "    -o Offline compile OpenCL kernel"
}

checkopts()
{
  # Init default values of build options
  THREAD_NUM="8"
  BUILD_TYPE="Release"
  BUILD_DEVICE_PLATFORM="off"
  MAKE_ONLY="off"
  ENABLE_FP16="off"
  ENABLE_GPU="off"
  ENABLE_CONVERTER="off"
  SUPPORT_TRAIN="off"
  OFFLINE_COMPILE="off"

  # Process the options
  while getopts 'j:da:mfcsgo' opt
  do
    OPTARG=$(echo ${OPTARG} | tr '[A-Z]' '[a-z]')
    case "${opt}" in
      m)
        MAKE_ONLY="on"
        echo "Incremental compilation"
        ;;
      d)
        BUILD_TYPE="Debug"
        echo "Build Debug version"
        ;;
      j)
        THREAD_NUM=$OPTARG
        ;;
      a)
        if [[ "X$OPTARG" == "Xarm64" ]]; then
          BUILD_DEVICE_PLATFORM="arm64"
          echo "Enable arm64"
        elif [[ "X$OPTARG" == "Xarm32" ]]; then
          BUILD_DEVICE_PLATFORM="arm32"
          echo "Enable arm32"
        else
          echo "-I parameter must be arm64 or arm32"
          exit 1
        fi
        ;;
      c)
        ENABLE_CONVERTER="on"
        echo "Enable converter"
        ;;
      s)
        SUPPORT_TRAIN="on"
        echo "Support train"
        ;;
      f)
        ENABLE_FP16="on"
        echo "Enable fp16"
        ;;
      g)
        ENABLE_GPU="on"
        echo "Enable gpu"
        ;;
      o)
        OFFLINE_COMPILE="on"
        echo "OpenCL kernel offline compile"
        ;;
      *)
        echo "Unknown option ${opt}!"
        usage
        exit 1
    esac
  done
}

checkndk() {
    if [ "${ANDROID_NDK}" ]; then
        echo -e "\e[31mANDROID_NDK_PATH=$ANDROID_NDK  \e[0m"
    else
        echo -e "\e[31mplease set ANDROID_NDK_PATH in environment variable for example: export ANDROID_NDK=/root/usr/android-ndk-r16b/ \e[0m"
        exit 1
    fi
}

gene_flatbuffer() {
    FLAT_DIR="${BASE_DIR}/mindspore/lite/schema"
    cd ${FLAT_DIR} && rm -rf "${FLAT_DIR}/inner" && mkdir -p "${FLAT_DIR}/inner"
    find . -name "*.fbs" -print0 | xargs -0 "${FLATC}" -c -b
    find . -name "*.fbs" -print0 | xargs -0 "${FLATC}" -c -b --reflect-types --gen-mutable --reflect-names --gen-object-api -o "${FLAT_DIR}/inner"

    FLAT_DIR="${BASE_DIR}/mindspore/lite/tools/converter/parser/tflite"
    cd ${FLAT_DIR}
    find . -name "*.fbs" -print0 | xargs -0 "${FLATC}" -c -b --reflect-types --gen-mutable --reflect-names --gen-object-api -o "${FLAT_DIR}/"
}

build_flatbuffer() {
    cd ${BASE_DIR}
    FLATC="${BASE_DIR}"/third_party/flatbuffers/build/flatc
    if [[ ! -f "${FLATC}" ]]; then
        git submodule update --init --recursive third_party/flatbuffers
        cd ${BASE_DIR}/third_party/flatbuffers
        rm -rf build && mkdir -pv build && cd build && cmake .. && make -j$THREAD_NUM
        gene_flatbuffer
    fi
    if [[ "${MAKE_ONLY}" == "off" ]]; then
        gene_flatbuffer
    fi
}

gene_protobuf() {
    PROTO_SRC_DIR="${BASE_DIR}/mindspore/lite/tools/converter/parser/caffe"
    find ${PROTO_SRC_DIR} -name "*.proto" -print0 | xargs -0 "${PROTOC}" -I"${PROTO_SRC_DIR}" --cpp_out="${PROTO_SRC_DIR}"
    PROTO_SRC_DIR="${BASE_DIR}/mindspore/lite/tools/converter/parser/onnx"
    find ${PROTO_SRC_DIR} -name "*.proto" -print0 | xargs -0 "${PROTOC}" -I"${PROTO_SRC_DIR}" --cpp_out="${PROTO_SRC_DIR}"
}

build_protobuf() {
    cd ${BASE_DIR}
    PROTOC="${BASE_DIR}"/third_party/protobuf/build/bin/protoc
    if [[ ! -f "${PROTOC}" ]]; then
        git submodule update --init --recursive third_party/protobuf
        cd ${BASE_DIR}/third_party/protobuf
        rm -rf build && mkdir -pv build && ./autogen.sh
        ./configure --prefix=${BASE_DIR}/third_party/protobuf/build
        make clean && make -j$THREAD_NUM && make install
        gene_protobuf
    fi
    if [[ "${MAKE_ONLY}" == "off" ]]; then
        gene_protobuf
    fi
}

build_gtest() {
    cd ${BASE_DIR}
    git submodule update --init --recursive third_party/googletest
}

gene_clhpp() {
    CL_SRC_DIR="${BASE_DIR}/mindspore/lite/src/runtime/kernel/opencl/cl"
    for sub_dir in "${CL_SRC_DIR}"/*
    do
        data_type="$(basename ${sub_dir})"
        if [ ! -d ${CL_SRC_DIR}/${data_type} ]; then
          continue
        fi
        cd ${CL_SRC_DIR}/${data_type}
        rm -rf *.inc
        echo "$(cd "$(dirname $0)"; pwd)"
        for file_path in "${CL_SRC_DIR}/${data_type}"/*
        do
            file="$(basename ${file_path})"
            inc_file=`echo ${CL_SRC_DIR}/${data_type}/${file} | sed 's/$/.inc/'`
            sed 's/^/\"/;s/$/    \\n\" \\/' ${CL_SRC_DIR}/${data_type}/${file} > ${inc_file}
            kernel_name=`echo ${file} | sed s'/.\{3\}$//'`
	    sed -i "1i\static const char *${kernel_name}_source_${data_type} =\"\\n\" \\" ${inc_file}
            sed -i '$a\;' ${inc_file}
        done
    done
}

gene_ocl_program() {
    CL_SRC_DIR="${BASE_DIR}/mindspore/lite/src/runtime/kernel/opencl/cl"
    SPIRV_DIR=build/spirv
    rm -rf ${SPIRV_DIR}
    mkdir -pv ${SPIRV_DIR}
    for sub_dir in "${CL_SRC_DIR}"/*
    do
        data_type="$(basename ${sub_dir})"
        if [ ! -d ${CL_SRC_DIR}/${data_type} ]; then
          continue
        fi
        #echo $(cd "$(dirname $0)"; pwd)
        for file_path in "${CL_SRC_DIR}/${data_type}"/*
        do
          file="$(basename ${file_path})"
          if [ "${file##*.}" != "cl" ]; then
            continue
          fi
          clang -Xclang -finclude-default-header -cl-std=CL2.0 --target=spir64-unknown-unknown -emit-llvm \
                -c -O0 -o ${SPIRV_DIR}/${file%.*}.bc ${CL_SRC_DIR}/${data_type}/${file}
        done
    done

    bcs=`ls ${SPIRV_DIR}/*.bc`
    llvm-link ${bcs} -o ${SPIRV_DIR}/program.bc
    llvm-spirv -o ${SPIRV_DIR}/program.spv ${SPIRV_DIR}/program.bc

    CL_PROGRAM_PATH="${BASE_DIR}/mindspore/lite/src/runtime/kernel/opencl/cl/program.inc"
    echo "#include <vector>" > ${CL_PROGRAM_PATH}
    echo "std::vector<unsigned char> g_program_binary = {" >> ${CL_PROGRAM_PATH}
    #hexdump -v -e '16/1 "0x%02x, " "\n"' ${SPIRV_DIR}/program.spv >> ${CL_PROGRAM_PATH}
    hexdump -v -e '1/1 "0x%02x, "' ${SPIRV_DIR}/program.spv >> ${CL_PROGRAM_PATH}
    echo "};" >> ${CL_PROGRAM_PATH}
    echo "Compile SPIRV done"
}

build_opencl() {
    cd ${BASE_DIR}
    git submodule update --init third_party/OpenCL-Headers
    git submodule update --init third_party/OpenCL-CLHPP
    if [[ "${OFFLINE_COMPILE}" == "on" ]]; then
        gene_ocl_program
    else
        gene_clhpp
    fi
}

buildlite() {
    if [[ "${MAKE_ONLY}" == "off" ]]; then
        cd ${CUR_DIR}
        rm -rf build
        mkdir -pv build
        cd build
        if [[ "${BUILD_DEVICE_PLATFORM}" == "arm64" ]]; then
            checkndk
            cmake -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK}/build/cmake/android.toolchain.cmake" -DANDROID_NATIVE_API_LEVEL="19"      \
                  -DANDROID_NDK="${ANDROID_NDK}" -DANDROID_ABI="arm64-v8a" -DANDROID_TOOLCHAIN_NAME="aarch64-linux-android-clang"  \
                  -DANDROID_STL="c++_shared" -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DSUPPORT_TRAIN=${SUPPORT_TRAIN}                     \
                  -DBUILD_DEVICE=on -DPLATFORM_ARM64=on -DBUILD_CONVERTER=off -DENABLE_NEON=on -DENABLE_FP16="${ENABLE_FP16}"      \
                  -DSUPPORT_GPU=${ENABLE_GPU} -DOFFLINE_COMPILE=${OFFLINE_COMPILE} ..
        elif [[ "${BUILD_DEVICE_PLATFORM}" == "arm32" ]]; then
            checkndk
            cmake -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK}/build/cmake/android.toolchain.cmake" -DANDROID_NATIVE_API_LEVEL="19"      \
                  -DANDROID_NDK="${ANDROID_NDK}" -DANDROID_ABI="armeabi-v7a" -DANDROID_TOOLCHAIN_NAME="clang"                      \
                  -DANDROID_STL="c++_shared" -DCMAKE_BUILD_TYPE=${BUILD_TYPE}                                                      \
                  -DBUILD_DEVICE=on -DPLATFORM_ARM32=on -DENABLE_NEON=on -DSUPPORT_TRAIN=${SUPPORT_TRAIN} -DBUILD_CONVERTER=off    \
                  -DSUPPORT_GPU=${ENABLE_GPU} -DOFFLINE_COMPILE=${OFFLINE_COMPILE} ..
        else
            cmake -DBUILD_DEVICE=on -DPLATFORM_ARM64=off -DBUILD_CONVERTER=${ENABLE_CONVERTER} -DSUPPORT_TRAIN=${SUPPORT_TRAIN}   \
            -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DSUPPORT_GPU=${ENABLE_GPU} -DOFFLINE_COMPILE=${OFFLINE_COMPILE} ..
        fi
    else
        cd ${CUR_DIR}/build
    fi
    VERBOSE=2 make -j$THREAD_NUM
}

echo "---------------- mindspore lite: build start ----------------"
checkopts "$@"
build_flatbuffer
if [[ "${ENABLE_CONVERTER}" == "on" ]]; then
  build_protobuf
fi
if [[ "${ENABLE_GPU}" == "on" ]]; then
  build_opencl
fi
build_gtest
buildlite
COMPILE_RET=$?
if [[ "${COMPILE_RET}" -ne 0 ]]; then
    echo "---------------- mindspore lite: build failed ----------------"
else
    echo "---------------- mindspore lite: build success ----------------"
fi
