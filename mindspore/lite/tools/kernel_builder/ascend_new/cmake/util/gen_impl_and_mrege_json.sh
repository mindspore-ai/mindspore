#!/usr/bin/bash

project_path=$1
build_path=$2
vendor_name=mslite
if [[ ! -d "$project_path" ]]; then
    echo "[ERROR] No projcet path is provided"
    exit 1
fi

if [[ ! -d "$build_path" ]]; then
    echo "[ERROR] No build path is provided"
    exit 1
fi

# copy ai_core operators implements
tbe_impl_files_num=$(ls $project_path/tbe/impl/* 2> /dev/null | wc -l)
if [[ "$tbe_impl_files_num" -gt 0 ]];then
    cp -r ${project_path}/tbe/impl/* ${build_path}/makepkg/packages/vendors/$vendor_name/op_impl/ai_core/tbe/mslite_impl
    cp -r ${project_path}/tbe/impl/* ${build_path}/makepkg/packages/vendors/$vendor_name/op_impl/vector_core/tbe/mslite_impl
fi

# copy aicpu kernel so operators
if [[ -d "${project_path}/cpukernel/aicpu_kernel_lib" ]]; then
    cp -f ${project_path}/cpukernel/aicpu_kernel_lib/* ${build_path}/makepkg/packages/vendors/$vendor_name/op_impl/cpu/aicpu_kernel/impl
    rm -rf ${project_path}/cpukernel/aicpu_kernel_lib
fi

# merge aicpu.ini and aicore.ini to generate npu_supported_ops.json
mkdir -p ${build_path}/framework/op_info_cfg
mkdir -p ${build_path}/framework/op_info_cfg/aicpu_kernel
mkdir -p ${build_path}/framework/op_info_cfg/ai_core

if [[ -d "${project_path}/tbe/op_info_cfg/ai_core" ]]; then
    bash ${project_path}/cmake/util/gen_ops_filter.sh ${project_path}/tbe/op_info_cfg/ai_core ${build_path}/framework/op_info_cfg/ai_core
fi

if [[ -d "${project_path}/cpukernel/op_info_cfg/aicpu_kernel" ]]; then
    bash ${project_path}/cmake/util/gen_ops_filter.sh ${project_path}/cpukernel/op_info_cfg/aicpu_kernel ${build_path}/framework/op_info_cfg/aicpu_kernel
fi

aicpu_filter_file=${build_path}/framework/op_info_cfg/aicpu_kernel/npu_supported_ops.json
aicore_filter_file=${build_path}/framework/op_info_cfg/ai_core/npu_supported_ops.json
if [[ -f "${aicpu_filter_file}" ]] && [[ ! -f "${aicore_filter_file}" ]]; then
    cp $aicpu_filter_file ${build_path}/makepkg/packages/vendors/$vendor_name/framework/tensorflow
fi
if [[ -f "${aicore_filter_file}" ]] && [[ ! -f "${aicpu_filter_file}" ]]; then
    cp $aicore_filter_file ${build_path}/makepkg/packages/vendors/$vendor_name/framework/tensorflow
fi

if [[ -f "${aicore_filter_file}" ]] && [[ -f "${aicpu_filter_file}" ]]; then
    chmod u+w ${aicpu_filter_file}
    python3 ${project_path}/cmake/util/insert_op_info.py ${aicore_filter_file} ${aicpu_filter_file}
    chmod u-w ${aicpu_filter_file}
    cp $aicpu_filter_file ${build_path}/makepkg/packages/vendors/$vendor_name/framework/tensorflow
fi

