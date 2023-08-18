#!/usr/bin/bash

project_path=$1
build_path=$2
vendor_name=customize
if [[ ! -d "$project_path" ]]; then
    echo "[ERROR] No projcet path is provided"
    exit 1
fi

if [[ ! -d "$build_path" ]]; then
    echo "[ERROR] No build path is provided"
    exit 1
fi

# copy aicpu kernel so operators
if [[ -d "${project_path}/cpukernel/aicpu_kernel_lib" ]]; then
    cp -f ${project_path}/cpukernel/aicpu_kernel_lib/* ${build_path}/makepkg/packages/vendors/$vendor_name/op_impl/cpu/aicpu_kernel/impl
    rm -rf ${project_path}/cpukernel/aicpu_kernel_lib
fi
