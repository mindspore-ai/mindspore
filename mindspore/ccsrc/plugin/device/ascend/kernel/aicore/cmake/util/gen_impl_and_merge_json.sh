#!/usr/bin/bash
project_path=$1
build_path=$2
vendor_name=$3
if [[ ! -d "$project_path" ]]; then
    echo "[ERROR] No projcet path is provided"
    exit 1
fi

if [[ ! -d "$build_path" ]]; then
    echo "[ERROR] No build path is provided"
    exit 1
fi

# copy ai_core operators implements
tbe_impl_files_num=$(ls $project_path/tbe/impl/* 2>/dev/null | wc -l)
if [[ "$tbe_impl_files_num" -gt 0 ]]; then
    cp -r ${project_path}/tbe/impl/* ${build_path}/vendors/$vendor_name/op_impl/ai_core/tbe/${vendor_name}_impl/
    cp -r ${project_path}/tbe/impl/* ${build_path}/vendors/$vendor_name/op_impl/vector_core/tbe/${vendor_name}_impl/
fi
