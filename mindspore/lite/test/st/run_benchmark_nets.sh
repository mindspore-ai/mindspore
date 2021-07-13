#!/bin/bash

# Example:sh call_scipts.sh -r /home/temp_test -m /home/temp_test/models -d "8KE5T19620002408" -e arm_cpu
while getopts "r:m:d:e:" opt; do
    case ${opt} in
        r)
            release_path=${OPTARG}
            echo "release_path is ${OPTARG}"
            ;;
        m)
            models_path=${OPTARG}
            echo "models_path is ${OPTARG}"
            ;;
        d)
            device_id=${OPTARG}
            echo "device_id is ${OPTARG}"
            ;;
        e)
            backend=${OPTARG}
            echo "backend is ${OPTARG}"
            ;;
        ?)
        echo "unknown para"
        exit 1;;
    esac
done

cur_path=$(pwd)
echo "cur_path is "$cur_path

if [[ $backend == "all" || $backend == "arm64_cpu" || $backend == "arm64_fp32" || $backend == "arm64_fp16" || \
        $backend == "arm64_codegen" ]]; then
    sh $cur_path/scripts/run_benchmark_arm64.sh -r $release_path -m $models_path -d $device_id -e $backend
    arm64_status=$?
    if [[ $arm64_status -ne 0 ]]; then
      echo "Run arm64 failed"
      exit 1
    fi
    # run codegen
    sh $cur_path/scripts/run_benchmark_codegen.sh -r $release_path -m $models_path -d $device_id -e "arm64_codegen"
    arm64_status=$?
    if [[ $arm64_status -ne 0 ]]; then
      echo "Run arm64 codegen failed"
      exit 1
    fi
    # run train
    sh $cur_path/scripts/run_net_train.sh -r $release_path -m ${models_path}/../../models_train -d $device_id -e "arm64_train"
    arm64_status=$?
    if [[ $arm64_status -ne 0 ]]; then
      echo "Run arm64 train failed"
      exit 1
    fi
fi

if [[ $backend == "all" || $backend == "arm32_cpu" || $backend == "arm32_fp32" || $backend == "arm32_fp16" || \
      $backend == "arm32_codegen" ]]; then
    sh $cur_path/scripts/run_benchmark_arm32.sh -r $release_path -m $models_path -d $device_id -e $backend
    arm32_status=$?
    if [[ $arm32_status -ne 0 ]]; then
      echo "Run arm32 failed"
      exit 1
    fi
    # run codegen
    sh $cur_path/scripts/run_benchmark_codegen.sh -r $release_path -m $models_path -d $device_id -e "arm32_codegen"
    arm32_status=$?
    if [[ $arm32_status -ne 0 ]]; then
      echo "Run arm32 codegen failed"
      exit 1
    fi
    # run train
    sh $cur_path/scripts/run_net_train.sh -r $release_path -m ${models_path}/../../models_train -d $device_id -e "arm32_train"
    arm32_status=$?
    if [[ $arm32_status -ne 0 ]]; then
      echo "Run arm32 train failed"
      exit 1
    fi
fi

if [[ $backend == "all" || $backend == "gpu" ]]; then
    sh $cur_path/scripts/run_benchmark_gpu.sh -r $release_path -m $models_path -d $device_id -e $backend
    gpu_status=$?
    if [[ $gpu_status -ne 0 ]]; then
      echo "Run gpu failed"
      exit 1
    fi
fi

if [[ $backend == "all" || $backend == "npu" ]]; then
    sh $cur_path/scripts/run_benchmark_npu.sh -r $release_path -m $models_path -d $device_id -e $backend
    npu_status=$?
    if [[ $npu_status -ne 0 ]]; then
      echo "Run npu failed"
      exit 1
    fi
fi

if [[ $backend == "all" || $backend == "x86-all" || $backend == "x86" || $backend == "x86-sse" || \
      $backend == "x86-avx" || $backend == "x86-java" || $backend == "x86_codegen" ]]; then
    sh $cur_path/scripts/run_benchmark_x86.sh -r $release_path -m $models_path -e $backend
    x86_status=$?
    if [[ $x86_status -ne 0 ]]; then
      echo "Run x86 failed"
      exit 1
    fi
    # run codegen
    sh $cur_path/scripts/run_benchmark_codegen.sh -r $release_path -m $models_path -e "x86_codegen"
    x86_status=$?
    if [[ $x86_status -ne 0 ]]; then
      echo "Run x86 codegen failed"
      exit 1
    fi
    # run train
    sh $cur_path/scripts/run_net_train.sh -r $release_path -m ${models_path}/../../models_train -e "x86_train"
    x86_status=$?
    if [[ $x86_status -ne 0 ]]; then
      echo "Run x86 train failed"
      exit 1
    fi
fi

if [[ $backend == "all" || $backend == "arm32_3516D" ]]; then
    exit 0
#    sh $cur_path/scripts/nnie/run_converter_nnie.sh -r $release_path -m $models_path -d $device_id -e $backend
#    hi3516_status=$?
#    if [[ $hi3516_status -ne 0 ]]; then
#      echo "Run nnie hi3516 failed"
#      exit 1
#    fi
fi
