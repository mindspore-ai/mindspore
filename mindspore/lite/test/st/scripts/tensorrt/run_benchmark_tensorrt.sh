#!/bin/bash

# Run on NVIDIA TensorRT platform:
function Run_TensorRT() {
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./
    source /etc/profile
    local line_info model_info spec_acc_limit model_name input_num input_shapes \
            mode model_file input_files output_file data_path acc_limit enableFp16 \
            run_result

    while read line; do
        line_info=${line}
        if [[ $line_info == \#* || $line_info == "" ]]; then
            continue
        fi

        # model_info     accuracy_limit      run_mode
        model_info=`echo ${line_info} | awk -F ' ' '{print $1}'`
        spec_acc_limit=`echo ${line_info} | awk -F ' ' '{print $2}'`
        run_mode=`echo ${line_info} | awk -F ' ' '{print $3}'`

        # model_info detail
        model_name=`echo ${model_info} | awk -F ';' '{print $1}'`
        input_info=`echo ${model_info} | awk -F ';' '{print $2}'`
        input_shapes=`echo ${model_info} | awk -F ';' '{print $3}'`
        mode=`echo ${model_info} | awk -F ';' '{print $3}'`
        input_num=`echo ${input_info} | sed 's/:/;/' | awk -F ';' '{print $1}'`
        if [[ ${model_name##*.} == "caffemodel" ]]; then
            model_name=${model_name%.*}
        fi

        # run_mode detail
        run_mode_name=`echo ${run_mode} | awk -F ';' '{print $1}'`
        run_mode_size=`echo ${run_mode} | awk -F ';' '{print $2}'`

        echo "Benchmarking ${model_name} ......"
        model_file=${basepath}'/'${model_name}'.ms'
        input_files=""
        output_file=""
        data_path=${basepath}'/../../input_output/'
        if [[ ${input_num} == "" || ${input_num} == 1 ]]; then
            input_files=${data_path}'input/'${model_name}'.ms.bin'
        else
            for i in $(seq 1 $input_num)
            do
            input_files=${input_files}${data_path}'input/'${model_name}'.ms.bin_'$i','
            done
        fi
        output_file=${data_path}'output/'${model_name}'.ms.out'
        # set accuracy limitation
        acc_limit="0.5"
        if [[ ${spec_acc_limit} != "" ]]; then
            acc_limit="${spec_acc_limit}"
        elif [[ ${mode} == "fp16" ]]; then
            acc_limit="5"
        fi
        # whether enable fp16
        enableFp16="false"
        if [[ ${mode} == "fp16" ]]; then
            enableFp16="true"
        fi

        # converter for distribution models
        if [[ ${run_mode_name} == "CONVERTER" ]]; then
            continue
        fi

        # different tensorrt run mode use different cuda command
        if [[ ${run_mode_name} == "DIS" ]]; then
            export ENABLE_NEW_API=true
            echo 'mpirun -np '${run_mode_size}' ./benchmark --modelFile='${model_file}' --inputShapes='${input_shapes}' --inDataFile='${input_files}' --benchmarkDataFile='${output_file}' --enableFp16='${enableFp16}' --accuracyThreshold='${acc_limit}' --device=GPU' >> "${run_tensorrt_log_file}"
            mpirun -np ${run_mode_size} ./benchmark --modelFile=${model_file} --inputShapes=${input_shapes} --inDataFile=${input_files} --benchmarkDataFile=${output_file} --enableFp16=${enableFp16} --accuracyThreshold=${acc_limit} --device=GPU >> ${run_tensorrt_log_file}
        else
            echo 'CUDA_VISILE_DEVICE='${cuda_device_id}' ./benchmark --modelFile='${model_file}' --inputShapes='${input_shapes}' --inDataFile='${input_files}' --benchmarkDataFile='${output_file}' --enableFp16='${enableFp16}' --accuracyThreshold='${acc_limit}' --device=GPU' >> "${run_tensorrt_log_file}"
            CUDA_VISILE_DEVICE=${cuda_device_id} ./benchmark --modelFile=${model_file} --inputShapes=${input_shapes} --inDataFile=${input_files} --benchmarkDataFile=${output_file} --enableFp16=${enableFp16} --accuracyThreshold=${acc_limit} --device=GPU >> ${run_tensorrt_log_file}
        fi

        if [ $? = 0 ]; then
            run_result='TensorRT'${run_mode_name}': '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='TensorRT'${run_mode_name}': '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi

    done < ${models_tensorrt_config}
}

# Print start msg before run testcase
function MS_PRINT_TESTCASE_START_MSG() {
    echo ""
    echo -e "-----------------------------------------------------------------------------------------------------------------------------------"
    echo -e "env                    Testcase                                                                                           Result   "
    echo -e "---                    --------                                                                                           ------   "
}

# Print start msg after run testcase
function MS_PRINT_TESTCASE_END_MSG() {
    echo -e "-----------------------------------------------------------------------------------------------------------------------------------"
}

function Print_Benchmark_Result() {
    MS_PRINT_TESTCASE_START_MSG
    while read line; do
        arr=("${line}")
        printf "%-20s %-100s %-7s\n" ${arr[0]} ${arr[1]} ${arr[2]}
    done < $1
    MS_PRINT_TESTCASE_END_MSG
}

basepath=$(pwd)
echo "on tensorrt device, bashpath is ${basepath}"

# Example:sh run_benchmark_tensorrt.sh -d 0
while getopts "d:" opt; do
    case ${opt} in
        d)
            cuda_device_id=${OPTARG}
            echo "cuda_device_id is ${cuda_device_id}."
            ;;
        ?)
        echo "unknown para"
        exit 1;;
    esac
done

# Set models config filepath
models_tensorrt_config=${basepath}/models_tensorrt.cfg
echo ${models_tensorrt_config}

# Write benchmark result to temp file
run_benchmark_result_file=${basepath}/run_benchmark_result.txt
echo ' ' > ${run_benchmark_result_file}

run_tensorrt_log_file=${basepath}/run_tensorrt_log.txt
echo 'run tensorrt logs: ' > ${run_tensorrt_log_file}

echo "Running in tensorrt on device ${cuda_device_id} ..."
Run_TensorRT &
Run_TensorRT_PID=$!
sleep 1

wait ${Run_TensorRT_PID}
Run_benchmark_status=$?

# Check converter result and return value
echo "Run x86 TensorRT GPU ended on device ${cuda_device_id}"

if [[ ${Run_benchmark_status} != 0 ]];then
    echo "Run_TensorRT failed"
    cat ${run_tensorrt_log_file}
fi

Print_Benchmark_Result $run_benchmark_result_file
exit ${Run_benchmark_status}