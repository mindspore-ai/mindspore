#!/bin/bash

# Example:sh run_remote_ascend.sh -v version -b backend
while getopts "v:b:" opt; do
    case ${opt} in
        v)
            version=${OPTARG}
            echo "release version is ${OPTARG}"
            ;;
        b)
            backend=${OPTARG}
            echo "backend is ${backend}"
            ;;
        ?)
        echo "unknown para"
        exit 1;;
    esac
done

# Run Benchmark in Ascend platform:
function Run_Benchmark() {
    cd ${x86_path}/mindspore-lite-${version}-linux-x64/ || exit 1
    cp tools/benchmark/benchmark ./ || exit 1
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:./runtime/lib

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

        # model_info detail
        model_name=`echo ${model_info} | awk -F ';' '{print $1}'`
        input_info=`echo ${model_info} | awk -F ';' '{print $2}'`
        #input_shapes=`echo ${model_info} | awk -F ';' '{print $3}'`
        mode=`echo ${model_info} | awk -F ';' '{print $3}'`
        input_num=`echo ${input_info} | sed 's/:/;/' | awk -F ';' '{print $1}'`
        if [[ ${model_name##*.} == "caffemodel" ]]; then
            model_name=${model_name%.*}
        fi

        echo "Benchmarking ${model_name} ......"
        model_file=${ms_models_path}'/'${model_name}'.ms'
        input_files=""
        output_file=""
        data_path=${basepath}'/data/'
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

        # different tensorrt run mode use different cuda command
        echo './benchmark --modelFile='${model_file}' --inputShapes='${input_shapes}' --inDataFile='${input_files}' --benchmarkDataFile='${output_file}' --enableFp16='${enableFp16}' --accuracyThreshold='${acc_limit}' --device='${backend} >> "${run_ascend_log_file}"
        ./benchmark --modelFile=${model_file} --inputShapes=${input_shapes} --inDataFile=${input_files} --benchmarkDataFile=${output_file} --enableFp16=${enableFp16} --accuracyThreshold=${acc_limit} --device=${backend} >> ${run_ascend_log_file}

        if [ $? = 0 ]; then
            run_result=${backend}': '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result=${backend}': '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi

    done < ${models_ascend_config}
}

basepath=/home/ascend
x86_path=${basepath}/release
ms_models_path=${basepath}/ms_models
models_ascend_config=${basepath}/config/models_ascend.cfg
data_path=${basepath}/data
# Write benchmark result to temp file
run_benchmark_result_file=${basepath}/scripts/log/run_benchmark_result.txt
echo ' ' > ${run_benchmark_result_file}

####################  run simple Ascend models
run_ascend_log_file=${basepath}/scripts/log/run_benchmark_log.txt
echo 'run Ascend logs: ' > ${run_ascend_log_file}

echo "Start to run benchmark in ${backend} ..."
Run_Benchmark
Run_benchmark_status=$?
if [[ ${Run_benchmark_status} = 0 ]];then
    echo "Run_Benchmark success"
    Print_Benchmark_Result $run_benchmark_result_file
else
    echo "Run_Benchmark failed"
    cat ${run_ascend_log_file}
    Print_Benchmark_Result $run_benchmark_result_file
    exit 1
fi

exit ${Run_benchmark_status}
