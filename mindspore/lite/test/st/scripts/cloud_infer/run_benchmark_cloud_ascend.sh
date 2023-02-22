#!/bin/bash
source ${benchmark_test}/run_benchmark_python.sh

# Example:sh run_remote_ascend.sh -v version -b backend
while getopts "v:b:d:a:c:" opt; do
    case ${opt} in
        v)
            version=${OPTARG}
            echo "release version is ${OPTARG}"
            ;;
        b)
            backend=${OPTARG}
            echo "backend is ${backend}"
            ;;
        d)
            device_id=${OPTARG}
            echo "device id is ${device_id}"
            ;;
        a)
            arch=${OPTARG}
            echo "arch is ${arch}"
            ;;
        c)
            compile_type=${OPTARG}
            echo "compile type is ${compile_type}"
            ;;
        ?)
        echo "unknown para"
        exit 1;;
    esac
done

export ASCEND_DEVICE_ID=${device_id}

# Run Benchmark in Ascend platform:
function Run_Benchmark() {
    cd ${benchmark_test}/mindspore-lite-${version}-linux-${arch} || exit 1
    cp tools/benchmark/benchmark ./ || exit 1
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:./runtime/lib
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:./tools/converter/lib/:./runtime/third_party/glog:./runtime/third_party/libjpeg-turbo/lib
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:./runtime/third_party/dnnl

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
        model_type=${model_name##*.}
        if [[ ${compile_type} == "cloud" ]]; then
          model_file=${models_path}'/'${model_name}'.mindir'
        else
          model_file=${models_path}'/'${model_name}'.ms'
        fi
        input_files=""
        output_file=""
        data_path=${model_data_path}'/models/hiai/input_output/'
        if [[ ${model_type} == "mindir" || ${model_type} == "ms" ]]; then
          if [[ ${input_num} == "" || ${input_num} == 1 ]]; then
            input_files=${data_path}'input/'${model_name}'.ms.bin'
          else
            for i in $(seq 1 $input_num)
            do
            input_files=${input_files}${data_path}'input/'${model_name}'.ms.bin_'$i','
            done
          fi
          output_file=${data_path}'output/'${model_name}'.ms.out'
        else
          if [[ ${input_num} == "" || ${input_num} == 1 ]]; then
            input_files=${data_path}'input/'${model_name}'.bin'
          else
            for i in $(seq 1 $input_num)
            do
            input_files=${input_files}${data_path}'input/'${model_name}'.bin_'$i','
            done
          fi
          output_file=${data_path}'output/'${model_name}'.out'
        fi



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
        echo './benchmark --modelFile='${model_file}' --inputShapes='${input_shapes}' --inDataFile='${input_files}' --benchmarkDataFile='${output_file}' --enableFp16='${enableFp16}' --accuracyThreshold='${acc_limit}' --device='${ascend_device} >> "${run_ascend_log_file}"
        ./benchmark --modelFile=${model_file} --inputShapes=${input_shapes} --inDataFile=${input_files} --benchmarkDataFile=${output_file} --enableFp16=${enableFp16} --accuracyThreshold=${acc_limit} --device=${ascend_device} >> ${run_ascend_log_file}

        if [ $? = 0 ]; then
            run_result=${backend}': '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result=${backend}': '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi

    done < ${models_ascend_config}
}

user_name=${USER}
benchmark_test=/home/${user_name}/benchmark_test/${device_id}
models_ascend_config=${benchmark_test}/models_mindir_cloud_ascend.cfg
model_data_path=/home/workspace/mindspore_dataset/mslite
models_path=${model_data_path}/models/hiai

# Write benchmark result to temp file
run_benchmark_result_file=${benchmark_test}/run_benchmark_result.txt
echo ' ' > ${run_benchmark_result_file}

####################  run simple Ascend models
run_ascend_log_file=${benchmark_test}/run_benchmark_log.txt
echo 'run Ascend logs: ' > ${run_ascend_log_file}

echo "Start to run benchmark in ${backend}, device id ${device_id}..."
if [[ ${backend} =~ "ascend310" ]]; then
  ascend_device=Ascend310
elif [[ ${backend} =~ "ascend310P" ]]; then
  ascend_device=Ascend310P
else
  echo "${backend} is not support."
  exit 1
fi
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

return ${Run_benchmark_status}
