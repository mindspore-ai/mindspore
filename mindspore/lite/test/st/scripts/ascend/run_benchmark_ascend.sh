#!/bin/bash
source ${benchmark_test}/run_benchmark_python.sh

# Example:sh run_remote_ascend.sh -v version -b backend
while getopts "v:b:d:a:c:p:" opt; do
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
        p)
            ascend_fail_not_return=${OPTARG}
            echo "ascend_fail_not_return is ${OPTARG}"
            ;;
        ?)
        echo "unknown para"
        exit 1;;
    esac
done

export ASCEND_DEVICE_ID=${device_id}
if [[ ${backend} =~ "_ge" ]]; then
    export ASCEND_BACK_POLICY="ge"
fi
# Run Benchmark in Ascend platform:
function Run_Benchmark() {
    cd ${benchmark_test}/mindspore-lite-${version}-linux-${arch} || exit 1
    cp tools/benchmark/benchmark ./ || exit 1
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:./runtime/lib

    local line_info model_info spec_acc_limit model_name input_num input_shapes \
            mode model_file input_files output_file data_path acc_limit enableFp16 \
            run_result elapsed_time ret

  for cfg_file in ${ascend_cfg_file_list[*]}; do
    while read line; do
        line_info=${line}
        if [[ $line_info == \#* || $line_info == "" ]]; then
            continue
        fi
        cfg_file_name=${cfg_file##*/}

        # model_info     accuracy_limit      run_mode
        model_info=`echo ${line_info} | awk -F ' ' '{print $1}'`
        spec_acc_limit=`echo ${line_info} | awk -F ' ' '{print $2}'`

        # model_info detail
        model_name=`echo ${model_info} | awk -F ';' '{print $1}'`
        input_info=`echo ${model_info} | awk -F ';' '{print $2}'`

        extra_info=`echo ${model_info} | awk -F ';' '{print $5}'`

        input_shapes=""
        infix=""
        # Without a configuration file, there is no need to set the inputshape
        if [[ ${cfg_file_name} =~ "_with_config_cloud_ascend" || ${extra_info} =~ "parallel_predict" ]]; then
          input_shapes=`echo ${model_info} | awk -F ';' '{print $3}'`
        elif [[ ${cfg_file_name} =~ "_on_the_fly_quant_ge_cloud" ]]; then
          infix="_on_the_fly_quant"
        elif [[ ${cfg_file_name} =~ "_full_quant_ge_cloud" ]]; then
          infix="_full_quant"
        fi
        mode=`echo ${model_info} | awk -F ';' '{print $3}'`
        input_num=`echo ${input_info} | sed 's/:/;/' | awk -F ';' '{print $1}'`
        if [[ ${model_name##*.} == "caffemodel" ]]; then
            model_name=${model_name%.*}
        fi

        echo "Benchmarking ${model_name} ......"
        model_type=${model_name##*.}
        if [[ ${compile_type} == "cloud" ]]; then
          model_file=${ms_models_path}'/'${model_name}${infix}'.mindir'
        else
          model_file=${ms_models_path}'/'${model_name}${infix}'.ms'
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
        if [[ ${cfg_file_name} =~ "_with_config_cloud_ascend" || ${cfg_file_name} =~ "_quant_ge_cloud" ]]; then
          echo "cfg file name: ${cfg_file_name}"
          input_files=""
          output_file=""
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
        use_parallel_predict="false"
        if [[ ${extra_info} =~ "parallel_predict" ]]; then
          use_parallel_predict="true"
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
        echo './benchmark --enableParallelPredict='${use_parallel_predict}' --modelFile='${model_file}' --inputShapes='${input_shapes}' --inDataFile='${input_files}' --benchmarkDataFile='${output_file}' --enableFp16='${enableFp16}' --accuracyThreshold='${acc_limit}' --device='${ascend_device} >> "${run_ascend_log_file}"
        elapsed_time=$(date +%s.%N)
        ./benchmark --enableParallelPredict=${use_parallel_predict} --modelFile=${model_file} --inputShapes=${input_shapes} --inDataFile=${input_files} --benchmarkDataFile=${output_file} --enableFp16=${enableFp16} --accuracyThreshold=${acc_limit} --device=${ascend_device} >> ${run_ascend_log_file}
        ret=$?
        elapsed_time=$(printf %.2f "$(echo "$(date +%s.%N) - $elapsed_time" | bc)")
        if [ ${ret} = 0 ]; then
          if [[ ${extra_info} =~ "parallel_predict" ]]; then
            run_result=${backend}': '${model_name}' '${elapsed_time}' parallel_pass'; echo ${run_result} >> ${run_benchmark_result_file}
          else
            run_result=${backend}': '${model_name}' '${elapsed_time}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
          fi
        else
          if [[ ${extra_info} =~ "parallel_predict" ]]; then
            run_result=${backend}': '${model_name}' '${elapsed_time}' parallel_failed'; echo ${run_result} >> ${run_benchmark_result_file}
          else
            run_result=${backend}': '${model_name}' '${elapsed_time}' failed'; echo ${run_result} >> ${run_benchmark_result_file}
          fi
            if [[ ${ascend_fail_not_return} != "ON" ]]; then
                return 1
            fi
        fi

    done < ${cfg_file}
  done
}

user_name=${USER}
benchmark_test=/home/${user_name}/benchmark_test/${device_id}
ms_models_path=${benchmark_test}/ms_models
ascend_cfg_file_list=()
if [[ ${backend} =~ "lite" ]]; then
    models_ascend_config=${benchmark_test}/models_ascend_lite.cfg
    ascend_cfg_file_list=("$models_ascend_config")
elif [[ ${backend} =~ "cloud" ]]; then
    models_ascend_config=${benchmark_test}/models_ascend_cloud.cfg
    models_ascend_with_config=${benchmark_test}/models_with_config_cloud_ascend.cfg
    ascend_cfg_file_list=("$models_ascend_config" "$models_ascend_with_config")
    if [[ ${backend} =~ "_ge" ]]; then
        models_ascend_config=${benchmark_test}/models_ascend_ge_cloud.cfg
        models_ascend_on_the_fly_quant_config=${benchmark_test}/models_ascend_on_the_fly_quant_ge_cloud.cfg
        models_ascend_fake_model_on_the_fly_quant_config=${benchmark_test}/models_ascend_fake_model_on_the_fly_quant_ge_cloud.cfg
        models_ascend_fake_model_full_quant_config=${benchmark_test}/models_ascend_fake_model_full_quant_ge_cloud.cfg
        ascend_cfg_file_list=("$models_ascend_on_the_fly_quant_config" "$models_ascend_config" "$models_ascend_fake_model_on_the_fly_quant_config" "$models_ascend_fake_model_full_quant_config")
    fi
fi
model_data_path=/home/workspace/mindspore_dataset/mslite

# Write benchmark result to temp file
run_benchmark_result_file=${benchmark_test}/run_benchmark_result.txt
echo ' ' > ${run_benchmark_result_file}

####################  run simple Ascend models
run_ascend_log_file=${benchmark_test}/run_benchmark_log.txt
echo 'run Ascend logs: ' > ${run_ascend_log_file}

echo "Start to run benchmark in ${backend}, device id ${device_id}..."
if [[ ${backend} =~ "ascend310" ]]; then
  ascend_device=Ascend
elif [[ ${backend} =~ "ascend310P" ]]; then
  ascend_device=Ascend
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

# Run Benchmark cloud inference
if [[ ${backend} =~ "cloud" &&! ${backend} =~ "ge" ]]; then
    echo "Run cloud fusion inference benchmark"
    source ${benchmark_test}/run_benchmark_cloud_ascend.sh -v ${version} -b ${backend} -d ${device_id} -a ${arch} -c ${compile_type}
    Run_benchmark_status=$?
else
    echo "Skip cloud fusion inference benchmark"
fi

# run python ST
if [[ ${backend} =~ "cloud" &&! ${backend} =~ "ge" ]]; then
  models_python_config=${benchmark_test}/models_python_ascend.cfg
  models_python_cfg_file_list=("$models_python_config")
  Run_python_ST ${benchmark_test} ${benchmark_test} ${ms_models_path} ${model_data_path}'/models/hiai' "${models_python_cfg_file_list[*]}" "Ascend"
  Run_python_status=$?
  if [[ ${Run_python_status} != 0 ]];then
      cat ${run_ascend_log_file}
      echo "Run_python_status failed"
      exit 1
  fi
  Run_python_ST ${benchmark_test} ${benchmark_test} ${ms_models_path} ${model_data_path}'/models/hiai' "${models_python_cfg_file_list[*]}" "Ascend_Model_Group"
  Run_python_model_group_status=$?
  if [[ ${Run_python_model_group_status} != 0 ]];then
      cat ${run_ascend_log_file}
      echo "Run_python_model_group_status failed"
      exit 1
  fi
fi

if [[ ${backend} =~ "cloud" &&! ${backend} =~ "ge" ]]; then
  export LITE_ST_MODEL=${model_data_path}/models/hiai/mindspore_uniir_mobilenetv2.mindir
  export LITE_ST_CPP_DIR=${benchmark_test}/cpp
  bash ${benchmark_test}/run_device_mem_test.sh > run_device_mem_test.log
  Run_device_example_status=$?
  if [[ ${Run_device_example_status} != 0 ]];then
    echo "Run device example failed"
    cat run_device_mem_test.log
    exit 1
  else
    echo "Run device example success"
  fi
else
  echo "Skip run device example, while backend is ${backend}"
fi

exit ${Run_benchmark_status}
