#!/bin/bash

# Run on hi3516 platform:
function Run_Hi3516() {
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${basepath}:/usr/lib:/lib

    # Run nnie converted models:
    while read line; do
        model_pass=${line:0:1}
        if [[ $model_pass == \# ]]; then
          continue
        fi
        nnie_line_info=${line}
        model_info=`echo ${nnie_line_info}|awk -F ' ' '{print $2}'`
        input_info=`echo ${nnie_line_info}|awk -F ' ' '{print $3}'`
        env_time_step=`echo ${nnie_line_info}|awk -F ' ' '{print $4}'`
        env_max_roi_num=`echo ${nnie_line_info}|awk -F ' ' '{print $5}'`
        accuracy_limit=`echo ${nnie_line_info}|awk -F ' ' '{print $6}'`
        model_name=${model_info%%;*}
        length=`expr ${#model_name} + 1`
        input_shapes=${model_info:${length}}
        input_files=''
        if [[ $input_info != 1 ]]; then
          input_num=`echo ${input_info} | awk -F ':' '{print $1}'`
          input_seq=`echo ${input_info} | awk -F ':' '{print $2}'`
          if [[ "${input_seq}" == "" ]]; then
            for i in $(seq 1 $input_num)
            do
              input_files=${input_files}${basepath}'/../input_output/input/'${model_name}'.ms.bin_'$i','
            done
          else
            for i in $(seq 1 $input_num)
            do
              cur_input_num=${input_seq%%,*}
              input_seq=${input_seq#*,}
              input_files=${input_files}${basepath}'/../input_output/input/'${model_name}'.ms.bin_'$cur_input_num','
            done
          fi
        else
          input_files=${basepath}/../input_output/input/${model_name}.ms.bin
        fi
        export TIME_STEP=${env_time_step}
        export MAX_ROI_NUM=${env_max_roi_num}

        echo './benchmark --modelFile='${basepath}'/'${model_name}'.ms --inDataFile='${input_files}' --inputShapes='${input_shapes}' --benchmarkDataFile='${basepath}'/../input_output/output/'${model_name}'.ms.out --accuracyThreshold='${accuracy_limit} >> "${run_hi3516_log_file}"
        ./benchmark --modelFile=${basepath}/${model_name}.ms --inDataFile=${input_files} --inputShapes=${input_shapes} --benchmarkDataFile=${basepath}/../input_output/output/${model_name}.ms.out --accuracyThreshold=${accuracy_limit} >> "${run_hi3516_log_file}"
        if [ $? = 0 ]; then
            run_result='hi3516: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='hi3516: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
    done < ${models_nnie_config}
}

# Print start msg after run testcase
function MS_PRINT_TESTCASE_END_MSG() {
    echo -e "-----------------------------------------------------------------------------------------------------------------------------------"
}

basepath=$(pwd)
echo "on hi3516, bashpath is ${basepath}"

# Set models config filepath
models_nnie_config=${basepath}/models_nnie.cfg
echo ${models_nnie_config}

# Write benchmark result to temp file
run_benchmark_result_file=${basepath}/run_benchmark_result.txt
echo ' ' > ${run_benchmark_result_file}

run_hi3516_log_file=${basepath}/run_hi3516_log.txt
echo 'run hi3516 logs: ' > ${run_hi3516_log_file}

echo "Running in hi3516 ..."
Run_Hi3516 &
Run_hi3516_PID=$!
sleep 1

wait ${Run_hi3516_PID}
Run_benchmark_status=$?

# Check converter result and return value
if [[ ${Run_benchmark_status} = 0 ]];then
    echo "Run benchmark success"
    MS_PRINT_TESTCASE_END_MSG
    cat ${run_benchmark_result_file}
    MS_PRINT_TESTCASE_END_MSG
    exit 0
else
    echo "Run benchmark failed"
    MS_PRINT_TESTCASE_END_MSG
    cat ${run_benchmark_result_file}
    MS_PRINT_TESTCASE_END_MSG
    exit 1
fi