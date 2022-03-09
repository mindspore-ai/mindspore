#!/bin/bash

function Run_3403() {
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${basepath}
  Run_3403_Samples
  if [ $? -eq 1 ]; then
    echo "samples failed"
    return 1
  fi
  Run_3403_Gate 'models_onnx_3403.cfg'
  if [ $? -eq 1 ]; then
    echo "onnx failed"
    return 1
  fi
  Run_3403_Gate 'models_tf_3403.cfg'
  if [ $? -eq 1 ]; then
    echo "tensorflow failed"
    return 1
  fi
}
# Run on 3403 platform:
function Run_3403_Samples() {

    # Run dpico converted models:
    while read line; do
        model_pass=${line:0:1}
        if [[ $model_pass == \#* || $model_pass == "" ]]; then
          continue
        fi
        dpico_line_info=${line}
        model_info=`echo ${dpico_line_info}|awk -F ' ' '{print $2}'`
        input_num=`echo ${dpico_line_info}|awk -F ' ' '{print $3}'`
        env_max_roi_num=`echo ${dpico_line_info}|awk -F ' ' '{print $5}'`
        accuracy_limit=`echo ${dpico_line_info}|awk -F ' ' '{print $6}'`
        cosine_distance_limit=`echo ${dpico_line_info}|awk -F ' ' '{print $7}'`
        nms_thr=`echo ${dpico_line_info}|awk -F ' ' '{print $8}'`
        score_thr=`echo ${dpico_line_info}|awk -F ' ' '{print $9}'`
        min_height=`echo ${dpico_line_info}|awk -F ' ' '{print $10}'`
        min_width=`echo ${dpico_line_info}|awk -F ' ' '{print $11}'`
        detection_all_net_out=`echo ${dpico_line_info}|awk -F ' ' '{print $12}'`
        model_name=${model_info%%;*}
        length=`expr ${#model_name} + 1`
        input_shapes=${model_info:${length}}
        input_files=''
        if [[ $input_num != 1 ]]; then
          for i in $(seq 1 $input_num)
          do
            input_files=$input_files${basepath}'/../../input_output/input/'${model_name}'.ms.bin_'$i','
          done
        else
          input_files=${basepath}/../../input_output/input/${model_name}.ms.bin
        fi

        DPICO_CONFIG_FILE=tmp.txt
        echo [dpico] > ${DPICO_CONFIG_FILE}
        echo MaxRoiNum=${env_max_roi_num} >> ${DPICO_CONFIG_FILE}
        echo NmsThreshold=${nms_thr} >> ${DPICO_CONFIG_FILE}
        echo ScoreThreshold=${score_thr} >> ${DPICO_CONFIG_FILE}
        echo MinHeight=${min_height} >> ${DPICO_CONFIG_FILE}
        echo MinWidth=${min_width} >> ${DPICO_CONFIG_FILE}
        if [ ${detection_all_net_out} == 1 ]; then
          echo DetectionPostProcess=on >> ${DPICO_CONFIG_FILE}
        else
          echo DetectionPostProcess=off >> ${DPICO_CONFIG_FILE}
        fi

        echo './benchmark --modelFile='${basepath}'/'${model_name}'.ms --inDataFile='${input_files}' --inputShapes='${input_shapes}' --benchmarkDataFile='${basepath}'/../../input_output/output_commercial/'${model_name}'.ms.out --accuracyThreshold='${accuracy_limit} >> "${run_3403_log_file}"
        ./benchmark --modelFile=${basepath}/${model_name}.ms --inDataFile=${input_files} --inputShapes=${input_shapes} --benchmarkDataFile=${basepath}/../../input_output/output_commercial/${model_name}.ms.out --accuracyThreshold=${accuracy_limit} --cosineDistanceThreshold=${cosine_distance_limit} --configFile=${DPICO_CONFIG_FILE}
        if [ $? = 0 ]; then
            run_result='benchmark: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='benchmark: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
    done < ${models_dpico_config}
}

# Run on 3403 platform:
function Run_3403_Gate() {
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${basepath}
  cfg_file=$1
  # Run dpico converted models:
  while read line; do
      model_pass=${line:0:1}
      if [[ $model_pass == \#* || $model_pass == "" ]]; then
        continue
      fi
      dpico_line_info=${line}
      model_info=`echo ${dpico_line_info}|awk -F ' ' '{print $1}'`
      input_num=`echo ${dpico_line_info}|awk -F ' ' '{print $2}'`
      env_max_roi_num=`echo ${dpico_line_info}|awk -F ' ' '{print $4}'`
      accuracy_limit=`echo ${dpico_line_info}|awk -F ' ' '{print $5}'`
      cosine_distance_limit=`echo ${dpico_line_info}|awk -F ' ' '{print $6}'`
      nms_thr=`echo ${dpico_line_info}|awk -F ' ' '{print $7}'`
      score_thr=`echo ${dpico_line_info}|awk -F ' ' '{print $8}'`
      min_height=`echo ${dpico_line_info}|awk -F ' ' '{print $9}'`
      min_width=`echo ${dpico_line_info}|awk -F ' ' '{print $10}'`
      detection_all_net_out=`echo ${dpico_line_info}|awk -F ' ' '{print $11}'`
      model_name=${model_info%%;*}
      length=`expr ${#model_name} + 1`
      input_shapes=${model_info:${length}}
      input_files=''
      if [[ $input_num != 1 ]]; then
        for i in $(seq 1 $input_num)
        do
          input_files=$input_files${basepath}'/../../input_output/input/'${model_name}'.ms.bin_'$i'.nchw,'
        done
      else
        input_files=${basepath}/../../input_output/input/${model_name}.ms.bin.nchw
      fi

      DPICO_CONFIG_FILE=tmp.txt
      echo [dpico] > ${DPICO_CONFIG_FILE}
      echo MaxRoiNum=${env_max_roi_num} >> ${DPICO_CONFIG_FILE}
      echo NmsThreshold=${nms_thr} >> ${DPICO_CONFIG_FILE}
      echo ScoreThreshold=${score_thr} >> ${DPICO_CONFIG_FILE}
      echo MinHeight=${min_height} >> ${DPICO_CONFIG_FILE}
      echo MinWidth=${min_width} >> ${DPICO_CONFIG_FILE}
      if [ ${detection_all_net_out} == "1" ]; then
        echo DetectionPostProcess=on >> ${DPICO_CONFIG_FILE}
      else
        echo DetectionPostProcess=off >> ${DPICO_CONFIG_FILE}
      fi

      echo './benchmark --modelFile='${basepath}'/'${model_name}'.ms --inDataFile='${input_files}' --benchmarkDataFile='${basepath}'/../../input_output/output/'${model_name}'.ms.out --accuracyThreshold='${accuracy_limit} >> "${run_3403_log_file}"
      ./benchmark --modelFile=${basepath}/${model_name}.ms --inDataFile=${input_files}  --benchmarkDataFile=${basepath}/../../input_output/output_commercial/${model_name}.ms.out --accuracyThreshold=${accuracy_limit} --cosineDistanceThreshold=${cosine_distance_limit} --configFile=${DPICO_CONFIG_FILE}
      if [ $? = 0 ]; then
          run_result='benchmark: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
      else
          run_result='benchmark: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
      fi
  done < ${cfg_file}
}

# Print start msg after run testcase
function MS_PRINT_TESTCASE_END_MSG() {
    echo -e "-----------------------------------------------------------------------------------------------------------------------------------"
}

basepath=$(pwd)
echo "on 3403, bashpath is ${basepath}"

# Set models config filepath
models_dpico_config=${basepath}/models_caffe_3403.cfg
echo ${models_dpico_config}

# Write benchmark result to temp file
run_benchmark_result_file=${basepath}/run_benchmark_result.txt
echo ' ' > ${run_benchmark_result_file}

run_3403_log_file=${basepath}/run_3403_log.txt
echo 'run 3403 logs: ' > ${run_3403_log_file}

echo "Running in 3403 ..."
Run_3403 &
Run_3403_PID=$!
sleep 1

wait ${Run_3403_PID}
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
