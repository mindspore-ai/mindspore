#!/bin/bash
source ./scripts/base_functions.sh

# Run converter on x86 platform:
function Run_Converter() {
    # Unzip x86 runtime and converter
    cd ${x86_path} || exit 1
    if [[ $backend != "linux_arm64_tflite" ]]; then
      tar -zxf ${x86_path}/avx/mindspore-lite-${version}-linux-x64.tar.gz || exit 1
    fi
    tar -zxf mindspore-lite-${version}-linux-*.tar.gz || exit 1
    cd ${x86_path}/mindspore-lite-${version}-linux-*/ || exit 1

    cp tools/converter/converter/converter_lite ./ || exit 1
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:./tools/converter/lib/:./tools/converter/third_party/glog/lib

    rm -rf ${ms_models_path}
    mkdir -p ${ms_models_path}

    # Convert TFLite parallel_split models:
    if [[ $backend == "x86-all" || $backend == "x86_tflite" ]]; then
      while read line; do
          parallel_split_line_info=${line}
          if [[ $parallel_split_line_info == \#* || $parallel_split_line_info == "" ]]; then
            continue
          fi
          model_name=`echo ${parallel_split_line_info}|awk -F ' ' '{print $1}'`
          echo ${model_name} >> "${run_converter_log_file}"
          echo 'convert mode name: '${model_name}' begin.'
          echo './converter_lite  --fmk=TFLITE --modelFile='${models_path}'/'${model_name}' --outputFile='${ms_models_path}'/'${model_name}_1_1_parallel_split' --config_file='${models_path}'/offline_parallel_split/1_1_parallel_split.config' >> "${run_converter_log_file}"
          ./converter_lite  --fmk=TFLITE --modelFile=$models_path/${model_name} --outputFile=${ms_models_path}/${model_name}_1_1_parallel_split --configFile=${models_path}/offline_parallel_split/1_1_parallel_split.config
          if [ $? = 0 ]; then
              converter_result='converter 1_1_parallel_split '${model_name}' pass';echo ${converter_result} >> ${run_converter_result_file}
          else
              converter_result='converter 1_1_parallel_split '${model_name}' failed';echo ${converter_result} >> ${run_converter_result_file}
              if [[ $x86_fail_not_return != "ON" ]]; then
                  return 1
              fi
          fi
          echo './converter_lite  --fmk=TFLITE --modelFile='${models_path}'/'${model_name}' --outputFile='${ms_models_path}'/'${model_name}_1_2_parallel_split' --config_file='${models_path}'/offline_parallel_split/1_2_parallel_split.config' >> "${run_converter_log_file}"
          ./converter_lite  --fmk=TFLITE --modelFile=$models_path/${model_name} --outputFile=${ms_models_path}/${model_name}_1_2_parallel_split --configFile=${models_path}/offline_parallel_split/1_2_parallel_split.config
          if [ $? = 0 ]; then
              converter_result='converter 1_2_parallel_split '${model_name}' pass';echo ${converter_result} >> ${run_converter_result_file}
          else
              converter_result='converter 1_2_parallel_split '${model_name}' failed';echo ${converter_result} >> ${run_converter_result_file}
              if [[ $x86_fail_not_return != "ON" ]]; then
                  return 1
              fi
          fi
          echo './converter_lite  --fmk=TFLITE --modelFile='${models_path}'/'${model_name}' --outputFile='${ms_models_path}'/'${model_name}_1_3_parallel_split' --config_file='${models_path}'/offline_parallel_split/1_3_parallel_split.config' >> "${run_converter_log_file}"
          ./converter_lite  --fmk=TFLITE --modelFile=$models_path/${model_name} --outputFile=${ms_models_path}/${model_name}_1_3_parallel_split --configFile=${models_path}/offline_parallel_split/1_3_parallel_split.config
          if [ $? = 0 ]; then
              converter_result='converter 1_3_parallel_split '${model_name}' pass';echo ${converter_result} >> ${run_converter_result_file}
          else
              converter_result='converter 1_3_parallel_split '${model_name}' failed';echo ${converter_result} >> ${run_converter_result_file}
              if [[ $x86_fail_not_return != "ON" ]]; then
                  return 1
              fi
          fi
      done < ${models_tflite_parallel_split_config}
    fi
    # Convert models:
    # $1:cfgFileList; $2:inModelPath; $3:outModelPath; $4:logFile; $5:resultFile;
    Convert "${x86_cfg_file_list[*]}" $models_path $ms_models_path $run_converter_log_file $run_converter_result_file $x86_fail_not_return
}

# Run on x86 platform:
function Run_x86() {
    # $1:framework;
    echo 'cd  '${x86_path}'/mindspore-lite-'${version}'-linux-*' >> "${run_x86_log_file}"
    cd ${x86_path}/mindspore-lite-${version}-linux-*/ || exit 1
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./runtime/lib:./runtime/third_party/glog
    cp tools/benchmark/benchmark ./ || exit 1
    # Run converted models:
    # $1:cfgFileList; $2:modelPath; $3:dataPath; $4:logFile; $5:resultFile; $6:platform; $7:processor; $8:phoneId;
    Run_Benchmark "${x86_cfg_file_list[*]}" $ms_models_path $models_path $run_x86_log_file $run_benchmark_result_file 'x86' 'CPU' '' $x86_fail_not_return
}

# Run on x86 sse platform:
function Run_x86_sse() {
    cd ${x86_path}/sse || exit 1
    tar -zxf mindspore-lite-${version}-linux-x64.tar.gz || exit 1
    cd ${x86_path}/sse/mindspore-lite-${version}-linux-x64 || exit 1
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./runtime/lib:./runtime/third_party/glog
    cp tools/benchmark/benchmark ./ || exit 1

    # Run converted models:
    # $1:cfgFileList; $2:modelPath; $3:dataPath; $4:logFile; $5:resultFile; $6:platform; $7:processor; $8:phoneId;
    Run_Benchmark "${x86_cfg_file_list[*]}" $ms_models_path $models_path $run_x86_sse_log_file $run_benchmark_result_file 'x86_sse' 'CPU' '' $x86_fail_not_return
}

# Run on x86 avx platform:
function Run_x86_avx() {
    cd ${x86_path}/avx || exit 1
    tar -zxf mindspore-lite-${version}-linux-x64.tar.gz || exit 1
    cd ${x86_path}/avx/mindspore-lite-${version}-linux-x64 || exit 1
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./runtime/lib:./runtime/third_party/glog
    cp tools/benchmark/benchmark ./ || exit 1

    # Run converted models:
    # $1:cfgFileList; $2:modelPath; $3:dataPath; $4:logFile; $5:resultFile; $6:platform; $7:processor; $8:phoneId; $9:benchmark_mode
    Run_Benchmark "${x86_cfg_file_list[*]}" $ms_models_path $models_path $run_x86_avx_log_file $run_benchmark_result_file 'x86_avx' 'CPU' '' $x86_fail_not_return
}

# Run on x86 avx512 platform:
function Run_x86_avx512() {
    cd ${x86_path}/avx512 || exit 1
    tar -zxf mindspore-lite-${version}-linux-x64.tar.gz || exit 1
    cd ${x86_path}/avx512/mindspore-lite-${version}-linux-x64 || exit 1
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./runtime/lib:./runtime/third_party/glog
    cp tools/benchmark/benchmark ./ || exit 1

    # Run converted models:
    # $1:cfgFileList; $2:modelPath; $3:dataPath; $4:logFile; $5:resultFile; $6:platform; $7:processor; $8:phoneId; $9:benchmark_mode
    Run_Benchmark "${x86_cfg_file_list[*]}" $ms_models_path $models_path $run_x86_avx512_log_file $run_benchmark_result_file 'x86_avx512' 'CPU' '' $x86_fail_not_return
}

# Run on x86 java platform:
function Run_x86_java() {
    cd ${x86_path} || exit 1
    mkdir java || exit 1
    cp ${x86_path}/avx/mindspore-lite-${version}-linux-x64.tar.gz ./java/ || exit 1
    cd ./java || exit 1
    tar -zxf mindspore-lite-${version}-linux-x64.tar.gz || exit 1
    # compile benchmark
    echo "javac -cp ${x86_path}/java/mindspore-lite-${version}-linux-x64/runtime/lib/mindspore-lite-java.jar ${basepath}/java/src/main/java/Benchmark.java -d ."
    javac -cp ${x86_path}/java/mindspore-lite-${version}-linux-x64/runtime/lib/mindspore-lite-java.jar ${basepath}/java/src/main/java/Benchmark.java -d .

    count=0
    # Run tflite converted models:
    while read line; do
        # only run top5.
        count=`expr ${count}+1`
        if [[ ${count} -gt 5 ]]; then
            break
        fi
        model_name=`echo ${line} | awk -F ';' '{print $1}'`
        if [[ $model_name == \#* ]]; then
          continue
        fi
        echo $LD_LIBRARY_PATH >> "${run_x86_java_log_file}"
        echo ${model_name} >> "${run_x86_java_log_file}"
        echo "java -classpath .:${x86_path}/java/mindspore-lite-${version}-linux-x64/runtime/lib/mindspore-lite-java.jar Benchmark ${ms_models_path}/${model_name}.ms '${models_path}'/input_output/input/${model_name}.ms.bin '${models_path}'/input_output/output/${model_name}.ms.out 1" >> "${run_x86_java_log_file}"
        java -classpath .:${x86_path}/java/mindspore-lite-${version}-linux-x64/runtime/lib/mindspore-lite-java.jar Benchmark ${ms_models_path}/${model_name}.ms ${models_path}/input_output/input/${model_name}.ms.bin ${models_path}/input_output/output/${model_name}.ms.out 1 >> ${run_x86_java_log_file}
        cat ${run_x86_java_log_file}
        if [ $? = 0 ]; then
          run_result='x86_java: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
          run_result='x86_java: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}
          if [[ $x86_fail_not_return != "ON" ]]; then
            return 1
          fi
        fi
        sleep 1
    done < ${models_tflite_config}
}

function Run_x86_parallel_split() {
    echo 'cd  '${x86_path}'/mindspore-lite-'${version}'-linux-*' >> "${run_x86_log_file}"
    cd ${x86_path}/mindspore-lite-${version}-linux-* || exit 1
    rm -rf parallel_split
    mkdir parallel_split
    cd parallel_split || exit 1
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:../runtime/lib:../runtime/third_party/glog
    cp ../tools/benchmark/benchmark ./ || exit 1

    # Run tflite parallel split converted models:
    while read line; do
        model_name=${line%;*}
        length=${#model_name}
        input_shapes=${line:length+1}
        if [[ $model_name == \#* || $model_name == "" ]]; then
          continue
        fi
        echo ${model_name} >> "${run_x86_log_file}"
        echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'_1_1_parallel_split.ms --inDataFile='${models_path}'/input_output/input/'${model_name}'.ms.bin --inputShapes='${input_shapes}' --benchmarkDataFile='${models_path}'/input_output/output/'${model_name}'.ms.out' >> "${run_x86_log_file}"
        ./benchmark --modelFile=${ms_models_path}/${model_name}_1_1_parallel_split.ms --inDataFile=${models_path}/input_output/input/${model_name}.ms.bin --inputShapes=${input_shapes} --benchmarkDataFile=${models_path}/input_output/output/${model_name}.ms.out >> "${run_x86_log_file}"
        if [ $? = 0 ]; then
            run_result='x86_Parallel_Split: '${model_name}_1_1_parallel_split' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='x86_Parallel_Split: '${model_name}_1_1_parallel_split' failed'; echo ${run_result} >> ${run_benchmark_result_file}
            if [[ $x86_fail_not_return != "ON" ]]; then
                return 1
            fi
        fi

        echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'_1_2_parallel_split.ms --inDataFile='${models_path}'/input_output/input/'${model_name}'.ms.bin --inputShapes='${input_shapes}' --benchmarkDataFile='${models_path}'/input_output/output/'${model_name}'.ms.out' >> "${run_x86_log_file}"
        ./benchmark --modelFile=${ms_models_path}/${model_name}_1_2_parallel_split.ms --inDataFile=${models_path}/input_output/input/${model_name}.ms.bin --inputShapes=${input_shapes} --benchmarkDataFile=${models_path}/input_output/output/${model_name}.ms.out >> "${run_x86_log_file}"
        if [ $? = 0 ]; then
            run_result='x86_Parallel_Split: '${model_name}_1_2_parallel_split' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='x86_Parallel_Split: '${model_name}_1_2_parallel_split' failed'; echo ${run_result} >> ${run_benchmark_result_file}
            if [[ $x86_fail_not_return != "ON" ]]; then
                return 1
            fi
        fi

        echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'_1_3_parallel_split.ms --inDataFile='${models_path}'/input_output/input/'${model_name}'.ms.bin --inputShapes='${input_shapes}' --benchmarkDataFile='${models_path}'/input_output/output/'${model_name}'.ms.out' >> "${run_x86_log_file}"
        ./benchmark --modelFile=${ms_models_path}/${model_name}_1_3_parallel_split.ms --inDataFile=${models_path}/input_output/input/${model_name}.ms.bin --inputShapes=${input_shapes} --benchmarkDataFile=${models_path}/input_output/output/${model_name}.ms.out >> "${run_x86_log_file}"
        if [ $? = 0 ]; then
            run_result='x86_Parallel_Split: '${model_name}_1_3_parallel_split' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='x86_Parallel_Split: '${model_name}_1_3_parallel_split' failed'; echo ${run_result} >> ${run_benchmark_result_file}
            if [[ $x86_fail_not_return != "ON" ]]; then
                return 1
            fi
        fi
    done < ${models_tflite_parallel_split_config}
}

basepath=$(pwd)
echo ${basepath}

# Example:sh run_benchmark_x86.sh -r /home/temp_test -m /home/temp_test/models -e arm_cpu
while getopts "r:m:e:p:l:" opt; do
    case ${opt} in
        r)
            release_path=${OPTARG}
            echo "release_path is ${OPTARG}"
            ;;
        m)
            models_path=${OPTARG}
            echo "models_path is ${OPTARG}"
            ;;
        e)
            backend=${OPTARG}
            echo "backend is ${OPTARG}"
            ;;
        p)
            x86_fail_not_return=${OPTARG}
            echo "x86_fail_not_return is ${OPTARG}"
            ;;
        l)
            level=${OPTARG}
            echo "level is ${OPTARG}"
            ;;
        ?)
        echo "unknown para"
        exit 1;;
    esac
done

if [[ $backend == "linux_arm64_tflite" ]]; then
  x86_path=${release_path}/linux_aarch64/
else
  x86_path=${release_path}/centos_x86
fi
cd ${x86_path}
file_name=$(ls *-linux-*.tar.gz)
IFS="-" read -r -a file_name_array <<< "$file_name"
version=${file_name_array[2]}
cd -

# Set models config filepath
config_folder="config_level0"
if [[ ${level} == "level1" ]]; then
    config_folder="config_level1"
fi
models_tflite_parallel_split_config=${basepath}/../${config_folder}/models_parallel_split.cfg
models_tflite_config=${basepath}/../${config_folder}/models_tflite.cfg
models_tf_config=${basepath}/../${config_folder}/models_tf.cfg
models_caffe_config=${basepath}/../${config_folder}/models_caffe.cfg
models_tflite_awaretraining_config=${basepath}/../${config_folder}/models_tflite_awaretraining.cfg
models_posttraining_config=${basepath}/../${config_folder}/models_posttraining.cfg
models_onnx_config=${basepath}/../${config_folder}/models_onnx.cfg
models_mindspore_config=${basepath}/../${config_folder}/models_mindspore.cfg
models_weightquant_0bit_config=${basepath}/../${config_folder}/models_weightquant_0bit.cfg
models_weightquant_7bit_config=${basepath}/../${config_folder}/models_weightquant_7bit.cfg
models_weightquant_9bit_config=${basepath}/../${config_folder}/models_weightquant_9bit.cfg
models_weightquant_8bit_config=${basepath}/../${config_folder}/models_weightquant_8bit.cfg
models_weightquant_0bit_auto_tune_config=${basepath}/../${config_folder}/models_weightquant_0bit_auto_tune.cfg
models_weightquant_0bit_auto_tune_cv_config=${basepath}/../${config_folder}/models_weightquant_0bit_auto_tune_cv.cfg
models_weightquant_8bit_debug_config=${basepath}/../${config_folder}/models_weightquant_8bit_debug.cfg
models_process_only_config=${basepath}/../${config_folder}/models_process_only.cfg

# Prepare the config file list
x86_cfg_file_list=()
if [[ $backend == "x86_tflite" || $backend == "x86_avx512_tflite" || $backend == "linux_arm64_tflite" ]]; then
  x86_cfg_file_list=("$models_tflite_config" "$models_tflite_awaretraining_config")
elif [[ $backend == "x86_tf" || $backend == "x86_avx512_tf" ]]; then
  x86_cfg_file_list=("$models_tf_config")
elif [[ $backend == "x86_caffe" || $backend == "x86_avx512_caffe" ]]; then
  x86_cfg_file_list=("$models_caffe_config")
elif [[ $backend == "x86_onnx" || $backend == "x86_avx512_onnx" ]]; then
  x86_cfg_file_list=("$models_onnx_config")
elif [[ $backend == "x86_mindir" || $backend == "x86_avx512_mindir" ]]; then
  x86_cfg_file_list=("$models_mindspore_config")
elif [[ $backend == "x86_quant" ]]; then
  x86_cfg_file_list=("$models_posttraining_config" "$models_weightquant_0bit_config" "$models_weightquant_8bit_config" \
                     "$models_weightquant_7bit_config" "$models_weightquant_0bit_auto_tune_config" \
                     "$models_weightquant_8bit_debug_config" "$models_weightquant_9bit_config" \
                     "$models_process_only_config" "$models_weightquant_0bit_auto_tune_cv_config" )
else
  x86_cfg_file_list=("$models_tf_config" "$models_tflite_config" "$models_caffe_config" "$models_onnx_config" "$models_mindspore_config" \
                     "$models_posttraining_config" "$models_tflite_awaretraining_config" "$models_weightquant_0bit_config" \
                     "$models_weightquant_8bit_config" "$models_weightquant_7bit_config" "$models_weightquant_9bit_config" \
                     "$models_process_only_config")
fi

ms_models_path=${basepath}/ms_models

# Write converter result to temp file
run_converter_log_file=${basepath}/run_converter_log.txt
echo ' ' > ${run_converter_log_file}

run_converter_result_file=${basepath}/run_converter_result.txt
echo ' ' > ${run_converter_result_file}

# Run converter
echo "start Run converter ..."
Run_Converter &
Run_converter_PID=$!
sleep 2

wait ${Run_converter_PID}
Run_converter_status=$?
# Check converter result and return value
if [[ ${Run_converter_status} = 0 ]];then
    echo "Run converter success"
    Print_Converter_Result $run_converter_result_file
else
    echo "Run converter failed"
    cat ${run_converter_log_file}
    Print_Converter_Result $run_converter_result_file
    exit 1
fi
# Empty config file is allowed, but warning message will be shown
if [[ $(Exist_File_In_Path ${ms_models_path} ".ms") != "true" ]]; then
  echo "No ms model found in ${ms_models_path}, please check if config file is empty!"
  exit 0
fi

# Write benchmark result to temp file
run_benchmark_result_file=${basepath}/run_benchmark_result.txt
echo ' ' > ${run_benchmark_result_file}

run_x86_log_file=${basepath}/run_x86_log.txt
echo 'run x86 logs: ' > ${run_x86_log_file}
run_x86_sse_log_file=${basepath}/run_x86_sse_log.txt
echo 'run x86 sse logs: ' > ${run_x86_sse_log_file}
run_x86_avx_log_file=${basepath}/run_x86_avx_log.txt
echo 'run x86 avx logs: ' > ${run_x86_avx_log_file}
run_x86_java_log_file=${basepath}/run_x86_java_log.txt
echo 'run x86 java logs: ' > ${run_x86_java_log_file}
run_x86_parallel_split_log_file=${basepath}/run_x86_parallel_split_log.txt
echo 'run x86 java logs: ' > ${run_x86_parallel_split_log_file}
run_x86_avx512_log_file=${basepath}/run_x86_avx512_log.txt
echo 'run x86 avx512 logs: ' > ${run_x86_avx512_log_file}

backend=${backend:-"all"}
isFailed=0

if [[ $backend == "all" || $backend == "x86-all" || $backend == "x86" || $backend == "x86_onnx" || $backend == "x86_tf" || \
      $backend == "x86_tflite" || $backend == "x86_caffe" || $backend == "x86_mindir" || $backend == "linux_arm64_tflite" ]]; then
    # Run on x86
    echo "start Run x86 $backend..."
    Run_x86 &
    Run_x86_PID=$!
    sleep 1
fi
if [[ $backend == "all" || $backend == "x86-all" || $backend == "x86_avx" || $backend == "x86_onnx" || $backend == "x86_tf" || \
      $backend == "x86_tflite" || $backend == "x86_caffe" || $backend == "x86_mindir" || $backend == "x86_quant" ]]; then
    # Run on x86_avx
    echo "start Run avx $backend..."
    Run_x86_avx &
    Run_x86_avx_PID=$!
    sleep 1
fi
if [[ $backend == "all" || $backend == "x86-all" || $backend == "x86_avx512" || $backend == "x86_avx512_onnx" || $backend == "x86_avx512_tf" || \
      $backend == "x86_avx512_tflite" || $backend == "x86_avx512_caffe" || $backend == "x86_avx512_mindir" ]]; then
    # Run on x86_avx512
    echo "start Run avx512 $backend..."
    Run_x86_avx512 &
    Run_x86_avx512_PID=$!
    sleep 1
fi
if [[ $backend == "all" || $backend == "x86-all" || $backend == "x86_sse" || $backend == "x86_onnx" || $backend == "x86_tf" || \
      $backend == "x86_tflite" || $backend == "x86_caffe" || $backend == "x86_mindir" ]]; then
    # Run on x86_sse
    echo "start Run sse $backend..."
    Run_x86_sse &
    Run_x86_sse_PID=$!
    sleep 1
fi
if [[ $backend == "all" || $backend == "x86-all" || $backend == "x86_java" || $backend == "x86_tflite" ]]; then
    # Run on x86_java
    echo "start Run x86 java ..."
    Run_x86_java &
    Run_x86_java_PID=$!
    sleep 1
fi
if [[ $backend == "all" || $backend == "x86-all" || $backend == "x86_parallel_split" || $backend == "x86_tflite" ]]; then
    # Run on x86_parallel_split
    echo "start Run x86 parallel_split ..."
    Run_x86_parallel_split &
    Run_x86_parallel_split_PID=$!
    sleep 1
fi

if [[ $backend == "all" || $backend == "x86-all" || $backend == "x86" || $backend == "x86_onnx" || $backend == "x86_tf" || \
      $backend == "x86_tflite" || $backend == "x86_caffe" || $backend == "x86_mindir" || $backend == "linux_arm64_tflite" ]]; then
    wait ${Run_x86_PID}
    Run_x86_status=$?
    # Check benchmark result and return value
    if [[ ${Run_x86_status} != 0 ]];then
        echo "Run_x86 failed"
        cat ${run_x86_log_file}
        isFailed=1
    fi
fi
if [[ $backend == "all" || $backend == "x86-all" || $backend == "x86_avx" || $backend == "x86_onnx" || $backend == "x86_tf" || \
      $backend == "x86_tflite" || $backend == "x86_caffe" || $backend == "x86_mindir" || $backend == "x86_quant" ]]; then
    wait ${Run_x86_avx_PID}
    Run_x86_avx_status=$?
    if [[ ${Run_x86_avx_status} != 0 ]];then
        echo "Run_x86 avx failed"
        cat ${run_x86_avx_log_file}
        isFailed=1
    fi
fi
if [[ $backend == "all" || $backend == "x86-all" || $backend == "x86_avx512" || $backend == "x86_avx512_onnx" || $backend == "x86_avx512_tf" || \
      $backend == "x86_avx512_tflite" || $backend == "x86_avx512_caffe" || $backend == "x86_avx512_mindir" ]]; then
    wait ${Run_x86_avx512_PID}
    Run_x86_avx512_status=$?
    if [[ ${Run_x86_avx512_status} != 0 ]];then
        echo "Run_x86 avx512 failed"
        cat ${run_x86_avx512_log_file}
        isFailed=1
    fi
fi
if [[ $backend == "all" || $backend == "x86-all" || $backend == "x86_sse" || $backend == "x86_onnx" || $backend == "x86_tf" || \
      $backend == "x86_tflite" || $backend == "x86_caffe" || $backend == "x86_mindir" ]]; then
    wait ${Run_x86_sse_PID}
    Run_x86_sse_status=$?
    if [[ ${Run_x86_sse_status} != 0 ]];then
        echo "Run_x86 sse failed"
        cat ${run_x86_sse_log_file}
        isFailed=1
    fi
fi
if [[ $backend == "all" || $backend == "x86-all" || $backend == "x86_java" || $backend == "x86_tflite" ]]; then
    wait ${Run_x86_java_PID}
    Run_x86_java_status=$?
    if [[ ${Run_x86_java_status} != 0 ]];then
        echo "Run_x86 java failed"
        cat ${run_x86_java_log_file}
        isFailed=1
    fi
fi
if [[ $backend == "all" || $backend == "x86-all" || $backend == "x86_parallel_split" || $backend == "x86_tflite" ]]; then
    wait ${Run_x86_parallel_split_PID}
    Run_x86_parallel_split_status=$?
    if [[ ${Run_x86_parallel_split_status} != 0 ]];then
        echo "Run_x86 parallel_split failed"
        cat ${run_x86_parallel_split_log_file}
        isFailed=1
    fi
fi

echo "Run_x86 and Run_x86_sse and Run_x86_avx and Run_x86-avx512 is ended"
Print_Benchmark_Result $run_benchmark_result_file
exit ${isFailed}
