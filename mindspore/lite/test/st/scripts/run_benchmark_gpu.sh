#!/bin/bash

# Run converter on x86 platform:
function Run_Converter() {
    # Unzip x86 runtime and converter
    cd ${x86_path} || exit 1
    tar -zxf mindspore-lite-${version}-inference-linux-x64.tar.gz || exit 1
    cd ${x86_path}/mindspore-lite-${version}-inference-linux-x64/ || exit 1

    cp tools/converter/converter/converter_lite ./ || exit 1
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:./tools/converter/lib/:./tools/converter/third_party/glog/lib

    rm -rf ${ms_models_path}
    mkdir -p ${ms_models_path}

    # Convert gpu models:
    while read line; do
        if [[ $line == \#* ]]; then
          continue
        fi
        model_name=${line%%;*}
        model_type=${model_name##*.}
        case $model_type in
          pb)
            model_fmk="TF"
            ;;
          tflite)
            model_fmk="TFLITE"
            ;;
          onnx)
            model_fmk="ONNX"
            ;;
          mindir)
            model_fmk="MINDIR"
            ;;
          *)
            model_type="caffe"
            model_fmk="CAFFE"
            ;;
        esac
        if [[ $model_fmk == "CAFFE" ]]; then
          echo ${model_name} >> "${run_converter_log_file}"
          echo './converter_lite  --fmk='${model_fmk}' --modelFile='$models_path/${model_name}'.prototxt --weightFile='$models_path'/'${model_name}'.caffemodel --outputFile='${ms_models_path}'/'${model_name} >> "${run_converter_log_file}"
          ./converter_lite  --fmk=${model_fmk} --modelFile=${models_path}/${model_name}.prototxt --weightFile=${models_path}/${model_name}.caffemodel --outputFile=${ms_models_path}/${model_name}
        else
          echo ${model_name} >> "${run_converter_log_file}"
          echo './converter_lite  --fmk='${model_fmk}' --modelFile='${models_path}'/'${model_name}' --outputFile='${ms_models_path}'/'${model_name} >> "${run_converter_log_file}"
          ./converter_lite  --fmk=${model_fmk} --modelFile=${models_path}/${model_name} --outputFile=${ms_models_path}/${model_name}
        fi
        if [ $? = 0 ]; then
            converter_result='converter '${model_type}' '${model_name}' pass';echo ${converter_result} >> ${run_converter_result_file}
        else
            converter_result='converter '${model_type}' '${model_name}' failed';echo ${converter_result} >> ${run_converter_result_file};return 1
        fi
    done < ${models_gpu_fp32_config}

    while read line; do
        if [[ $line == \#* ]]; then
          continue
        fi
        model_name=${line%%;*}
        model_type=${model_name##*.}
        case $model_type in
          pb)
            model_fmk="TF"
            ;;
          tflite)
            model_fmk="TFLITE"
            ;;
          onnx)
            model_fmk="ONNX"
            ;;
          mindir)
            model_fmk="MINDIR"
            ;;
          *)
            model_type="caffe"
            model_fmk="CAFFE"
            ;;
        esac
        if [[ $model_fmk == "CAFFE" ]]; then
          echo ${model_name} >> "${run_converter_log_file}"
          echo './converter_lite  --fmk='${model_fmk}' --modelFile='$models_path/${model_name}'.prototxt --weightFile='$models_path'/'${model_name}'.caffemodel --outputFile='${ms_models_path}'/'${model_name}'_weightquant --quantType=WeightQuant --bitNum=8 --quantWeightChannel=0' >> "${run_converter_log_file}"
          ./converter_lite  --fmk=${model_fmk} --modelFile=${models_path}/${model_name}.prototxt --weightFile=${models_path}/${model_name}.caffemodel --outputFile=${ms_models_path}/${model_name}_weightquant --quantType=WeightQuant --bitNum=8 --quantWeightChannel=0
        else
          echo ${model_name} >> "${run_converter_log_file}"
          echo './converter_lite  --fmk='${model_fmk}' --modelFile='${models_path}'/'${model_name}' --outputFile='${ms_models_path}'/'${model_name}'_weightquant --quantType=WeightQuant --bitNum=8 --quantWeightChannel=0' >> "${run_converter_log_file}"
          ./converter_lite  --fmk=${model_fmk} --modelFile=${models_path}/${model_name} --outputFile=${ms_models_path}/${model_name}_weightquant --quantType=WeightQuant --bitNum=8 --quantWeightChannel=0
        fi
        if [ $? = 0 ]; then
            converter_result='converter gpu weightquant'${model_type}' '${model_name}' pass';echo ${converter_result} >> ${run_converter_result_file}
        else
            converter_result='converter gpu weightquant'${model_type}' '${model_name}' failed';echo ${converter_result} >> ${run_converter_result_file};return 1
        fi
    done < ${models_gpu_weightquant_config}

    while read line; do
      fp16_line_info=${line}
      if [[ $fp16_line_info == \#* ]]; then
        continue
      fi
      model_info=`echo ${fp16_line_info}|awk -F ' ' '{print $1}'`
      model_name=${model_info%%;*}
      echo 'cp '${ms_models_path}'/'${model_name}'.ms' ${ms_models_path}'/'${model_name}'.fp16.ms'
      cp ${ms_models_path}/${model_name}.ms ${ms_models_path}/${model_name}.fp16.ms
      if [ $? = 0 ]; then
          converter_result='converter fp16 '${model_name}' pass';echo ${converter_result} >> ${run_converter_result_file}
      else
          converter_result='converter fp16 '${model_name}' failed';echo ${converter_result} >> ${run_converter_result_file};return 1
      fi
    done < ${models_gpu_fp16_config}
}

# Run on gpu platform:
function Run_gpu() {
    cd ${arm64_path} || exit 1
    tar -zxf mindspore-lite-${version}-inference-android-aarch64.tar.gz || exit 1

    # If build with minddata, copy the minddata related libs
    cd ${benchmark_test_path} || exit 1
    if [ -f ${arm64_path}/mindspore-lite-${version}-inference-android-aarch64/inference/lib/libminddata-lite.so ]; then
        cp -a ${arm64_path}/mindspore-lite-${version}-inference-android-aarch64/inference/lib/libminddata-lite.so ${benchmark_test_path}/libminddata-lite.so || exit 1
    fi
    cp -a ${arm64_path}/mindspore-lite-${version}-inference-android-aarch64/inference/third_party/hiai_ddk/lib/libhiai.so ${benchmark_test_path}/libhiai.so || exit 1
    cp -a ${arm64_path}/mindspore-lite-${version}-inference-android-aarch64/inference/third_party/hiai_ddk/lib/libhiai_ir.so ${benchmark_test_path}/libhiai_ir.so || exit 1
    cp -a ${arm64_path}/mindspore-lite-${version}-inference-android-aarch64/inference/third_party/hiai_ddk/lib/libhiai_ir_build.so ${benchmark_test_path}/libhiai_ir_build.so || exit 1

    cp -a ${arm64_path}/mindspore-lite-${version}-inference-android-aarch64/inference/lib/libmindspore-lite.so ${benchmark_test_path}/libmindspore-lite.so || exit 1
    cp -a ${arm64_path}/mindspore-lite-${version}-inference-android-aarch64/inference/lib/libmslite_kernel_reg.so ${benchmark_test_path}/libmslite_kernel_reg.so || exit 1
    cp -a ${arm64_path}/mindspore-lite-${version}-inference-android-aarch64/tools/benchmark/benchmark ${benchmark_test_path}/benchmark || exit 1

    # adb push all needed files to the phone
    adb -s ${device_id} push ${benchmark_test_path} /data/local/tmp/ > adb_push_log.txt

    # run adb ,run session ,check the result:
    echo 'cd  /data/local/tmp/benchmark_test' > adb_cmd.txt
    echo 'cp  /data/local/tmp/libc++_shared.so ./' >> adb_cmd.txt
    echo 'chmod 777 benchmark' >> adb_cmd.txt

    adb -s ${device_id} shell < adb_cmd.txt

    # Run gpu fp32 converted models:
    while read line; do
        model_name=${line%%;*}
        if [[ $model_name == \#* ]]; then
          continue
        fi
        model_name=`echo ${line} | awk -F ';' '{print $1}'`
        accuracy_limit=`echo ${line} | awk -F ';' '{print $2}'`
        input_num=`echo ${line} | awk -F ';' '{print $3}'`
        input_shapes=`echo ${line} | awk -F ';' '{print $4}'`
        input_files=""
        data_path="/data/local/tmp/input_output/"
        output_file=${data_path}'output/'${model_name}'.ms.out'
        if [[ ${input_num} == "" || ${input_num} == 1 ]]; then
          input_files=/data/local/tmp/input_output/input/${model_name}.ms.bin
        else
          for i in $(seq 1 $input_num)
          do
            input_files=$input_files${data_path}'input/'$model_name'.ms.bin_'$i','
          done
        fi
        if [[ ${accuracy_limit} == "" ]]; then
          accuracy_limit="0.5"
        fi
        echo ${model_name} >> "${run_gpu_log_file}"
        echo 'cd  /data/local/tmp/benchmark_test' > adb_run_cmd.txt
        if [[ $input_shapes == "" ]]; then
          echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --device=GPU --accuracyThreshold='${accuracy_limit}' --modelFile='${model_name}'.ms --inDataFile='${input_files}' --benchmarkDataFile='${output_file} >> "${run_gpu_log_file}"
          echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --device=GPU --accuracyThreshold='${accuracy_limit}' --modelFile='${model_name}'.ms --inDataFile='${input_files}' --benchmarkDataFile='${output_file} >> adb_run_cmd.txt
        else
          echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --inputShapes='${input_shapes}' --accuracyThreshold='${accuracy_limit}' --device=GPU --modelFile='${model_name}'.ms --inDataFile='${input_files}' --benchmarkDataFile='${output_file} >> "${run_gpu_log_file}"
          echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --inputShapes='${input_shapes}' --accuracyThreshold='${accuracy_limit}' --device=GPU --modelFile='${model_name}'.ms --inDataFile='${input_files}' --benchmarkDataFile='${output_file} >> adb_run_cmd.txt
        fi

        adb -s ${device_id} shell < adb_run_cmd.txt >> "${run_gpu_log_file}"
        if [ $? = 0 ]; then
            run_result='arm64_gpu: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='arm64_gpu: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
    done < ${models_gpu_fp32_config}

    # Run GPU fp16 converted models:
    while read line; do
        model_name=${line}
        if [[ $model_name == \#* ]]; then
          continue
        fi
        model_name=`echo ${line} | awk -F ';' '{print $1}'`
        accuracy_limit=`echo ${line} | awk -F ';' '{print $2}'`
        input_num=`echo ${line} | awk -F ';' '{print $3}'`
        input_shapes=`echo ${line} | awk -F ';' '{print $4}'`
        input_files=""
        data_path="/data/local/tmp/input_output/"
        output_file=${data_path}'output/'${model_name}'.ms.out'
        if [[ ${input_num} == "" || ${input_num} == 1 ]]; then
          input_files=/data/local/tmp/input_output/input/${model_name}.ms.bin
        else
          for i in $(seq 1 $input_num)
          do
            input_files=$input_files${data_path}'input/'$model_name'.ms.bin_'$i','
          done
        fi
        if [[ ${accuracy_limit} == "" ]]; then
          accuracy_limit="5"
        fi
        echo ${model_name} >> "${run_gpu_log_file}"
        echo 'cd  /data/local/tmp/benchmark_test' > adb_run_cmd.txt
        if [[ $input_shapes == "" ]]; then
          echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --device=GPU --enableFp16=true --accuracyThreshold='${accuracy_limit}' --modelFile='${model_name}'.ms --inDataFile='${input_files}' --benchmarkDataFile='${output_file} >> "${run_gpu_log_file}"
          echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --device=GPU --enableFp16=true --accuracyThreshold='${accuracy_limit}' --modelFile='${model_name}'.ms --inDataFile='${input_files}' --benchmarkDataFile='${output_file} >> adb_run_cmd.txt
        else
          echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --device=GPU --enableFp16=true --inputShapes='${input_shapes}' --accuracyThreshold='${accuracy_limit}' --modelFile='${model_name}'.ms --inDataFile='${input_files}' --benchmarkDataFile='${output_file} >> "${run_gpu_log_file}"
          echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --device=GPU --enableFp16=true --inputShapes='${input_shapes}' --accuracyThreshold='${accuracy_limit}' --modelFile='${model_name}'.ms --inDataFile='${input_files}' --benchmarkDataFile='${output_file} >> adb_run_cmd.txt
        fi
        adb -s ${device_id} shell < adb_run_cmd.txt >> "${run_gpu_log_file}"
        if [ $? = 0 ]; then
            run_result='arm64_gpu_fp16: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='arm64_gpu_fp16: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
    done < ${models_gpu_fp16_config}

    # Run GPU weightquant converted models:
    while read line; do
        model_name=${line}
        if [[ $model_name == \#* ]]; then
          continue
        fi
        model_name=`echo ${line} | awk -F ';' '{print $1}'`
        accuracy_limit=`echo ${line} | awk -F ';' '{print $2}'`
        input_num=`echo ${line} | awk -F ';' '{print $3}'`
        input_shapes=`echo ${line} | awk -F ';' '{print $4}'`
        input_files=""
        data_path="/data/local/tmp/input_output/"
        output_file=${data_path}'output/'${model_name}'.ms.out'
        if [[ ${input_num} == "" || ${input_num} == 1 ]]; then
          input_files=/data/local/tmp/input_output/input/${model_name}.ms.bin
        else
          for i in $(seq 1 $input_num)
          do
            input_files=$input_files${data_path}'input/'$model_name'.ms.bin_'$i','
          done
        fi
        if [[ ${accuracy_limit} == "" ]]; then
          accuracy_limit="5"
        fi
        echo ${model_name} >> "${run_gpu_log_file}"
        echo 'cd  /data/local/tmp/benchmark_test' > adb_run_cmd.txt
        if [[ $input_shapes == "" ]]; then
          echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --device=GPU --enableFp16=true --accuracyThreshold='${accuracy_limit}' --modelFile='${model_name}'_weightquant.ms --inDataFile='${input_files}' --benchmarkDataFile='${output_file} >> "${run_gpu_log_file}"
          echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --device=GPU --enableFp16=true --accuracyThreshold='${accuracy_limit}' --modelFile='${model_name}'_weightquant.ms --inDataFile='${input_files}' --benchmarkDataFile='${output_file} >> adb_run_cmd.txt
        else
          echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --device=GPU --enableFp16=true --inputShapes='${input_shapes}' --accuracyThreshold='${accuracy_limit}' --modelFile='${model_name}'_weightquant.ms --inDataFile='${input_files}' --benchmarkDataFile='${output_file} >> "${run_gpu_log_file}"
          echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --device=GPU --enableFp16=true --inputShapes='${input_shapes}' --accuracyThreshold='${accuracy_limit}' --modelFile='${model_name}'_weightquant.ms --inDataFile='${input_files}' --benchmarkDataFile='${output_file} >> adb_run_cmd.txt
        fi
        adb -s ${device_id} shell < adb_run_cmd.txt >> "${run_gpu_log_file}"
        if [ $? = 0 ]; then
            run_result='arm64_gpu_weightquant: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='arm64_gpu_weightquant: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
    done < ${models_gpu_weightquant_config}
}

# Print start msg before run testcase
function MS_PRINT_TESTCASE_START_MSG() {
    echo ""
    echo -e "-----------------------------------------------------------------------------------------------------------------------------------"
    echo -e "env                  Testcase                                                                                             Result   "
    echo -e "---                  --------                                                                                             ------   "
}

# Print start msg after run testcase
function MS_PRINT_TESTCASE_END_MSG() {
    echo -e "-----------------------------------------------------------------------------------------------------------------------------------"
}

function Print_Converter_Result() {
    MS_PRINT_TESTCASE_END_MSG
    while read line; do
        arr=("${line}")
        printf "%-15s %-20s %-90s %-7s\n" ${arr[0]} ${arr[1]} ${arr[2]} ${arr[3]}
    done < ${run_converter_result_file}
    MS_PRINT_TESTCASE_END_MSG
}

function Print_Benchmark_Result() {
    MS_PRINT_TESTCASE_START_MSG
    while read line; do
        arr=("${line}")
        printf "%-20s %-100s %-7s\n" ${arr[0]} ${arr[1]} ${arr[2]}
    done < ${run_benchmark_result_file}
    MS_PRINT_TESTCASE_END_MSG
}

basepath=$(pwd)
echo ${basepath}
#set -e

# Example:sh run_benchmark_gpu.sh -r /home/temp_test -m /home/temp_test/models -d "8KE5T19620002408" -e arm_cpu
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

# mkdir train

x86_path=${release_path}/ubuntu_x86
file_name=$(ls ${x86_path}/*inference-linux-x64.tar.gz)
IFS="-" read -r -a file_name_array <<< "$file_name"
version=${file_name_array[2]}

# Set models config filepath
models_gpu_fp32_config=${basepath}/../config/models_gpu_fp32.cfg
models_gpu_fp16_config=${basepath}/../config/models_gpu_fp16.cfg
models_gpu_weightquant_config=${basepath}/../config/models_gpu_weightquant.cfg

ms_models_path=${basepath}/ms_models

# Write converter result to temp file
run_converter_log_file=${basepath}/run_converter_log.txt
echo ' ' > ${run_converter_log_file}

run_converter_result_file=${basepath}/run_converter_result.txt
echo ' ' > ${run_converter_result_file}

# Run converter
echo "start Run converter ..."
Run_Converter
Run_converter_PID=$!
sleep 1

wait ${Run_converter_PID}
Run_converter_status=$?

# Check converter result and return value
if [[ ${Run_converter_status} = 0 ]];then
    echo "Run converter success"
    Print_Converter_Result
else
    echo "Run converter failed"
    cat ${run_converter_log_file}
    Print_Converter_Result
    exit 1
fi

# Write benchmark result to temp file
run_benchmark_result_file=${basepath}/run_benchmark_result.txt
echo ' ' > ${run_benchmark_result_file}

run_gpu_log_file=${basepath}/run_gpu_log.txt
echo 'run gpu logs: ' > ${run_gpu_log_file}

# Copy the MindSpore models:
echo "Push files to the arm and run benchmark"
benchmark_test_path=${basepath}/benchmark_test
rm -rf ${benchmark_test_path}
mkdir -p ${benchmark_test_path}
cp -a ${ms_models_path}/*.ms ${benchmark_test_path} || exit 1
# Copy models converted using old release of mslite converter for compatibility test
cp -a ${models_path}/compatibility_test/*.ms ${benchmark_test_path} || exit 1

backend=${backend:-"all"}
isFailed=0

if [[ $backend == "all" || $backend == "gpu" ]]; then
    # Run on gpu
    arm64_path=${release_path}/android_aarch64
    # mv ${arm64_path}/*train-android-aarch64* ./train
    file_name=$(ls ${arm64_path}/*inference-android-aarch64.tar.gz)
    IFS="-" read -r -a file_name_array <<< "$file_name"
    version=${file_name_array[2]}

    echo "start Run gpu ..."
    Run_gpu
    Run_gpu_status=$?
    sleep 1
fi
if [[ $backend == "all" || $backend == "gpu" ]]; then
    if [[ ${Run_gpu_status} != 0 ]];then
        echo "Run_gpu failed"
        cat ${run_gpu_log_file}
        isFailed=1
    fi
fi

echo "Run_gpu is ended"
Print_Benchmark_Result
if [[ $isFailed == 1 ]]; then
    exit 1
fi
exit 0
