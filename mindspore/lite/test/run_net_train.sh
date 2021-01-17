#!/bin/bash

# Run Export on x86 platform and create output test files:
docker_image=mindspore_dev:8
function Run_Export(){
    cd $models_path || exit 1
    if [[ -z "${CLOUD_MODEL_ZOO}" ]]; then
        echo "CLOUD_MODEL_ZOO is not defined - exiting export models"
        exit 1
    fi    
    # Export mindspore train models:
    while read line; do
        model_name=${line}
        if [[ $model_name == \#* ]]; then
          continue
        fi
        echo ${model_name}'_train_export.py' >> "${export_log_file}"
        echo 'exporting' ${model_name}
        echo 'docker run --user '"$(id -u):$(id -g)"' --env CLOUD_MODEL_ZOO=${CLOUD_MODEL_ZOO} -w $PWD --runtime=nvidia -v /home/$USER:/home/$USER -v /opt/share:/opt/share --privileged=true '${docker_image}' python '${models_path}'/'${model_name}'_train_export.py' >>  "${export_log_file}"
        docker run --user "$(id -u):$(id -g)" --env CLOUD_MODEL_ZOO=${CLOUD_MODEL_ZOO} -w $PWD --runtime=nvidia -v /home/$USER:/home/$USER -v /opt/share:/opt/share --privileged=true "${docker_image}" python ${models_path}'/'${model_name}_train_export.py "${epoch_num}"
        if [ $? = 0 ]; then
            export_result='export mindspore '${model_name}'_train_export pass';echo ${export_result} >> ${export_result_file}
        else
            export_result='export mindspore '${model_name}'_train_export failed';echo ${export_result} >> ${export_result_file}
        fi
    done < ${models_mindspore_train_config}
}

# Run converter on x86 platform:
function Run_Converter() {
    # Unzip x86 runtime and convertor
    cd ${x86_path} || exit 1
    tar -zxf mindspore-lite-${version}-train-linux-x64.tar.gz || exit 1

    tar -zxf mindspore-lite-${version}-train-converter-linux-x64.tar.gz || exit 1
    cd ${x86_path}/mindspore-lite-${version}-train-converter-linux-x64 || exit 1
    cp converter/converter_lite ./ || exit 1
    

    # Convert the models
    cd ${x86_path}/mindspore-lite-${version}-train-converter-linux-x64 || exit 1

    rm -rf ${ms_models_path}
    mkdir -p ${ms_models_path}

    # Convert mindspore train models:
    while read line; do
        model_name=${line}
        if [[ $model_name == \#* ]]; then
          continue
        fi
        echo ${model_name}'_train' >> "${run_converter_log_file}"
        echo './converter_lite  --fmk=MINDIR --modelFile='${models_path}'/'${model_name}'_train.mindir --outputFile='${ms_models_path}'/'${model_name}'_train  --trainModel=true' >> "${run_converter_log_file}"
        LD_LIBRARY_PATH=./lib/:./third_party/protobuf/lib:./third_party/flatbuffers/lib:./third_party/glog/lib \
         ./converter_lite  --fmk=MINDIR --modelFile=${models_path}/${model_name}_train.mindir \
         --outputFile=${ms_models_path}/${model_name}'_train' \
         --trainModel=true
        if [ $? = 0 ]; then
            converter_result='converter mindspore '${model_name}'_train pass';echo ${converter_result} >> ${run_converter_result_file}
        else
            converter_result='converter mindspore '${model_name}'_train failed';echo ${converter_result} >> ${run_converter_result_file}
        fi
    done < ${models_mindspore_train_config}
}

# Run on x86 platform:
function Run_x86() {
    # Run mindspore converted train models:
    while read line; do
        model_name=${line}
        if [[ $model_name == \#* ]]; then
          continue
        fi
        
        echo ${model_name}'_train' >> "${run_x86_log_file}"
        echo 'cd  '${x86_path}'/mindspore-lite-'${version}'-train-linux-x64' >> "${run_x86_log_file}"
        cd ${x86_path}/mindspore-lite-${version}-train-linux-x64 || return 1
        echo 'LD_LIBRARY_PATH='${LD_LIBRARY_PATH}':./lib:./third_party/libjpeg-turbo/lib:./third_party/opencv/lib;./benchmark_train/benchmark_train --epochs='${epoch_num}' --modelFile='${ms_models_path}'/'${model_name}'_train.ms --inDataFile='${train_io_path}/${model_name}_input1.bin,${train_io_path}/${model_name}_input2.bin' --expectedDataFile='${train_io_path}'/'${model_name}'_outputs.bin --exportFile='${ms_models_path}'/'${model_name}'_train_exported.ms'  >> "${run_x86_log_file}"
        echo '-------------------------------------------------------------------------------' >> "${run_x86_log_file}"
        LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:./lib:./third_party/libjpeg-turbo/lib:./third_party/opencv/lib \
        ${run_valgrind}./benchmark_train/benchmark_train \
        --modelFile=${ms_models_path}/${model_name}_train.ms \
        --inDataFile=${train_io_path}/${model_name}_input1.bin,${train_io_path}/${model_name}_input2.bin \
        --expectedDataFile=${train_io_path}/${model_name}_outputs.bin \
        --exportFile=${ms_models_path}/${model_name}_train_exported.ms >> "${run_x86_log_file}" \
        --epochs=${epoch_num} --numThreads=${threads}
        if [ $? = 0 ]; then
            run_result='x86: '${model_name}'_train pass'; echo ${run_result} >> ${run_benchmark_train_result_file}
        else
            run_result='x86: '${model_name}'_train failed'; echo ${run_result} >> ${run_benchmark_train_result_file}
        fi
    done < ${models_mindspore_train_config}
}

# Run on arm platform: 
# Gets a parameter - arm64/arm32
function Run_arm() {
    tmp_dir=/data/local/tmp/benchmark_train_test
    if [ "$1" == arm64 ]; then
        arm_path=${arm64_path}
        process_unit="aarch64"
        version_arm=${version_arm64}
        run_arm_log_file=${run_arm64_log_file} 
        adb_cmd_run_file=${adb_cmd_arm64_run_file} 
        adb_push_log_file=${adb_push_arm64_log_file} 
        adb_cmd_file=${adb_cmd_arm64_file}
    elif [ "$1" == arm32 ]; then
        arm_path=${arm32_path}
        process_unit="aarch32"
        version_arm=${version_arm32}
        run_arm_log_file=${run_arm32_log_file} 
        adb_cmd_run_file=${adb_cmd_arm32_run_file} 
        adb_push_log_file=${adb_push_arm32_log_file} 
        adb_cmd_file=${adb_cmd_arm32_file}
    else
        echo 'type ' $1 'is not supported'
        exit 1
    fi

    # Unzip
    cd ${arm_path} || exit 1
    tar -zxf mindspore-lite-${version_arm}-train-android-${process_unit}.tar.gz || exit 1

    # If build with minddata, copy the minddata related libs
    cd ${benchmark_train_test_path} || exit 1
    if [ -f ${arm_path}/mindspore-lite-${version_arm}-train-android-${process_unit}/lib/libminddata-lite.so ]; then
        cp -a ${arm_path}/mindspore-lite-${version_arm}-train-android-${process_unit}/third_party/libjpeg-turbo/lib/libjpeg.so ${benchmark_train_test_path}/libjpeg.so || exit 1
        cp -a ${arm_path}/mindspore-lite-${version_arm}-train-android-${process_unit}/third_party/libjpeg-turbo/lib/libturbojpeg.so ${benchmark_train_test_path}/libturbojpeg.so || exit 1
        cp -a ${arm_path}/mindspore-lite-${version_arm}-train-android-${process_unit}/third_party/opencv/lib/libopencv_core.so ${benchmark_train_test_path}/libopencv_core.so || exit 1
        cp -a ${arm_path}/mindspore-lite-${version_arm}-train-android-${process_unit}/third_party/opencv/lib/libopencv_imgcodecs.so ${benchmark_train_test_path}/libopencv_imgcodecs.so || exit 1
        cp -a ${arm_path}/mindspore-lite-${version_arm}-train-android-${process_unit}/third_party/opencv/lib/libopencv_imgproc.so ${benchmark_train_test_path}/libopencv_imgproc.so || exit 1
        cp -a ${arm_path}/mindspore-lite-${version_arm}-train-android-${process_unit}/lib/libminddata-lite.so ${benchmark_train_test_path}/libminddata-lite.so || exit 1
    fi
    if [ "$1" == arm64 ]; then
        cp -a ${arm_path}/mindspore-lite-${version_arm}-train-android-${process_unit}/third_party/hiai_ddk/lib/libhiai.so ${benchmark_train_test_path}/libhiai.so || exit 1
        cp -a ${arm_path}/mindspore-lite-${version_arm}-train-android-${process_unit}/third_party/hiai_ddk/lib/libhiai_ir.so ${benchmark_train_test_path}/libhiai_ir.so || exit 1
        cp -a ${arm_path}/mindspore-lite-${version_arm}-train-android-${process_unit}/third_party/hiai_ddk/lib/libhiai_ir_build.so ${benchmark_train_test_path}/libhiai_ir_build.so || exit 1
    fi

    cp -a ${arm_path}/mindspore-lite-${version_arm}-train-android-${process_unit}/lib/libmindspore-lite.so ${benchmark_train_test_path}/libmindspore-lite.so || exit 1
#    if [ "$1" == arm64 ]; then
#        cp -a ${arm_path}/mindspore-lite-${version_arm}-runtime-${arm_type}-${process_unit}-train/lib/libmindspore-lite-fp16.so ${benchmark_train_test_path}/libmindspore-lite-fp16.so || exit 1
#        cp -a ${arm_path}/mindspore-lite-${version_arm}-runtime-${arm_type}-${process_unit}-train/lib/libmindspore-lite-optimize.so ${benchmark_train_test_path}/libmindspore-lite-optimize.so || exit 1
#    fi
    cp -a ${arm_path}/mindspore-lite-${version_arm}-train-android-${process_unit}/benchmark_train/benchmark_train ${benchmark_train_test_path}/benchmark_train || exit 1

    # adb push all needed files to the phone
    adb -s ${device_id} push ${benchmark_train_test_path} /data/local/tmp/ > ${adb_push_log_file}

    # run adb ,run session ,check the result:
    echo 'cd  /data/local/tmp/benchmark_train_test' > ${adb_cmd_file}
    echo 'chmod 777 benchmark_train' >> ${adb_cmd_file}

    adb -s ${device_id} shell < ${adb_cmd_file}
    
    # Run mindir converted train models:
    while read line; do
        model_name=${line}
        if [[ $model_name == \#* ]]; then
          continue
        fi

        # run benchmark_train test without clib data
        echo ${model_name}'_train' >> "${run_arm_log_file}"
        adb -s ${device_id} push ${train_io_path}/${model_name}_input*.bin ${train_io_path}/${model_name}_outputs.bin  /data/local/tmp/benchmark_train_test >> ${adb_push_log_file}
        echo 'cd /data/local/tmp/benchmark_train_test' > ${adb_cmd_run_file}
        echo 'chmod 777 benchmark_train' >> ${adb_cmd_run_file}
        if [ "$1" == arm64 ]; then
            echo 'cp  /data/local/tmp/libc++_shared.so ./' >> ${adb_cmd_run_file}
        elif [ "$1" == arm32 ]; then
            echo 'cp  /data/local/tmp/arm32/libc++_shared.so ./' >> ${adb_cmd_run_file}
        fi 
        echo "rm -f ${tmp_dir}/${model_name}_train_exported.ms" >> ${run_arm_log_file}
        echo "rm -f ${tmp_dir}/${model_name}_train_exported.ms" >> ${adb_cmd_run_file}
        adb_cmd=$(cat <<-ENDM
        export LD_LIBRARY_PATH=./:/data/local/tmp/:/data/local/tmp/benchmark_train_test;./benchmark_train \
        --epochs=${epoch_num} \
        --modelFile=${model_name}_train.ms \
        --inDataFile=${tmp_dir}/${model_name}_input1.bin,${tmp_dir}/${model_name}_input2.bin \
        --expectedDataFile=${tmp_dir}/${model_name}_outputs.bin \
        --exportFile=${tmp_dir}/${model_name}_train_exported.ms \
        --numThreads=${threads}
ENDM
        )
        echo "${adb_cmd}" >> ${run_arm_log_file}
        echo "${adb_cmd}" >> ${adb_cmd_run_file}
        adb -s ${device_id} shell < ${adb_cmd_run_file} >> ${run_arm_log_file}
        # TODO: change to arm_type
        if [ $? = 0 ]; then
            run_result=$1': '${model_name}'_train pass'; echo ${run_result} >> ${run_benchmark_train_result_file}
        else
            run_result=$1': '${model_name}'_train failed'; echo ${run_result} >> ${run_benchmark_train_result_file}; return 1
        fi
    done < ${models_mindspore_train_config}
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

function Print_Result() {
    MS_PRINT_TESTCASE_END_MSG
    while read line; do
        arr=("${line}")
        printf "%-15s %-20s %-90s %-7s\n" ${arr[0]} ${arr[1]} ${arr[2]} ${arr[3]}
    done < $1
    MS_PRINT_TESTCASE_END_MSG
}

basepath=$(pwd)
echo ${basepath}

# Example:run_benchmark_train.sh -r /home/emir/Work/TestingEnv/release -m /home/emir/Work/TestingEnv/train_models -i /home/emir/Work/TestingEnv/train_io -d "8KE5T19620002408"
# For running on arm64, use -t to set platform tools path (for using adb commands)
epoch_num=1
threads=1
train_io_path=""
while getopts "r:m:d:i:e:vt:q:" opt; do
    case ${opt} in
        r)
           release_path=${OPTARG}
           echo "release_path is ${OPTARG}"
            ;;
        m)

            models_path=${OPTARG}"/models_train"
            echo "models_path is ${OPTARG}"
            ;;
        i)
            train_io_path=${OPTARG}
            echo "train_io_path is ${OPTARG}"
            ;;
        d)
           device_id=${OPTARG}
            echo "device_id is ${OPTARG}"
            ;;
        e)
            enable_export=${OPTARG}
            echo "enable_export = ${OPTARG}"
            ;;          
        v)
            run_valgrind="valgrind --log-file=valgrind.log "
            echo "Run x86 with valgrind"
            ;;
        q)
           threads=${OPTARG}
           echo "threads=${threads}"
           ;;
        t)
            epoch_num=${OPTARG}
            echo "train epoch num is ${epoch_num}"
            ;;                          
        ?)
            echo "unknown para"
            exit 1;;
    esac
done

if [[ $train_io_path == "" ]]
then
  train_io_path=${models_path}/input_output
fi

arm64_path=${release_path}/android_aarch64
file=$(ls ${arm64_path}/*train-android-aarch64.tar.gz)
file_name="${file##*/}"
IFS="-" read -r -a file_name_array <<< "$file_name"
version_arm64=${file_name_array[2]}

arm32_path=${release_path}/android_aarch32
file=$(ls ${arm32_path}/*train-android-aarch32.tar.gz)
file_name="${file##*/}"
IFS="-" read -r -a file_name_array <<< "$file_name"
version_arm32=${file_name_array[2]}

x86_path=${release_path}/ubuntu_x86
file=$(ls ${x86_path}/*train-linux-x64.tar.gz)
file_name="${file##*/}"
IFS="-" read -r -a file_name_array <<< "$file_name"
version=${file_name_array[2]}

# Set models config filepath
models_mindspore_train_config=${basepath}/models_ms_train.cfg

ms_models_path=${basepath}/ms_models_train

logs_path=${basepath}/logs_train
rm -rf ${logs_path}
mkdir -p ${logs_path}

# Export model if enabled
if [[ $enable_export == 1 ]]; then
    echo "Start Exporting models ..."
    # Write export result to temp file
    export_log_file=${logs_path}/export_log.txt
    echo ' ' > ${export_log_file}

    export_result_file=${logs_path}/export_result.txt
    echo ' ' > ${export_result_file}
    # Run export
    Run_Export
    Print_Result ${export_result_file}

fi    

# Write converter result to temp file
run_converter_log_file=${logs_path}/run_converter_log.txt
echo ' ' > ${run_converter_log_file}

run_converter_result_file=${logs_path}/run_converter_result.txt
echo ' ' > ${run_converter_result_file}

START=$(date +%s.%N)

# Run converter
echo "start run converter ..."
Run_Converter
Run_converter_PID=$!
sleep 1

wait ${Run_converter_PID}
Run_converter_status=$?

# Check converter result and return value
if [[ ${Run_converter_status} = 0 ]];then
    echo "Run converter success"
    Print_Result ${run_converter_result_file}
else
    echo "Run converter failed"
    cat ${run_converter_log_file}
    Print_Result ${run_converter_result_file}
    exit 1
fi


# Write benchmark_train result to temp file
run_benchmark_train_result_file=${logs_path}/run_benchmark_train_result.txt
echo ' ' > ${run_benchmark_train_result_file}

# Create log files
run_x86_log_file=${logs_path}/run_x86_log.txt
echo 'run x86 logs: ' > ${run_x86_log_file}

run_arm64_log_file=${logs_path}/run_arm64_log.txt
echo 'run arm64 logs: ' > ${run_arm64_log_file}
adb_push_arm64_log_file=${logs_path}/adb_push_arm64_log.txt
adb_cmd_arm64_file=${logs_path}/adb_arm64_cmd.txt
adb_cmd_arm64_run_file=${logs_path}/adb_arm64_cmd_run.txt

run_arm32_log_file=${logs_path}/run_arm32_log.txt
echo 'run arm32 logs: ' > ${run_arm32_log_file}
adb_push_arm32_log_file=${logs_path}/adb_push_arm32_log.txt
adb_cmd_arm32_file=${logs_path}/adb_arm32_cmd.txt
adb_cmd_arm32_run_file=${logs_path}/adb_arm32_cmd_run.txt

# Copy the MindSpore models:
echo "Push files to benchmark_train_test folder and run benchmark_train"
benchmark_train_test_path=${basepath}/benchmark_train_test
rm -rf ${benchmark_train_test_path}
mkdir -p ${benchmark_train_test_path}
cp -a ${ms_models_path}/*.ms ${benchmark_train_test_path} || exit 1

# Run on x86
echo "start Run x86 ..."
Run_x86 &
Run_x86_PID=$!
sleep 1


# wait ${Run_x86_PID}
cat ${run_benchmark_train_result_file}
wait ${Run_x86_PID}
Run_x86_status=$?

# Run on arm64
echo "start Run arm64 ..."
Run_arm arm64 
Run_arm64_status=$?
sleep 3

# Run on arm32
echo "start Run arm32 ..."
Run_arm arm32
Run_arm32_status=$?
sleep 1

END=$(date +%s.%N)
DIFF=$(echo "$END - $START" | bc)

function Print_Benchmark_Result() {
    MS_PRINT_TESTCASE_START_MSG
    while read line; do
        arr=("${line}")
        printf "%-20s %-100s %-7s\n" ${arr[0]} ${arr[1]} ${arr[2]}
    done < ${run_benchmark_train_result_file}
    MS_PRINT_TESTCASE_END_MSG
}

# Check benchmark_train result and return value
if [[ ${Run_x86_status} != 0 ]];then
    echo "Run_x86 failed"
    cat ${run_x86_log_file}
    exit 1
fi

if [[ ${Run_arm64_status} != 0 ]];then
    echo "Run_arm64 failed"
    cat ${run_arm64_log_file}
    exit 1
fi

if [[ ${Run_arm32_status} != 0 ]];then
    echo "Run_arm32 failed"
    cat ${run_arm32_log_file}
    exit 1
fi

echo "Test ended - Results:"
Print_Benchmark_Result
echo "Test run Time:" $DIFF
exit 0
