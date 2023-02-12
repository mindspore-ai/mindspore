#!/bin/bash
source ./scripts/base_functions.sh
version=1.6.1

# Run Export on x86 platform and create output test files:
docker_image=mindspore_build:210301
function Run_Export(){
    cd $models_path || exit 1
    if [[ -z "${CLOUD_MODEL_ZOO}" ]]; then
        echo "CLOUD_MODEL_ZOO is not defined - exiting export models"
        exit 1
    fi
    # Export mindspore train models:
    while read line; do
        LFS=" " read -r -a line_array <<< ${line}
        model_name=${line_array[0]}
        if [[ $model_name == \#* || $model_name == "" ]]; then
          continue
        fi
        echo ${model_name}'_train_export.py' >> "${export_log_file}"
        echo 'exporting' ${model_name}
        if [ -n "$docker_image" ]; then
          echo 'docker run --user '"$(id -u):$(id -g)"' --env CLOUD_MODEL_ZOO=${CLOUD_MODEL_ZOO} -w $PWD --runtime=nvidia -v /home/$USER:/home/$USER -v /opt/share:/opt/share --privileged=true '${docker_image}' python '${models_path}'/'${model_name}'_train_export.py' >>  "${export_log_file}"
          docker run --user "$(id -u):$(id -g)" --env CLOUD_MODEL_ZOO=${CLOUD_MODEL_ZOO} -w $PWD --runtime=nvidia -v /home/$USER:/home/$USER -v /opt/share:/opt/share --privileged=true "${docker_image}" python ${models_path}'/'${model_name}_train_export.py "${epoch_num}"
        else
          echo 'CLOUD_MODEL_ZOO=${CLOUD_MODEL_ZOO} python '${models_path}'/'${model_name}'_train_export.py' >>  "${export_log_file}"
          CLOUD_MODEL_ZOO=${CLOUD_MODEL_ZOO} python ${models_path}'/'${model_name}_train_export.py "${epoch_num}"
        fi
        if [ $? = 0 ]; then
            export_result='export mindspore '${model_name}'_train_export pass';echo ${export_result} >> ${export_result_file}
        else
            export_result='export mindspore '${model_name}'_train_export failed';echo ${export_result} >> ${export_result_file}
        fi
    done < ${models_ms_train_config}
}

# Run converter on x86 platform:
function Run_Converter() {
    # $1:cfgFileList
    cd ${x86_path} || exit 1
    tar -zxf mindspore-lite-${version}-linux-x64.tar.gz || exit 1
    cd ${x86_path}/mindspore-lite-${version}-linux-x64/ || exit 1

    cp tools/converter/converter/converter_lite ./ || exit 1
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:./tools/converter/lib/:./tools/converter/third_party/glog/lib

    rm -rf ${ms_models_path}
    mkdir -p ${ms_models_path}

    cur_timestamp=$((`date '+%s'`*1000+10#`date '+%N'`/1000000))
    fail_status_file=${cur_timestamp}.log
    echo '0' > ${fail_status_file}

    # parallel processing
    fifo_file="fifo_file.txt"
    mkfifo ${fifo_file}
    exec 6<>${fifo_file}
    rm -f ${fifo_file}
    max_parallel_jobs=6
    for ((i = 0; i < ${max_parallel_jobs}; i++)); do echo; done >&6

    # Convert mindspore train models:
    while read line; do
        LFS=" " read -r -a line_array <<< ${line}
        if [[ ${line_array[0]} == \#* || ${line_array[0]} == "" ]]; then
          continue
        fi
        local model_prefix=${line_array[0]}_train
        parse_line convert
        if [[ "$?" == "1" ]]; then continue; fi
        if [[ "${enable_transfer}" == "1" ]]; then
            model_prefix="${line_array[0]}_head"
            model_name=${line_array[0]}'_head'
        fi
        if [[ "${expression}" == "1" ]]; then
            model_prefix="${model_name}_fwd"
        fi
        if [[ ${check_convert} == "1" ]]; then
            ms_file=$ms_models_path'/'$model_name'.ms'
            if [ -f "$ms_file" ]; then
                echo $model_name'.ms already exist, continue without convert'
                continue
            fi
        fi
        read -u6
        {
          echo ${model_name} >> "${run_converter_log_file}"
          echo './converter_lite  --fmk=MINDIR --modelFile='${models_path}'/'${model_prefix}'.mindir --outputFile='${ms_models_path}'/'${model_name}' --trainModel=true' ${WEIGHT_QUANT} >> "${run_converter_log_file}"
          ./converter_lite --fmk=MINDIR --modelFile=${models_path}/${model_prefix}.mindir --outputFile=${ms_models_path}/${model_name} --trainModel=true ${no_opt} ${WEIGHT_QUANT}
          if [ $? = 0 ]; then
              converter_result='converter mindspore '${model_name}' pass';echo ${converter_result} >> ${run_converter_result_file}
          else
              converter_result='converter mindspore '${model_name}' failed';echo ${converter_result} >> ${run_converter_result_file}
              echo '1' > ${fail_status_file}
          fi
          # If Transfer sesstion convert backbone model
          if [[ "${enable_transfer}" == "1" ]]; then
              model_prefix=${line_array[0]}'_bb'
              model_name=${line_array[0]}'_bb'
              echo ${model_name} >> "${run_converter_log_file}"
              echo './converter_lite  --fmk=MINDIR --modelFile='${models_path}'/'${model_name}'.mindir --outputFile='${ms_models_path}'/'${model_name} ${WEIGHT_QUANT} >> "${run_converter_log_file}"
              ./converter_lite --fmk=MINDIR --modelFile=${models_path}/${model_prefix}.mindir --outputFile=${ms_models_path}/${model_name} ${WEIGHT_QUANT}
              if [ $? = 0 ]; then
                  converter_result='converter mindspore '${model_name}' pass';echo ${converter_result} >> ${run_converter_result_file}
              else
                  converter_result='converter mindspore '${model_name}' failed';echo ${converter_result} >> ${run_converter_result_file}
                  echo '1' > ${fail_status_file}
              fi
          fi
          echo >&6
        } &
    done < ${models_ms_train_config}
    wait
    exec 6>&-
    read fail_status < ${fail_status_file}
    return ${fail_status}
}

function should_run_example() {
  ret=0
  while read line; do
    LFS=" " read -r -a line_array <<< ${line}
    model_name=${line_array[0]}
    if [[ $model_name == \#* || $model_name == "" ]]; then
      continue
    fi
    if [[ $model_name == "$1" ]]; then
      if [[ "${line_array[1]}" == "code_example" ]]; then
        ret=1
      fi
      if [[ "${line_array[2]}" == "expression" ]]; then
        ret=2
      fi
    fi
  done < ${models_ms_train_config}
  return $ret
}

function GenerateWeightQuantConfig() {
  echo "[common_quant_param]" > $4
  echo "quant_type=WEIGHT_QUANT" >> $4
  echo "bit_num=$1" >> $4
  echo "min_quant_weight_size=$2" >> $4
  echo "min_quant_weight_channel=$3" >> $4
}

function parse_line() {
    i=1
    loss_name=
    inputShapes=
    enable_fp16="false"
    virtual_batch="false"
    accuracy_limit=0.5
    ret=0
    WEIGHT_QUANT=""
    enable_transfer=0
    suffix_print=""
    check_convert=0
    do_api=false
    no_opt="--NoFusion=false"
    expression=0
    model_name=${line_array[0]}_train
    while [[ $i < ${#line_array[@]} ]] ; do
        case ${line_array[i]} in
           "weight_quant")
            model_name="${line_array[0]}_train_quant"
            weight_quant_config="${model_name}.cfg"
            GenerateWeightQuantConfig 8 0 0 ${weight_quant_config}
            WEIGHT_QUANT="--configFile=${weight_quant_config}"
            ;;
          "vb")
            virtual_batch="true"
            suffix_print="_virtual_batch"
            check_convert=1
            ;;
          "loss_name")
            i=$(($i+1))
            loss_name=${line_array[i]}
            ;;
          "noarm32")
            if [[ $1 == "arm32" ]]; then
               run_result=$1': '${model_name}' irrelevant'; echo ${run_result} >> ${run_benchmark_train_result_file}
               ret=1
            fi
            ;;
          "transfer")
             enable_transfer=1
             ;;
          "fp16")
             if [[ "$1" == "arm64" ]]; then
                enable_fp16="true"
                suffix_print="_fp16"
             elif [[ "$1" != "convert" ]]; then
               ret=1
             fi
             check_convert=1
            ;;
          "code_example")
             ret=1
             ;;
          "inputShapes")
            i=$(($i+1))
            inputShapes=${line_array[i]}
            ;;
          "expression")
            model_name="${line_array[0]}_expr"
            do_api=true
            no_opt="--NoFusion=true"
            expression=1
            ;;
          *)
            check=`echo "${line_array[i]}" | grep -E '^\-?[0-9]*\.?[0-9]+$'`
            if [ "${check}" != "" ] ; then
                accuracy_limit=${line_array[i]}
            fi
            ;;
        esac
        i=$(($i+1))
    done
    return $ret
}

# Run on x86 platform:
function Run_x86() {
    # $1:cfgFileList;
    cd ${x86_path}/mindspore-lite-${version}-linux-x64 || return 1
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${x86_path}/mindspore-lite-${version}-linux-x64/runtime/lib:${x86_path}/mindspore-lite-${version}-linux-x64/runtime/third_party/libjpeg-turbo/lib
    # Run mindspore converted train models:
    local fail=0
    while read line; do
        LFS=" " read -r -a line_array <<< ${line}
        if [[ ${line_array[0]} == \#* || ${line_array[0]} == "" ]]; then
          continue
        fi
        local model_prefix=${line_array[0]}
        local bb_model_file=""
        local log_suffix="_train"
        parse_line x86
        if [[ "$?" == "1" ]]; then continue; fi
        if [[ "${expression}" == "1" ]]; then
            model_prefix=${model_prefix}_expr
        fi
        local model_file="${ms_models_path}/${model_name}.ms"
        local inference_file="${ms_models_path}/${model_name}_infer"
        if [[ "${enable_transfer}" == "1" ]]; then
            model_file="${ms_models_path}/${model_prefix}_head.ms"
            bb_model_file="${ms_models_path}/${model_prefix}_bb.ms"
            suffix_print="_transfer"
            log_suffix="_transfer"
            export_file="${ms_models_path}/${model_prefix}_tod_head"
            inference_file="${ms_models_path}/${model_prefix}_infer"
        fi
        if [ ! -z "$inference_file" ]; then
            rm -f ${inference_file}"*"
        fi
        if [ ! -z "$export_file" ]; then
            rm -f ${export_file}"*"
        fi
        # start running benchmark
        local run_result
        echo ${model_name} >> ${run_x86_log_file}
        echo "${run_valgrind} ./tools/benchmark_train/benchmark_train \
            --modelFile=${model_file} \
            --bbModelFile=${bb_model_file} \
            --inDataFile=${train_io_path}/${model_prefix}_input \
            --expectedDataFile=${train_io_path}/${model_prefix}_output --epochs=${epoch_num} --numThreads=${threads} \
            --accuracyThreshold=${accuracy_limit} --inferenceFile=${inference_file} \
            --exportFile=${export_file} \
            --virtualBatch=${virtual_batch} \
            --inputShapes=${inputShapes} \
            --lossName=${loss_name}  \
            --unifiedApi=${do_api}" >> ${run_x86_log_file}
        ${run_valgrind} ./tools/benchmark_train/benchmark_train \
            --modelFile=${model_file} \
            --bbModelFile=${bb_model_file} \
            --inDataFile=${train_io_path}/${model_prefix}_input \
            --expectedDataFile=${train_io_path}/${model_prefix}_output --epochs=${epoch_num} --numThreads=${threads} \
            --accuracyThreshold=${accuracy_limit} --inferenceFile=${inference_file} \
            --exportFile=${export_file} \
            --virtualBatch=${virtual_batch} \
            --inputShapes=${inputShapes} \
            --lossName=${loss_name} \
            --unifiedApi=${do_api} >> "${run_x86_log_file}"
        if [ $? = 0 ]; then
            run_result='x86'${log_suffix}': '${model_name}''${suffix_print}' pass'; echo ${run_result} >> ${run_benchmark_train_result_file}
        else
            run_result='x86'${log_suffix}': '${model_name}''${suffix_print}' failed'; echo ${run_result} >> ${run_benchmark_train_result_file}
            fail=1
        fi
    done < ${models_ms_train_config}
    return ${fail}
}

# Run on arm platform:
# Gets two parameters
function Run_arm() {
    # $1:platform(arm64/arm32)
    local arm_path process_unit version_arm run_arm_log_file adb_cmd_run_file adb_push_log_file adb_cmd_file adb_cmd
    if [ "$1" == "arm64" ]; then
        arm_path=${arm64_path}
        process_unit="aarch64"
        version_arm=${version_arm64}
        run_arm_log_file=${run_arm64_log_file}
        adb_cmd_run_file=${adb_cmd_arm64_run_file}
        adb_push_log_file=${adb_push_arm64_log_file}
        adb_cmd_file=${adb_cmd_arm64_file}
    elif [ "$1" == "arm32" ]; then
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

    # Copy the MindSpore models:
    echo "Copy files to benchmark_train_test_arm32/64 folder, push them to the phone and run benchmark_train"
    benchmark_train_test_path="${basepath}/benchmark_train_test_$1"
    rm -rf ${benchmark_train_test_path}
    mkdir -p ${benchmark_train_test_path}
    cp -a ${ms_models_path}/*.ms ${benchmark_train_test_path} || exit 1

    # Unzip
    cd ${arm_path} || exit 1
    tar -zxf mindspore-lite-${version_arm}-android-${process_unit}.tar.gz || exit 1

    # If build with minddata, copy the minddata related libs
    cd ${benchmark_train_test_path} || exit 1
    if [ -f ${arm_path}/mindspore-lite-${version_arm}-android-${process_unit}/runtime/lib/libminddata-lite.so ]; then
        cp -a ${arm_path}/mindspore-lite-${version_arm}-android-${process_unit}/runtime/third_party/libjpeg-turbo/lib/libjpeg.so* ${benchmark_train_test_path}/ || exit 1
        cp -a ${arm_path}/mindspore-lite-${version_arm}-android-${process_unit}/runtime/third_party/libjpeg-turbo/lib/libturbojpeg.so* ${benchmark_train_test_path}/ || exit 1
        cp -a ${arm_path}/mindspore-lite-${version_arm}-android-${process_unit}/runtime/lib/libminddata-lite.so ${benchmark_train_test_path}/libminddata-lite.so || exit 1
    fi
    cp -a ${arm_path}/mindspore-lite-${version_arm}-android-${process_unit}/runtime/third_party/hiai_ddk/lib/libhiai.so ${benchmark_train_test_path}/libhiai.so || exit 1
    cp -a ${arm_path}/mindspore-lite-${version_arm}-android-${process_unit}/runtime/third_party/hiai_ddk/lib/libhiai_ir.so ${benchmark_train_test_path}/libhiai_ir.so || exit 1
    cp -a ${arm_path}/mindspore-lite-${version_arm}-android-${process_unit}/runtime/third_party/hiai_ddk/lib/libhiai_ir_build.so ${benchmark_train_test_path}/libhiai_ir_build.so || exit 1
    cp -a ${arm_path}/mindspore-lite-${version_arm}-android-${process_unit}/runtime/lib/libmindspore-lite.so ${benchmark_train_test_path}/libmindspore-lite.so || exit 1
    cp -a ${arm_path}/mindspore-lite-${version_arm}-android-${process_unit}/runtime/lib/libmindspore-lite-train.so ${benchmark_train_test_path}/libmindspore-lite-train.so || exit 1
    cp -a ${arm_path}/mindspore-lite-${version_arm}-android-${process_unit}/tools/benchmark_train/benchmark_train ${benchmark_train_test_path}/benchmark_train || exit 1

    # adb push all needed files to the phone
    adb -s ${device_id} push ${benchmark_train_test_path} /data/local/tmp/ > ${adb_push_log_file}

    # run adb ,run session ,check the result:
    tmp_dir="/data/local/tmp/benchmark_train_test_$1"
    echo "cd ${tmp_dir}" > ${adb_cmd_file}
    echo 'chmod 777 benchmark_train' >> ${adb_cmd_file}
    adb -s ${device_id} shell < ${adb_cmd_file}
    local fail=0
    # Run mindir converted train models:
    while read line; do
        local line_array
        local bb_model_file=""
        LFS=" " read -r -a line_array <<< ${line}
        if [[ ${line_array[0]} == \#* || ${line_array[0]} == "" ]]; then
          continue
        fi
        parse_line $1
        if [[ "$?" == "1" ]]; then continue; fi
        local model_prefix=${line_array[0]}
        if [[ ${expression} == "1" ]]; then
          model_prefix=${model_prefix}_expr
        fi
        local run_result=""
        local log_suffix="_train"
        if [[ "$?" == "1" ]]; then continue; fi
        local export_file="${tmp_dir}/${model_name}_tod"
        local inference_file="${tmp_dir}/${model_name}_infer"
        local model_file="${model_name}.ms"
        if [[ "${enable_transfer}" == "1" ]]; then
            model_file="${model_prefix}_head.ms"
            bb_model_file="${model_prefix}_bb.ms"
            suffix_print="_transfer"
            export_file="${tmp_dir}/${model_prefix}_tod_head"
            inference_file="${tmp_dir}/${model_prefix}_infer"
            log_suffix="_transfer"
        fi
        # run benchmark_train test without clib data
        echo ${model_name} >> "${run_arm_log_file}"
        adb -s ${device_id} push ${train_io_path}/${model_prefix}_input*.bin ${train_io_path}/${model_prefix}_output*.bin  ${tmp_dir}  >> ${adb_push_log_file}
        echo "cd ${tmp_dir}" > ${adb_cmd_run_file}
        echo 'chmod 777 benchmark_train' >> ${adb_cmd_run_file}
        if [ "$1" == arm64 ]; then
            echo 'cp  /data/local/tmp/libc++_shared.so ./' >> ${adb_cmd_run_file}
        elif [ "$1" == arm32 ]; then
            echo 'cp  /data/local/tmp/arm32/libc++_shared.so ./' >> ${adb_cmd_run_file}
        fi
        adb -s ${device_id} shell < ${adb_cmd_run_file} >> ${run_arm_log_file}
        echo "$export_file"
        if [ ! -z "$export_file" ]; then
            echo "rm -f ${export_file}*" >> ${run_arm_log_file}
            echo "rm -f ${export_file}*" >> ${adb_cmd_run_file}
        fi
        if [ ! -z "$inference_file" ]; then
         echo "rm -f ${inference_file}*" >> ${run_arm_log_file}
            echo "rm -f ${inference_file}*" >> ${adb_cmd_run_file}
        fi
        adb -s ${device_id} shell < ${adb_cmd_run_file} >> ${run_arm_log_file}
        adb_cmd=$(cat <<-ENDM
        export LD_LIBRARY_PATH=./:/data/local/tmp/:${tmp_dir};./benchmark_train \
        --epochs=${epoch_num} \
        --modelFile=${model_file} \
        --bbModelFile=${bb_model_file} \
        --inDataFile=${tmp_dir}/${model_prefix}_input \
        --expectedDataFile=${tmp_dir}/${model_prefix}_output \
        --numThreads=${threads} \
        --accuracyThreshold=${accuracy_limit} \
        --enableFp16=${enable_fp16} \
        --inferenceFile=${inference_file} \
        --exportFile=${export_file} \
        --virtualBatch=${virtual_batch} \
        --inputShapes=${inputShapes} \
        --lossName=${loss_name} \
        --unifiedApi=${do_api}
ENDM
)
        echo "${adb_cmd}" >> ${run_arm_log_file}
        echo "${adb_cmd}" >> ${adb_cmd_run_file}
        adb -s ${device_id} shell < ${adb_cmd_run_file} >> ${run_arm_log_file}
        # TODO: change to arm_type
        if [ $? = 0 ]; then
            run_result=$1${log_suffix}': '${model_name}''${suffix_print}' pass'; echo ${run_result} >> ${run_benchmark_train_result_file}
        else
            run_result=$1${log_suffix}': '${model_name}''${suffix_print}' failed'; echo ${run_result} >> ${run_benchmark_train_result_file};
            fail=1
        fi
    done < ${models_ms_train_config}
    adb -s ${device_id} shell "rm -rf ${tmp_dir}"
    return ${fail}
}

function Run_CodeExamples() {
    fail=0
    target="x86"
    tarball_path=${x86_path}/mindspore-lite-${version}-linux-x64.tar.gz
    if [[ $backend == "arm64_train" ]]; then
      target="arm64"
      tarball_path=${arm64_path}/mindspore-lite-${version_arm64}-android-aarch64.tar.gz
      export ANDROID_SERIAL=${device_id}
    fi
    should_run_example "train_lenet_java"
    should_run=$?

    export PATH=${x86_path}/mindspore-lite-${version}-linux-x64/tools/converter/converter/:$PATH
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${x86_path}/mindspore-lite-${version}-linux-x64/tools/converter/lib/:${x86_path}/mindspore-lite-${version}-linux-x64/tools/converter/third_party/glog/lib

    if [[ "$should_run" == "1" && ($backend == "all" || $backend == "codegen_and_train"  || $backend == "x86-all" || $backend == "x86_train" || $backend == "x86-java") ]]; then
      cd ${basepath}/../../examples/train_lenet_java || exit 1
      chmod 777 ./prepare_and_run.sh
      ./prepare_and_run.sh -D ${datasets_path}/mnist -r ${tarball_path} -m ${models_path}/code_example.mindir >> ${run_code_examples_log_file}
      if [ "$?" != "0" ]; then
          echo "train_lenet_java prepare_and_run.sh failed"
          exit 1
      fi
      accurate=$(tail -10 ${run_code_examples_log_file} | awk -F= 'NF==2 && /accuracy/ { sum += $2} END { print (sum > 0.80) }')
      if [ $accurate -eq 1 ]; then
        echo "Lenet Java Trained  and reached accuracy" >> ${run_code_examples_log_file}
      else
        echo "train lenet java demo failure" >> ${run_code_examples_log_file}
        fail=1
      fi
      cd -
    fi

    if [[ $backend == "all" || $backend == "train" || $backend == "x86_train" || $backend == "codegen_and_train" || $backend == "arm64_train" ]]; then

      should_run_example "unified_api"
      should_run=$?
      local expression_flag=""
      if [[ "$should_run" == "2" ]]; then
        expression_flag="-x"
        should_run=1
      fi
      if [[ "$should_run" == "1" ]]; then
        cd ${basepath}/../../examples/train_lenet_cpp || exit 1
        chmod 777 ./prepare_and_run.sh
        chmod 777 ./*/*.sh
        ./prepare_and_run.sh -D ${datasets_path}/mnist -r ${tarball_path} -t ${target} -m ${models_path}/code_example.mindir -e 1 ${expression_flag} >> ${run_code_examples_log_file}
        if [ "$?" != "0" ]; then
          echo "Unified API prepare_and_run.sh failed"
          exit 1
        fi
        accurate=$(tail -20 ${run_code_examples_log_file} | awk 'NF==3 && /Accuracy is/ { sum += $3} END { print (sum > 1.6) }')
        if [ $accurate -eq 1 ]; then
          echo "Unified API Trained and reached accuracy" >> ${run_code_examples_log_file}
          echo 'code_examples: train_lenet_cpp pass' >> ${run_benchmark_train_result_file}
        else
          echo "Unified API demo failure" >> ${run_code_examples_log_file}
          echo 'code_examples: train_lenet_cpp failed' >> ${run_benchmark_train_result_file}
          fail=1
        fi
        rm -rf package*/dataset
        cd -
      fi
    fi
    return ${fail}
}

function Print_Result() {
    MS_PRINT_TESTCASE_END_MSG
    while read line; do
        arr=("${line}")
        printf "%-15s %-20s %-90s %-7s\n" ${arr[0]} ${arr[1]} ${arr[2]} ${arr[3]}
    done < $1
    MS_PRINT_TESTCASE_END_MSG
}

# Example:run_benchmark_train.sh -r /home/emir/Work/TestingEnv/release -m /home/emir/Work/TestingEnv/train_models -i /home/emir/Work/TestingEnv/train_io -d "8KE5T19620002408"
# For running on arm64, use -t to set platform tools path (for using adb commands)
epoch_num=1
threads=2
train_io_path=""
while getopts "r:c:m:d:i:e:vt:q:D:M:l:" opt; do
    case ${opt} in
        r)
           release_path=${OPTARG}
           echo "release_path is ${OPTARG}"
            ;;
        m)
            models_path=${OPTARG}
            echo "models_path is ${OPTARG}"
            ;;
         c)
           models_ms_train_config=${OPTARG}
           echo  "models_ms_train_config ${models_ms_train_config}"
           ;;
        i)
            train_io_path=${OPTARG}
            echo "train_io_path is ${OPTARG}"
            ;;
        d)
           device_id=${OPTARG}
            echo "device_id is ${OPTARG}"
            ;;
        D)
            enable_export=1
            docker_image=${OPTARG}
            echo "enable_export = 1, docker_image = ${OPTARG}"
            ;;
        e)
            backend=${OPTARG}
            echo "backend is ${OPTARG}"
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
        l)
            level=${OPTARG}
            echo "level is ${OPTARG}"
            ;;
        ?)
            echo "unknown para"
            exit 1;;
    esac
done

basepath=$(pwd)
echo "base path is: ${basepath}"

# Set models config filepath
config_folder="config_level0"
if [[ ${level} == "level1" ]];then
    config_folder="config_level1"
fi
if [[ "${models_ms_train_config}" == "" ]]; then
    models_ms_train_config=${basepath}/../${config_folder}/models_ms_train.cfg
fi

if [[ $train_io_path == "" ]]; then
  echo "train_io path is empty"
  train_io_path=${models_path}/input_output
fi
echo $train_io_path

datasets_path=${models_path}/../datasets/

arm64_path=${release_path}/android_aarch64/npu
file=$(ls ${arm64_path}/*android-aarch64.tar.gz)
file_name="${file##*/}"
IFS="-" read -r -a file_name_array <<< "$file_name"
version_arm64=${file_name_array[2]}

arm32_path=${release_path}/android_aarch32/npu
file=$(ls ${arm32_path}/*android-aarch32.tar.gz)
file_name="${file##*/}"
IFS="-" read -r -a file_name_array <<< "$file_name"
version_arm32=${file_name_array[2]}

x86_path=${release_path}/centos_x86
file=$(ls ${x86_path}/*linux-x64.tar.gz)
file_name="${file##*/}"
IFS="-" read -r -a file_name_array <<< "$file_name"
version=${file_name_array[2]}

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

echo "start run converter ..."
train_cfg_file_list=("${models_ms_train_config}")
Run_Converter "${train_cfg_file_list[*]}"
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

# Empty config file is allowed, but warning message will be shown
if [[ $(Exist_File_In_Path ${ms_models_path} ".ms") != "true" ]]; then
  echo "No ms model found in ${ms_models_path}, please check if config file is empty!"
  exit 0
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

run_code_examples_log_file=${logs_path}/run_code_examples_log.txt
echo 'run code example logs: ' > ${run_code_examples_log_file}

isFailed=0
if [[ $backend == "all" || $backend == "train" || $backend == "x86_train" ]]; then
    # Run on x86
    echo "Start Run x86 ..."
    Run_x86 &
    Run_x86_PID=$!
    sleep 1
fi
if [[ $backend == "all" || $backend == "train" || $backend == "x86_train" || $backend == "x86-java" || $backend == "arm64_train" ]]; then
    # Run Code Examples 
    echo "Start Code Examples ..."
    Run_CodeExamples &
    Run_CodeExamples_PID=$!
    sleep 1
fi
if [[ $backend == "all" || $backend == "train" || $backend == "arm64_train" ]]; then
    # Run on arm64
    echo "Start Run arm64 ..."
    Run_arm arm64 &
    Run_arm64_PID=$!
    sleep 1
fi
if [[ $backend == "all" || $backend == "train" || $backend == "arm32_train" ]]; then
    # Run on arm32
    echo "Start Run arm32 ..."
    Run_arm arm32 &
    Run_arm32_PID=$!
    sleep 1
fi

if [[ $backend == "all" || $backend == "train" || $backend == "x86_train" ]]; then
    wait ${Run_x86_PID}
    Run_x86_status=$?
    if [[ ${Run_x86_status} != 0 ]];then
        echo "Run_x86 train failed"
        cat ${run_x86_log_file}
        isFailed=1
    fi
fi
if [[ $backend == "all" || $backend == "train" || $backend == "x86_train" || $backend == "x86-java" || $backend == "arm64_train" ]]; then
    wait ${Run_CodeExamples_PID}
    Run_CodeExamples_status=$?
    if [[ ${Run_CodeExamples_status} != 0 ]];then
        echo "Run CodeExamples failed"
        cat ${run_code_examples_log_file}
        isFailed=1
    fi
fi
if [[ $backend == "all" || $backend == "train" || $backend == "arm64_train" ]]; then
    wait ${Run_arm64_PID}
    Run_arm64_status=$?
    if [[ ${Run_arm64_status} != 0 ]];then
        echo "Run_arm64 train failed"
        cat ${run_arm64_log_file}
        isFailed=1
    fi
fi
if [[ $backend == "all" || $backend == "train" || $backend == "arm32_train" ]]; then
    wait ${Run_arm32_PID}
    Run_arm32_status=$?
    if [[ ${Run_arm32_status} != 0 ]];then
        echo "Run_arm32 train failed"
        cat ${run_arm32_log_file}
        isFailed=1
    fi
fi

END=$(date +%s.%N)
DIFF=$(echo "$END - $START" | bc)
echo "Test ended - Results:"
Print_Benchmark_Result $run_benchmark_train_result_file
echo "Test run Time:" $DIFF
exit ${isFailed}
