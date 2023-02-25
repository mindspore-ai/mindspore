#!/bin/bash
source ./scripts/base_functions.sh

function Run_x86_codegen() {
    # $1:buildPath $2:modelPath $3:models_list $4:logFile $5:resultFile $6:micro_cofig $7:parallel_flag
    local bind_mode thread_num suffix run_result
    rm -rf $1
    mkdir -p $1

    # Unzip x86 runtime and converter
    cd ${x86_path} || exit 1
    tar -zxf mindspore-lite-${version}-linux-x64.tar.gz || exit 1
    cd ${x86_path}/mindspore-lite-${version}-linux-x64/ || exit 1

    cp tools/converter/converter/converter_lite ./ || exit 1
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:./tools/converter/lib/:./tools/converter/third_party/glog/lib

    while read line; do
      if [[ $line == \#* || $line == "" ]]; then
        continue
      fi
      model_info=`echo ${line} | awk -F ' ' '{print $1}'`
      model_name=`echo ${model_info} | awk -F ';' '{print $1}'`
      input_info=`echo ${model_info} | awk -F ';' '{print $2}'`
      input_shapes=`echo ${model_info} | awk -F ';' '{print $3}'`
      input_num=`echo ${input_info} | sed 's/:/;/' | awk -F ';' '{print $1}'`
      input_names=`echo ${input_info} | sed 's/:/;/' | awk -F ';' '{print $2}'`
      spec_shapes=""
      if [[ ${input_shapes} != "" && ${input_names} != "" ]]; then
          if [[ ${input_num} == "" ]]; then
            input_num=1
          fi
          IFS="," read -r -a name_array <<< ${input_names}
          IFS=":" read -r -a shape_array <<< ${input_shapes}
          for i in $(seq 0 $((${input_num}-1)))
          do
            spec_shapes=${spec_shapes}${name_array[$i]}':'${shape_array[$i]}';'
          done
      fi

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
      # set parameters
      model_file=$2"/"${model_name}
      weight_file=""
      if [[ $model_fmk == "CAFFE" ]]; then
        model_file=${model_file}".prototxt"
        weight_file=${model_file%.*}".caffemodel"
      fi
      output_file=$1"/"${model_name}
      quant_type=""
      config_file=$6
      train_model="false"
      in_dtype="DEFAULT"
      out_dtype="DEFAULT"

      # start running converter
      cd ${x86_path}/mindspore-lite-${version}-linux-x64/ || exit 1
      echo ${model_name} >> "$4"
      echo './converter_lite  --fmk='${model_fmk}' --modelFile='${model_file}' --weightFile='${weight_file}' --outputFile='${output_file}\
        ' --inputDataType='${in_dtype}' --outputDataType='${out_dtype}' --inputShape='${spec_shapes}\
        ' --configFile='${config_file}' --trainModel='${train_model} >> "$4"
      ./converter_lite  --fmk=${model_fmk} --modelFile=${model_file} --weightFile=${weight_file} --outputFile=${output_file}\
        --inputDataType=${in_dtype} --outputDataType=${out_dtype} --inputShape=${spec_shapes}\
        --configFile=${config_file} --trainModel=${train_model} >> "$4"
      if [ $? = 0 ]; then
          converter_result='converter '${model_type}''${quant_type}' '${model_name}' pass';echo ${converter_result} >> $5
      else
          converter_result='converter '${model_type}''${quant_type}' '${model_name}' failed';echo ${converter_result} >> $5
          return 1;
      fi

      bind_mode="0"
      thread_num="1"
      suffix=""
      if [[ $7 == "parallel" ]]; then
          bind_mode="0"
          thread_num="4"
          suffix="_parallel"
      fi
      echo ${model_name} >> "$4"

      # 1. build benchmark
      mkdir -p ${output_file}/build && cd ${output_file}/build || exit 1
      cmake -DPKG_PATH=${x86_path}/mindspore-lite-${version}-linux-x64 ${output_file} >> $4
      make || return 1
      # 2. run benchmark
      if [[ ${input_num} == "" || ${input_num} == 1 ]]; then
        input_files=${models_path}/input_output/input/${model_name}.ms.bin
      else
        for i in $(seq 1 $input_num)
        do
          input_files=${input_files}${models_path}'/input_output/input/'${model_name}'.ms.bin_'$i','
        done
      fi
      echo "net file: ${output_file}/src/model0/net0.bin" >> $4
      echo "./benchmark ${input_files} ${output_file}/src/model0/net0.bin 1 ${models_path}/input_output/output/${model_name}.ms.out ${thread_num} ${bind_mode} 0" >> $4
      ./benchmark ${input_files} ${output_file}/src/model0/net0.bin 1 ${models_path}/input_output/output/${model_name}.ms.out ${thread_num} ${bind_mode} 0 >> $4
      if [ $? = 0 ]; then
          run_result='x86_codegen'${suffix}': '${model_name}' pass'; echo ${run_result} >> $5
      else
          run_result='x86_codegen'${suffix}': '${model_name}' failed'; echo ${run_result} >> $5;
          return 1;
      fi
    done < $3
}

function Run_cortex_m_codegen() {
    # $1:buildPath $2:modelPath $3:models_list $4:logFile $5:resultFile $6:micro_cofig
    local bind_mode thread_num suffix run_result
    rm -rf $1
    mkdir -p $1

    if [[ "X$STM32_DEMO_PATH" == "X" ]]; then
      echo "error: to run cortex-m ci, you need to set STM32_DEMO_PATH to declare the path of STM32 project."
      exit 1
    fi
    if [[ "X$STM32_CUBE_PROG_PATH" == "X" ]]; then
      echo "error: to run cortex-m ci, you need to set STM32_CUBE_PROG_PATH to declare the path of STM32CubeProgrammer."
      exit 1
    fi

    # Unzip x86 runtime and converter
    cd ${x86_path} || exit 1
    tar -zxf mindspore-lite-${version}-linux-x64.tar.gz || exit 1
    cd ${x86_path}/mindspore-lite-${version}-linux-x64/ || exit 1

    cp tools/converter/converter/converter_lite ./ || exit 1
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:./tools/converter/lib/:./tools/converter/third_party/glog/lib

    while read line; do
      if [[ $line == \#* || $line == "" ]]; then
        continue
      fi
      model_info=`echo ${line} | awk -F ' ' '{print $1}'`
      model_name=`echo ${model_info} | awk -F ';' '{print $1}'`
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
      # set parameters
      model_file=$2"/"${model_name}
      weight_file=""
      if [[ $model_fmk == "CAFFE" ]]; then
        model_file=${model_file}".prototxt"
        weight_file=${model_file%.*}".caffemodel"
      fi
      stm_demo_file=$1"/stm32f767/"
      [ -n "${stm_demo_file}" ] && rm -rf ${stm_demo_file}
      cp -r ${STM32_DEMO_PATH}/stm32f767 $1/
      # output_file=$1"/stm32f767/"${model_name}
      output_file=$1"/stm32f767/gen_output"
      quant_type=""
      config_file=$6
      spec_shapes=""
      train_model="false"
      in_dtype="DEFAULT"
      out_dtype="DEFAULT"

      # start running converter
      cd ${x86_path}/mindspore-lite-${version}-linux-x64/ || exit 1
      echo ${model_name} >> "$4"
      echo './converter_lite  --fmk='${model_fmk}' --modelFile='${model_file}' --weightFile='${weight_file}' --outputFile='${output_file}\
        ' --inputDataType='${in_dtype}' --outputDataType='${out_dtype}' --inputShape='${spec_shapes}\
        ' --configFile='${config_file}' --trainModel='${train_model} >> "$4"
      ./converter_lite  --fmk=${model_fmk} --modelFile=${model_file} --weightFile=${weight_file} --outputFile=${output_file}\
        --inputDataType=${in_dtype} --outputDataType=${out_dtype} --inputShape=${spec_shapes}\
        --configFile=${config_file} --trainModel=${train_model} >> "$4"
      if [ $? = 0 ]; then
          converter_result='converter '${model_type}''${quant_type}' '${model_name}' pass';echo ${converter_result} >> $5
      else
          converter_result='converter '${model_type}''${quant_type}' '${model_name}' failed';echo ${converter_result} >> $5
          return 1;
      fi

      bind_mode="0"
      thread_num="1"
      suffix=""
      if [[ $3 =~ "parallel" ]]; then
          bind_mode="0"
          thread_num="4"
          suffix="_parallel"
      fi
      echo ${model_name} >> "$4"

      # 1. build benchmark
      mkdir -p ${output_file}/build || exit 1
      cp ${cortex_path}/mindspore-lite-${version}-none-cortex-m7.tar.gz ${output_file}/ || exit 1
      cd ${output_file} || exit 1
      in_data=`cat ${models_path}/input_output/input/${model_name}.ms.in.txt`
      out_data=`cat ${models_path}/input_output/output/${model_name}.ms.out.txt`
      sed -i "s/float calib_input0_data\[NET_INPUT0_SIZE\] = {};/float calib_input0_data\[NET_INPUT0_SIZE\] = {${in_data}};/g" benchmark/data.c
      sed -i "s/float calib_output0_data\[NET_OUTPUT0_SIZE\] = {};/float calib_output0_data\[NET_OUTPUT0_SIZE\] = {${out_data}};/g" benchmark/data.c
      sed -i "s/VERSION_STR=.*/VERSION_STR=${version}/g" build.sh
      bash build.sh || exit 1
      cp -r ${output_file}/mindspore-lite-${version}-none-cortex-m7 ${output_file}/build/
      cd ${stm_demo_file} || exit 1
      [ -n "${stm_demo_file}" ] && rm -rf ${stm_demo_file}/build
      sed -i "s/LITE_PACK =/LITE_PACK = mindspore-lite-${version}-none-cortex-m7/g" Makefile
      make >> "$4" || return 1

      # 2. run benchmark
      bash ${STM32_CUBE_PROG_PATH}/bin/STM32_Programmer.sh -c port=SWD -w build/test_767_01.bin 0x08000000 -s 0x08000000 || exit 1
      sleep 3
      bash ${STM32_CUBE_PROG_PATH}/bin/STM32_Programmer.sh -c port=SWD model=HOTPLUG --upload 0x20000000 0x1 ret.bin  || exit 1
      calib_ret=`cat ret.bin`
      if [[ ${calib_ret} = 1 ]];then
          run_result='cortex_codegen'${suffix}': '${model_name}' pass'; echo ${run_result} >> $5
      else
          run_result='cortex_codegen'${suffix}': '${model_name}' failed'; echo ${run_result} >> $5;
          echo "return is "${calib_ret} >> $5;
          return 1;
      fi
    done < $3
}

function Run_quant_codegen() {
    # $1:buildPath $2:modelPath $3:models_list $4:logFile $5:resultFile $6:micro_cofig
    local bind_mode thread_num suffix run_result
    rm -rf $1
    mkdir -p $1

    # Unzip x86 runtime and converter
    cd ${x86_path} || exit 1
    tar -zxf mindspore-lite-${version}-linux-x64.tar.gz || exit 1
    cd ${x86_path}/mindspore-lite-${version}-linux-x64/ || exit 1

    cp tools/converter/converter/converter_lite ./ || exit 1
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:./tools/converter/lib/:./tools/converter/third_party/glog/lib

    while read line; do
      if [[ $line == \#* || $line == "" ]]; then
        continue
      fi
      model_info=`echo ${line} | awk -F ' ' '{print $1}'`
      cosine_threshold=`echo ${line} | awk -F ' ' '{print $2}'`
      model_name=`echo ${model_info} | awk -F ';' '{print $1}'`
      input_info=`echo ${model_info} | awk -F ';' '{print $2}'`
      input_shapes=`echo ${model_info} | awk -F ';' '{print $3}'`
      input_num=`echo ${input_info} | sed 's/:/;/' | awk -F ';' '{print $1}'`
      input_names=`echo ${input_info} | sed 's/:/;/' | awk -F ';' '{print $2}'`

      spec_shapes=""
      IFS="," read -r -a name_array <<< ${input_names}
      if [[ ${input_shapes} != "" && ${input_names} != "" ]]; then
          if [[ ${input_num} == "" ]]; then
            input_num=1
          fi
          IFS=":" read -r -a shape_array <<< ${input_shapes}
          for i in $(seq 0 $((${input_num}-1)))
          do
            spec_shapes=${spec_shapes}${name_array[$i]}':'${shape_array[$i]}';'
          done
      fi
      data_path=${models_path}"/input_output/"
      model_calib_dir=$1"/"${model_name}_calib
      rm -rf ${model_calib_dir}
      mkdir -p ${model_calib_dir}
      calibrate_paths=""
      if [[ ${input_num} == "" || ${input_num} == 1 ]]; then
        cp ${data_path}'input/'${model_name}'.ms.bin' ${model_calib_dir}"/0.bin"
        calibrate_paths='"'${name_array[0]}'":'${model_calib_dir}"/"
      else
        for i in $(seq 1 $input_num)
        do
          mkdir -p ${model_calib_dir}"/"${i}
          cp ${data_path}'input/'${model_name}'.ms.bin_'$i  ${model_calib_dir}"/"${i}
          calibrate_paths=${calibrate_paths}'"'${name_array[$i-1]}'":'${model_calib_dir}"/"${i}','
        done
      fi
      echo "calib_paths:${calibrate_paths}" >> "$4"

      config_file=$1"/"${model_name}_micro_quant.cfg
      cp $6 ${config_file} || exit 1
      sed -i "s#calibrate_path=#calibrate_path=${calibrate_paths}#g" ${config_file}

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
      # set parameters
      model_file=$2"/"${model_name}
      weight_file=""
      if [[ $model_fmk == "CAFFE" ]]; then
        model_file=${model_file}".prototxt"
        weight_file=${model_file%.*}".caffemodel"
      fi

      output_file=$1"/"${model_name}
      quant_type=""
      train_model="false"
      in_dtype="DEFAULT"
      out_dtype="DEFAULT"

      # start running converter
      cd ${x86_path}/mindspore-lite-${version}-linux-x64/ || exit 1
      echo "model:"${model_name}
      echo ${model_name} >> "$4"
      echo './converter_lite  --fmk='${model_fmk}' --modelFile='${model_file}' --weightFile='${weight_file}' --outputFile='${output_file}\
        ' --inputDataType='${in_dtype}' --outputDataType='${out_dtype}' --inputShape='\"${spec_shapes}\"\
        ' --configFile='${config_file}' --trainModel='${train_model} >> "$4"
      ./converter_lite  --fmk=${model_fmk} --modelFile=${model_file} --weightFile=${weight_file} --outputFile=${output_file}\
        --inputDataType=${in_dtype} --outputDataType=${out_dtype} --inputShape="${spec_shapes}"\
        --configFile=${config_file} --trainModel=${train_model} >> "$4"
      if [ $? = 0 ]; then
          converter_result='converter '${model_type}' quant '${model_name}' pass';echo ${converter_result} >> $5
      else
          converter_result='converter '${model_type}' quant '${model_name}' failed';echo ${converter_result} >> $5
          return 1;
      fi

      bind_mode="0"
      thread_num="1"
      suffix="_quant"
      echo ${model_name} >> "$4"

      # 1. build benchmark
      mkdir -p ${output_file}/build && cd ${output_file}/build || exit 1
      cmake -DPKG_PATH=${x86_path}/mindspore-lite-${version}-linux-x64 ${output_file} >> $4
      make || return 1
      # 2. run benchmark
      input_files=""
      if [[ ${input_num} == "" || ${input_num} == 1 ]]; then
        input_files=${models_path}/input_output/input/${model_name}.ms.bin
      else
        for i in $(seq 1 $input_num)
        do
          input_files=${input_files}${models_path}'/input_output/input/'${model_name}'.ms.bin_'$i','
        done
      fi
      benchmark_data_file=${models_path}"/input_output/output/"${model_name}.ms.out
      echo "net file: ${output_file}/src/model0/net0.bin" >> $4
      echo "./benchmark ${input_files} ${output_file}/src/model0/net0.bin 1  ${benchmark_data_file} ${thread_num} ${bind_mode} 0 ${cosine_threshold}" >> $4
      ./benchmark ${input_files} ${output_file}/src/model0/net0.bin 1 ${benchmark_data_file} ${thread_num} ${bind_mode} 0 ${cosine_threshold}>> $4
      if [ $? = 0 ]; then
          run_result='x86_codegen'${suffix}': '${model_name}' pass'; echo ${run_result} >> $5
      else
          run_result='x86_codegen'${suffix}': '${model_name}' failed'; echo ${run_result} >> $5;
          return 1;
      fi
    done < $3
}

function Run_arm_codegen() {
    # $1:buildPath $2:modelPath $3:model_list $4:logFile $5:resultFile $6:deviceID $7:processor $8:micro_cofig $9:failNotReturn;
    local package_path package_suffix target platform android_abi toolchain_name package_path run_result
    echo "ANDROID_NDK: ${ANDROID_NDK}" >> $4
    package_path=${arm64_path}
    package_suffix="aarch64"
    target="ARM64"
    platform=${target}
    android_abi="arm64-v8a"
    toolchain_name="aarch64-linux-android-clang"
    if [[ $7 == "arm32" ]]; then
      package_suffix="aarch32"
      target="ARM32"
      platform="ARM32"
      android_abi="armeabi-v7a"
      toolchain_name="clang"
      package_path=${arm32_path}
    fi
    cd ${package_path} || exit 1
    tar -zxf mindspore-lite-${version}-android-${package_suffix}.tar.gz || exit 1
    local PKG_PATH=${package_path}/mindspore-lite-${version}-android-${package_suffix}

    # Unzip x86 runtime and converter
    cd ${x86_path} || exit 1
    tar -zxf mindspore-lite-${version}-linux-x64.tar.gz || exit 1
    cd ${x86_path}/mindspore-lite-${version}-linux-x64/ || exit 1

    cp tools/converter/converter/converter_lite ./ || exit 1
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:./tools/converter/lib/:./tools/converter/third_party/glog/lib

    rm -rf $1
    mkdir -p $1

    # Run tflite converted models:
    while read line; do
      if [[ $line == \#* || $line == "" ]]; then
        continue
      fi
      model_info=`echo ${line} | awk -F ' ' '{print $1}'`
      model_name=`echo ${model_info} | awk -F ';' '{print $1}'`
      input_info=`echo ${model_info} | awk -F ';' '{print $2}'`
      input_shapes=`echo ${model_info} | awk -F ';' '{print $3}'`
      input_num=`echo ${input_info} | sed 's/:/;/' | awk -F ';' '{print $1}'`
      input_names=`echo ${input_info} | sed 's/:/;/' | awk -F ';' '{print $2}'`
      spec_shapes=""
      if [[ ${input_shapes} != "" && ${input_names} != "" ]]; then
          if [[ ${input_num} == "" ]]; then
            input_num=1
          fi
          IFS="," read -r -a name_array <<< ${input_names}
          IFS=":" read -r -a shape_array <<< ${input_shapes}
          for i in $(seq 0 $((${input_num}-1)))
          do
            spec_shapes=${spec_shapes}${name_array[$i]}':'${shape_array[$i]}';'
          done
      fi

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
      # set parameters
      model_file=$2"/"${model_name}
      weight_file=""
      if [[ $model_fmk == "CAFFE" ]]; then
        model_file=${model_file}".prototxt"
        weight_file=${model_file%.*}".caffemodel"
      fi
      output_file=$1"/"${model_name}
      quant_type=""
      config_file=$8
      train_model="false"
      in_dtype="DEFAULT"
      out_dtype="DEFAULT"

      # start running converter
      cd ${x86_path}/mindspore-lite-${version}-linux-x64/ || exit 1
      echo ${model_name} >> "$4"
      echo './converter_lite  --fmk='${model_fmk}' --modelFile='${model_file}' --weightFile='${weight_file}' --outputFile='${output_file}\
        ' --inputDataType='${in_dtype}' --outputDataType='${out_dtype}' --inputShape='${spec_shapes}\
        ' --configFile='${config_file}' --trainModel='${train_model} >> "$4"
      ./converter_lite  --fmk=${model_fmk} --modelFile=${model_file} --weightFile=${weight_file} --outputFile=${output_file}\
        --inputDataType=${in_dtype} --outputDataType=${out_dtype} --inputShape=${spec_shapes}\
        --configFile=${config_file} --trainModel=${train_model} >> "$4"
      if [ $? = 0 ]; then
          converter_result='converter '${model_type}''${quant_type}' '${model_name}' pass';echo ${converter_result} >> $5
      else
          converter_result='converter '${model_type}''${quant_type}' '${model_name}' failed';echo ${converter_result} >> $5
          return 1;
      fi

      rm -rf $1/benchmark
      mkdir -p $1/benchmark && cd $1/benchmark || exit 1

      {
          echo "cmake -DCMAKE_BUILD_TYPE=Release
                -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake
                -DANDROID_ABI=${android_abi}
                -DANDROID_TOOLCHAIN_NAME=${toolchain_name}
                -DANDROID_NATIVE_API_LEVEL=19
                -DPLATFORM_${platform}=ON
                -DPKG_PATH=${PKG_PATH} $1/${model_name}"

          cmake -DCMAKE_BUILD_TYPE=Release \
                -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK}/build/cmake/android.toolchain.cmake" \
                -DANDROID_ABI=${android_abi} \
                -DANDROID_TOOLCHAIN_NAME=${toolchain_name} \
                -DANDROID_NATIVE_API_LEVEL="19" \
                -DPLATFORM_${platform}=ON \
                -DPKG_PATH=${PKG_PATH} $1/${model_name}
          make -j4
      } >> $4  2>&1 || return 1

      benchmark_dir="$1/codegen_test_$7"
      rm -rf "$benchmark_dir"
      mkdir "$benchmark_dir" && cd "$benchmark_dir" || exit 1
      cp -a "$1/benchmark/benchmark" "$benchmark_dir/benchmark" || exit 1
      cp -a "$1/$model_name/src/model0/net0.bin" "$benchmark_dir/net.bin" || exit 1

      {
            echo "ls $benchmark_dir:"
            ls "$benchmark_dir"
      } >> $4

      # adb push all needed files to the phone
      adb -s $6 push "$benchmark_dir" /data/local/tmp/ > adb_push_log.txt
      {
          echo "cd  /data/local/tmp/codegen_test_$7"
          echo 'chmod 777 benchmark'
          echo 'chmod 777 net0.bin'
          echo 'ls'
          echo './benchmark /data/local/tmp/input_output/input/'${model_name}'.ms.bin ./net0.bin 1 /data/local/tmp/input_output/output/'${model_name}'.ms.out'
          echo "cd .. && rm -rf codegen_test_$7"
      } >> $4

        {
            echo "cd  /data/local/tmp/codegen_test_$7"
            echo 'chmod 777 benchmark'
            echo 'chmod 777 net0.bin'
            echo 'ls'
            echo './benchmark /data/local/tmp/input_output/input/'${model_name}'.ms.bin ./net0.bin 1 /data/local/tmp/input_output/output/'${model_name}'.ms.out'
            echo "cd .. && rm -rf codegen_test_$7"
        } > adb_run_cmd.txt

        adb -s $6 shell < adb_run_cmd.txt >> $4
        if [ $? = 0 ]; then
            run_result=$7'_codegen: '${model_name}' pass'; echo ${run_result} >> $5
        else
            run_result=$7'_codegen: '${model_name}' failed'; echo ${run_result} >> $5;
            return 1;
        fi
    done < $3

    rm -rf $1
}

basepath=$(pwd)
echo ${basepath}

# Example:sh run_benchmark_codegen.sh -r /home/temp_test -m /home/temp_test/models -d "8KE5T19620002408" -e arm_cpu
while getopts "r:m:d:e:l:" opt; do
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
        l)
            level=${OPTARG}
            echo "level is ${OPTARG}"
            ;;
        ?)
        echo "unknown para"
        exit 1;;
    esac
done

# package info
x86_path=${release_path}/centos_x86
arm32_path=${release_path}/android_aarch32/npu
arm64_path=${release_path}/android_aarch64/npu
cortex_path=${release_path}/none_cortex-m
file_name=$(ls ${x86_path}/*-linux-x64.tar.gz)
IFS="-" read -r -a file_name_array <<< "$file_name"
version=${file_name_array[2]}

config_folder="config_level0"
if [[ ${level} == "level1" ]]; then
    config_folder="config_level1"
fi
# Set model-list
models_codegen_config=${basepath}/../${config_folder}/models_codegen.cfg
models_cortex_codegen_config=${basepath}/../${config_folder}/models_codegen_cortex.cfg
models_codegen_parallel_config=${basepath}/../${config_folder}/models_codegen.cfg
models_quant_codegen_config=${basepath}/../${config_folder}/models_codegen_quant.cfg

#micro config
micro_x86_config=${basepath}/../${config_folder}/micro/micro_x86.cfg
micro_x86_parallel_config=${basepath}/../${config_folder}/micro/micro_x86_parallel.cfg
micro_arm64_config=${basepath}/../${config_folder}/micro/micro_arm64.cfg
micro_ARM32_config=${basepath}/../${config_folder}/micro/micro_arm32A.cfg
micro_cortex_config=${basepath}/../${config_folder}/micro/micro_cortex_m.cfg
micro_quant_config=${basepath}/../${config_folder}/micro/micro_x86_quant.cfg

# Set models and build path
build_path_x86=${basepath}/codegen_build_x86
build_path_parallel=${basepath}/codegen_build_parallel
build_path_arm64=${basepath}/codegen_build_arm64
build_path_arm32=${basepath}/codegen_build_arm32
build_path_cortex=${basepath}/codegen_build_cortex
build_path_quant=${basepath}/codegen_build_quant

# Write converter result to temp file
run_converter_log_file=${basepath}/run_converter_log.txt
echo ' ' > ${run_converter_log_file}
run_converter_result_file=${basepath}/run_converter_result.txt
echo ' ' > ${run_converter_result_file}

# Write benchmark result to temp file
run_benchmark_result_file=${basepath}/run_benchmark_result.txt
echo ' ' > ${run_benchmark_result_file}

run_arm32_fp32_codegen_log_file=${basepath}/run_arm32_fp32_codegen_log.txt
echo 'run arm32_codegen logs: ' > ${run_arm32_fp32_codegen_log_file}
run_arm64_fp32_codegen_log_file=${basepath}/run_arm64_fp32_codegen_log.txt
echo 'run arm64_codegen logs: ' > ${run_arm64_fp32_codegen_log_file}
run_x86_codegen_log_file=${basepath}/run_x86_codegen_log.txt
echo 'run x86 codegen logs: ' > ${run_x86_codegen_log_file}
run_x86_codegen_parallel_log_file=${basepath}/run_x86_codegen_parallel_log.txt
echo 'run x86 codegen parallel logs: ' > ${run_x86_codegen_parallel_log_file}
run_cortex_codegen_log_file=${basepath}/run_cortex_codegen_log.txt
echo 'run cortex_codegen logs: ' > ${run_cortex_codegen_log_file}
run_quant_codegen_log_file=${basepath}/run_quant_codegen_log.txt
echo 'run x86_quant_codegen logs: ' > ${run_quant_codegen_log_file}

echo "input backend is ${backend}"
backend=${backend:-"all"}
isFailed=0
echo "current backend is ${backend}"
if [[ $backend == "all" || $backend == "codegen" || $backend == "x86_codegen" ]]; then
    # Run on x86-codegen
    echo "start Run x86 codegen ..."
    Run_x86_codegen ${build_path_x86} ${models_path} ${models_codegen_config} ${run_x86_codegen_log_file} ${run_benchmark_result_file} ${micro_x86_config} ""
    Run_x86_codegen_status=$?
#    Run_x86_codegen_PID=$!
#    sleep 1
fi
if [[ $backend == "all" || $backend == "codegen" || $backend == "x86_codegen" || $backend == "x86_codegen_parallel" ]]; then
    # Run on x86-codegen-parallel
    echo "start Run x86 codegen parallel ..."
    Run_x86_codegen ${build_path_parallel} ${models_path} ${models_codegen_parallel_config} ${run_x86_codegen_parallel_log_file} ${run_benchmark_result_file} ${micro_x86_parallel_config} "parallel"
    Run_x86_codegen_parallel_status=$?
#    Run_x86_codegen_parallel_PID=$!
#    sleep 1
fi
if [[ $backend == "all" || $backend == "codegen" || $backend == "arm64_codegen" ]]; then
    # Run on codegen
    echo "start Run arm64 codegen ..."
    Run_arm_codegen ${build_path_arm64} ${models_path} ${models_codegen_config} ${run_arm64_fp32_codegen_log_file} ${run_benchmark_result_file} ${device_id} "arm64" ${micro_arm64_config}
    Run_arm64_codegen_status=$?
#    Run_arm64_codegen_PID=$!
#    sleep 1
fi
if [[ $backend == "all" || $backend == "codegen" || $backend == "arm32_codegen" ]]; then
    # Run on arm32 codegen
    echo "start Run arm32 codegen ..."
    Run_arm_codegen ${build_path_arm32} ${models_path} ${models_codegen_config} ${run_arm32_fp32_codegen_log_file} ${run_benchmark_result_file} ${device_id} "arm32" ${micro_ARM32_config}
    Run_arm32_codegen_status=$?
#    Run_arm32_codegen_PID=$!
#    sleep 1
fi
if [[ $backend == "cortex_codegen" ]]; then
    # Run on codegen
    echo "start Run cortex codegen ..."
    Run_cortex_m_codegen ${build_path_cortex} ${models_path} ${models_cortex_codegen_config} ${run_cortex_codegen_log_file} ${run_benchmark_result_file} ${micro_cortex_config}
    Run_cortex_codegen_status=$?
#    Run_arm64_codegen_PID=$!
#    sleep 1
fi
if [[ $backend == "all" || $backend == "codegen" || $backend == "quant_codegen" ]]; then
    # Run on codegen
    echo "start Run quant codegen ..."
    Run_quant_codegen ${build_path_quant} ${models_path} ${models_quant_codegen_config} ${run_quant_codegen_log_file} ${run_benchmark_result_file} ${micro_quant_config}
    Run_quant_codegen_status=$?
#    Run_arm64_codegen_PID=$!
#    sleep 1
fi

if [[ $backend == "all" || $backend == "codegen" || $backend == "x86_codegen" ]]; then
#    wait ${Run_x86_codegen_PID}
#    Run_x86_codegen_status=$?
    if [[ ${Run_x86_codegen_status} != 0 ]];then
        echo "Run_x86 codegen failed"
        cat ${run_x86_codegen_log_file}
        isFailed=1
    fi
fi
if [[ $backend == "all" || $backend == "codegen" || $backend == "x86_codegen" || $backend == "x86_codegen_parallel" ]]; then
#    wait ${Run_x86_codegen_parallel_PID}
#    Run_x86_codegen_parallel_status=$?
    if [[ ${Run_x86_codegen_parallel_status} != 0 ]];then
        echo "Run_x86 codegen parallel failed"
        cat ${run_x86_codegen_log_file}
        isFailed=1
    fi
fi
if [[ $backend == "all" || $backend == "codegen" || $backend == "arm64_codegen" ]]; then
#    wait ${Run_arm64_codegen_PID}
#    Run_arm64_codegen_status=$?
    if [[ ${Run_arm64_codegen_status} != 0 ]];then
        echo "Run_arm64_codegen failed"
        cat ${run_arm64_fp32_codegen_log_file}
        isFailed=1
    fi
fi
if [[ $backend == "all" || $backend == "codegen" || $backend == "arm32_codegen" ]]; then
#    wait ${Run_arm32_codegen_PID}
#    Run_arm32_codegen_status=$?
    if [[ ${Run_arm32_codegen_status} != 0 ]];then
        echo "Run_arm32_codegen failed"
        cat ${run_arm32_fp32_codegen_log_file}
        isFailed=1
    fi
fi
if [[ $backend == "cortex_codegen" ]]; then
#    wait ${Run_arm32_codegen_PID}
#    Run_arm32_codegen_status=$?
    if [[ ${Run_cortex_codegen_status} != 0 ]];then
        echo "Run_cortex_codegen failed"
        cat ${run_cortex_codegen_log_file}
        isFailed=1
    fi
fi
if [[ $backend == "all" || $backend == "codegen" || $backend == "quant_codegen" ]]; then
#    wait ${Run_arm32_codegen_PID}
#    Run_arm32_codegen_status=$?
    if [[ ${Run_quant_codegen_status} != 0 ]];then
        echo "Run_quant_codegen failed"
        cat ${run_quant_codegen_log_file}
        isFailed=1
    fi
fi

echo "Run codegen is ended"
Print_Benchmark_Result $run_benchmark_result_file
exit ${isFailed}
