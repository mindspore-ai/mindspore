#!/bin/bash

# Convert models:
function Convert() {
  # $1:cfgFileList; $2:inModelPath; $3:outModelPath; $4:logFile; $5:resultFile; $6:failNotReturn;
  local cfg_file_list model_info model_name extra_info model_type cfg_file_name model_file weight_file output_file \
        quant_type bit_num config_file train_model in_dtype out_dtype converter_result cfg_file
  cfg_file_list=$1
  for cfg_file in ${cfg_file_list[*]}; do
    while read line; do
      if [[ $line == \#* || $line == "" ]]; then
        continue
      fi
      model_info=${line%% *}
      model_name=${model_info%%;*}
      extra_info=${model_info##*;}
      model_type=${model_name##*.}
      cfg_file_name=${cfg_file##*/}
      quant_config_path="${cfg_file%/*}/quant"
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
      output_file=$3"/"${model_name}
      quant_type=""
      bit_num=8
      config_file=""
      train_model="false"
      in_dtype="DEFAULT"
      out_dtype="DEFAULT"
      if [[ ${cfg_file_name} =~ "weightquant" ]]; then
        postfix=${cfg_file##*_}
        bit_num=${postfix:0:1}
        quant_type="WeightQuant"
        output_file=${output_file}"_${bit_num}bit"
        config_file="${quant_config_path}/weight_quant_${bit_num}bit.cfg"
      elif [[ ${cfg_file_name} =~ "_train" ]]; then
        train_model="true"
      elif [[ ${cfg_file_name} =~ "posttraining" ]]; then
        quant_type="PostTraining"
        output_file=${output_file}"_posttraining"
        config_file="${quant_config_path}/${model_name}_posttraining.config"
      elif [[ ${cfg_file_name} =~ "awaretraining" || ${extra_info} =~ "aware_training" ]]; then
        in_dtype="FLOAT"
        out_dtype="FLOAT"
      fi
      # start running converter
      echo ${model_name} >> "$4"
      echo './converter_lite  --fmk='${model_fmk}' --modelFile='${model_file}' --weightFile='${weight_file}' --outputFile='${output_file}\
        ' --inputDataType='${in_dtype}' --outputDataType='${out_dtype}' \
         --configFile='${config_file}' --trainModel='${train_model} >> "$4"
      ./converter_lite  --fmk=${model_fmk} --modelFile=${model_file} --weightFile=${weight_file} --outputFile=${output_file}\
        --inputDataType=${in_dtype} --outputDataType=${out_dtype} \
        --configFile=${config_file} --trainModel=${train_model} >> "$4"
      if [ $? = 0 ]; then
          converter_result='converter '${model_type}''${quant_type}' '${model_name}' pass';echo ${converter_result} >> $5
      else
          converter_result='converter '${model_type}''${quant_type}' '${model_name}' failed';echo ${converter_result} >> $5
          if [[ $6 != "ON" ]]; then
              return 1
          fi
      fi
    done < ${cfg_file}
  done
}

function Push_Files() {
    # $1:packagePath; $2:platform; $3:version; $4:localPath; $5:logFile; $6:deviceID;
    cd $1 || exit 1
    tar -zxf mindspore-lite-$3-android-$2.tar.gz || exit 1

    # If build with minddata, copy the minddata related libs
    cd $4 || exit 1
    if [ -f $1/mindspore-lite-$3-android-$2/runtime/lib/libminddata-lite.so ]; then
        cp -a $1/mindspore-lite-$3-android-$2/runtime/lib/libminddata-lite.so $4/libminddata-lite.so || exit 1
    fi
    if [ -f $1/mindspore-lite-$3-android-$2/runtime/third_party/hiai_ddk/lib/libhiai.so ]; then
      cp -a $1/mindspore-lite-$3-android-$2/runtime/third_party/hiai_ddk/lib/libhiai.so $4/libhiai.so || exit 1
      cp -a $1/mindspore-lite-$3-android-$2/runtime/third_party/hiai_ddk/lib/libhiai_ir.so $4/libhiai_ir.so || exit 1
      cp -a $1/mindspore-lite-$3-android-$2/runtime/third_party/hiai_ddk/lib/libhiai_ir_build.so $4/libhiai_ir_build.so || exit 1
    fi
    if [ -f $1/mindspore-lite-$3-android-$2/runtime/third_party/hiai_ddk/lib/libhiai_hcl_model_runtime.so ]; then
      cp -a $1/mindspore-lite-$3-android-$2/runtime/third_party/hiai_ddk/lib/libhiai_hcl_model_runtime.so $4/libhiai_hcl_model_runtime.so || exit 1
    fi

    cp -a $1/mindspore-lite-$3-android-$2/runtime/lib/libmindspore-lite.so $4/libmindspore-lite.so || exit 1
    cp -a $1/mindspore-lite-$3-android-$2/tools/benchmark/benchmark $4/benchmark || exit 1

    # adb push all needed files to the phone
    adb -s $6 push $4 /data/local/tmp/ > $5

    arm32_dir=""
    if [[ $2 == "aarch32" ]]; then
      arm32_dir="arm32/"
    fi
    # run adb ,run session ,check the result:
    echo 'cd  /data/local/tmp/benchmark_test' > adb_cmd.txt
    echo 'cp  /data/local/tmp/'$arm32_dir'libc++_shared.so ./' >> adb_cmd.txt
    echo 'chmod 777 benchmark' >> adb_cmd.txt

    adb -s $6 shell < adb_cmd.txt
}

# Run converted models:
function Run_Benchmark() {
  # $1:cfgFileList; $2:modelPath; $3:dataPath; $4:logFile; $5:resultFile; $6:platform; $7:processor; $8:phoneId; $9:failNotReturn;
  local cfg_file_list cfg_file_name line_info model_info spec_acc_limit model_name input_num input_shapes spec_threads \
        extra_info benchmark_mode infix mode model_file input_files output_file data_path threads acc_limit enableFp16 \
        run_result cfg_file
  cfg_file_list=$1
  for cfg_file in ${cfg_file_list[*]}; do
    cfg_file_name=${cfg_file##*/}
    while read line; do
      line_info=${line}
      if [[ $line_info == \#* || $line_info == "" ]]; then
        continue
      fi
      model_info=`echo ${line_info} | awk -F ' ' '{print $1}'`
      spec_acc_limit=`echo ${line_info} | awk -F ' ' '{print $2}'`
      model_name=`echo ${model_info} | awk -F ';' '{print $1}'`
      input_info=`echo ${model_info} | awk -F ';' '{print $2}'`
      input_shapes=`echo ${model_info} | awk -F ';' '{print $3}'`
      spec_threads=`echo ${model_info} | awk -F ';' '{print $4}'`
      extra_info=`echo ${model_info} | awk -F ';' '{print $5}'`
      input_num=`echo ${input_info} | sed 's/:/;/' | awk -F ';' '{print $1}'`
      input_names=`echo ${input_info} | sed 's/:/;/' | awk -F ';' '{print $2}'`
      if [[ ${model_name##*.} == "caffemodel" ]]; then
        model_name=${model_name%.*}
      fi
      echo "Benchmarking ${model_name} $6 $7 ......"
      # adjust benchmark mode
      benchmark_mode="calib"
      if [[ $6 == "arm64" && $7 == "CPU" && ! ${cfg_file_name} =~ "fp16" ]]; then
        benchmark_mode="calib+loop"
      fi
      # adjust precision mode
      mode="fp32"
      if [[ ${cfg_file_name} =~ "fp16" ]]; then
        mode="fp16"
      fi
      # adjust file name
      infix=""
      if [[ ${cfg_file_name} =~ "weightquant" ]]; then
        infix="_${cfg_file##*_}"
        infix=${infix%.*}
        benchmark_mode="calib"
      elif [[ ${cfg_file_name} =~ "_train" ]]; then
        infix="_train"
      elif [[ ${cfg_file_name} =~ "_posttraining" ]]; then
        model_name=${model_name}"_posttraining"
      elif [[ ${cfg_file_name} =~ "_process_only" ]]; then
        benchmark_mode="loop"
      elif [[ ${cfg_file_name} =~ "_compatibility" && ${spec_acc_limit} == "" ]]; then
        benchmark_mode="loop"
      fi
      model_file=$2"/${model_name}${infix}.ms"
      input_files=""
      output_file=""
      data_path=$3"/input_output/"
      if [[ ${input_num} == "" || ${input_num} == 1 ]]; then
        input_files=${data_path}'input/'${model_name}'.ms.bin'
      else
        for i in $(seq 1 $input_num)
        do
          input_files=${input_files}${data_path}'input/'${model_name}'.ms.bin_'$i','
        done
      fi
      output_file=${data_path}'output/'${model_name}'.ms.out'
      # adjust threads
      threads="2"
      if [[ ${spec_threads} != "" ]]; then
        threads="${spec_threads}"
      fi
      # set accuracy limitation
      acc_limit="0.5"
      if [[ ${cfg_file_name} =~ "_train" ]]; then
        acc_limit="1.5"
      fi
      if [[ ${spec_acc_limit} != "" ]]; then
        acc_limit="${spec_acc_limit}"
      elif [[ $7 == "GPU" ]] && [[ ${mode} == "fp16" || ${cfg_file_name} =~ "_weightquant" ]]; then
        acc_limit="5"
      fi
      # whether enable fp16
      enableFp16="false"
      if [[ ${mode} == "fp16" ]]; then
        enableFp16="true"
      fi
      if [[ ${extra_info} =~ "calib_only" ]]; then
        benchmark_mode="calib"
      fi
      # start running benchmark
      echo "---------------------------------------------------------" >> "$4"
      if [[ ${benchmark_mode} = "calib" || ${benchmark_mode} = "calib+loop" ]]; then
        echo "$6 $7 ${mode} run calib: ${model_name}, accuracy limit:${acc_limit}" >> "$4"
        if [[ $6 == "arm64" || $6 == "arm32" ]]; then
          echo 'cd  /data/local/tmp/benchmark_test' > adb_run_cmd.txt
          echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test' >> adb_run_cmd.txt
          echo './benchmark --modelFile='${model_file}' --inDataFile='${input_files}' --inputShapes='${input_shapes}' --benchmarkDataFile='${output_file}' --enableFp16='${enableFp16}' --accuracyThreshold='${acc_limit}' --device='$7' --numThreads='${threads} >> adb_run_cmd.txt
          echo './benchmark --modelFile='${model_file}' --inDataFile='${input_files}' --inputShapes='${input_shapes}' --benchmarkDataFile='${output_file}' --enableFp16='${enableFp16}' --accuracyThreshold='${acc_limit}' --device='$7' --numThreads='${threads}>> $4
          cat adb_run_cmd.txt >> "$4"
          adb -s $8 shell < adb_run_cmd.txt >> "$4"
        else
          echo 'MSLITE_BENCH_INPUT_NAMES=${input_names} ./benchmark --modelFile='${model_file}' --inDataFile='${input_files}' --inputShapes='${input_shapes}' --benchmarkDataFile='${output_file}' --accuracyThreshold='${acc_limit}' --numThreads='${threads} >> "$4"
          MSLITE_BENCH_INPUT_NAMES=${input_names} ./benchmark --modelFile=${model_file} --inDataFile=${input_files} --inputShapes=${input_shapes} --benchmarkDataFile=${output_file} --accuracyThreshold=${acc_limit} --numThreads=${threads} >> "$4"
        fi
        if [ $? = 0 ]; then
          run_result="$6_$7_${mode}: ${model_file##*/} pass"; echo ${run_result} >> $5
        else
          run_result="$6_$7_${mode}: ${model_file##*/} failed"; echo ${run_result} >> $5
          if [[ $9 != "ON" ]]; then
              return 1
          fi
        fi
      fi
      # run benchmark without clib data recurrently for guarding the repeated graph execution scene
      if [[ ${benchmark_mode} = "loop" || ${benchmark_mode} = "calib+loop" ]]; then
        echo "$6 $7 ${mode} run loop: ${model_name}" >> "$4"
        if [[ ! ${extra_info} =~ "input_dependent" ]]; then
          input_files=""
        fi
        if [[ $6 == "arm64" || $6 == "arm32" ]]; then
          echo 'cd  /data/local/tmp/benchmark_test' > adb_run_cmd.txt
          echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test' >> adb_run_cmd.txt
          echo './benchmark --inDataFile='${input_files}' --modelFile='${model_file}' --inputShapes='${input_shapes}' --enableFp16='${enableFp16}' --warmUpLoopCount=0 --loopCount=2 --device='$7' --numThreads='${threads} >> adb_run_cmd.txt
          echo './benchmark --inDataFile='${input_files}' --modelFile='${model_file}' --inputShapes='${input_shapes}' --enableFp16='${enableFp16}' --warmUpLoopCount=0 --loopCount=2 --device='$7' --numThreads='${threads} >> $4
          cat adb_run_cmd.txt >> "$4"
          adb -s $8 shell < adb_run_cmd.txt >> "$4"
        else
          echo './benchmark --inDataFile='${input_files}' --modelFile='${model_file}' --inputShapes='${input_shapes}' --warmUpLoopCount=0 --loopCount=2 --numThreads='${threads} >> "$4"
          ./benchmark --inDataFile=${input_files} --modelFile=${model_file} --inputShapes=${input_shapes} --warmUpLoopCount=0 --loopCount=2 --numThreads=${threads} >> "$4"
        fi
        if [ $? = 0 ]; then
            run_result="$6_$7_${mode}_loop: ${model_file##*/} pass"; echo ${run_result} >> $5
        else
            run_result="$6_$7_${mode}_loop: ${model_file##*/} failed"; echo ${run_result} >> $5
            if [[ $9 != "ON" ]]; then
                return 1
            fi
        fi
      fi
    done < ${cfg_file}
  done
}

# Print start msg before run testcase
function MS_PRINT_TESTCASE_START_MSG() {
    echo ""
    echo -e "-----------------------------------------------------------------------------------------------------------------------------------"
    echo -e "env                    Testcase                                                                                           Result   "
    echo -e "---                    --------                                                                                           ------   "
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
    done < $1
    MS_PRINT_TESTCASE_END_MSG
}

function Print_Benchmark_Result() {
    MS_PRINT_TESTCASE_START_MSG
    while read line; do
        arr=("${line}")
        printf "%-20s %-100s %-7s\n" ${arr[0]} ${arr[1]} ${arr[2]}
    done < $1
    MS_PRINT_TESTCASE_END_MSG
}