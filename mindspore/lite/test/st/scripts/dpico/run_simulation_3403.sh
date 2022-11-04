#!/bin/bash

function Run_Convert_MODELS() {
  framework=$1
  models_3403_cfg=$2
  while read line; do
      dpico_line_info=${line}
      if [[ $dpico_line_info == \#* || $dpico_line_info == "" ]]; then
        continue
      fi
      model_location=`echo ${dpico_line_info}|awk -F ' ' '{print $1}'`
      model_info=`echo ${dpico_line_info}|awk -F ' ' '{print $2}'`
      model_name=${model_info%%;*}
      length=`expr ${#model_name} + 1`
      input_shape=${model_info:${length}}

      # converter_lite convert model
      cp ${models_path}/${model_location}/${model_name}.cfg ./ || exit 1
      cp ${model_name}.cfg ${model_name}_atc.cfg
      sed -i '$a \[instruction_name] '${om_generated_path}/${model_name}_lib ./${model_name}.cfg

      ms_config_file=./converter_for_dpico.cfg
      echo '[registry]' > ${ms_config_file}
      echo 'plugin_path=./tools/converter/providers/SD3403/libdpico_atc_adapter.so' >> ${ms_config_file}
      echo -e 'disable_fusion=on\n' >> ${ms_config_file}
      echo '[dpico]' >> ${ms_config_file}
      echo 'dpico_config_path='./${model_name}.cfg >> ${ms_config_file}
      echo 'save_temporary_files=on' >> ${ms_config_file}
      echo -e 'benchmark_path=./tools/benchmark/benchmark' >> ${ms_config_file}
      echo ${model_name} >> "${run_converter_log_file}"
      if [[ ${framework} == 'CAFFE' ]]; then
        echo './converter_lite --inputDataFormat=NCHW --fmk='${framework}' --inputShape='${input_shape} '--modelFile='${models_path}'/'${model_location}'/model/'${model_name}.prototxt' --weightFile='${models_path}'/'${model_location}'/model/'${model_name}.caffemodel' --configFile='${ms_config_file}' --outputFile='${om_generated_path}'/'${model_name}'' >> "${run_converter_log_file}"
        ./converter_lite --inputDataFormat=NCHW --inputShape=${input_shape} --fmk=${framework} --modelFile=${models_path}/${model_location}/model/${model_name}.prototxt --weightFile=${models_path}/${model_location}/model/${model_name}.caffemodel --configFile=${ms_config_file} --outputFile=${om_generated_path}/${model_name}
        if [ $? = 0 ]; then
            converter_result='converter CAFFE '${model_name}' pass';echo ${converter_result} >> ${run_converter_result_file}
        else
            converter_result='converter CAFFE '${model_name}' failed';echo ${converter_result} >> ${run_converter_result_file};exit 1
        fi
      elif [[ ${framework} == 'ONNX' ]]; then
        echo './converter_lite --inputDataFormat=NCHW --fmk='${framework}' --inputShape='${input_shape} '--modelFile='${models_path}'/'${model_location}'/models/'${model_name}' --configFile='${ms_config_file}' --outputFile='${om_generated_path}'/'${model_name}'' >> "${run_converter_log_file}"
        ./converter_lite --inputDataFormat=NCHW --inputShape=${input_shape} --fmk=${framework} --modelFile=${models_path}/${model_location}/models/${model_name} --configFile=${ms_config_file} --outputFile=${om_generated_path}/${model_name}
        if [ $? = 0 ]; then
            converter_result='converter ONNX '${model_name}' pass';echo ${converter_result} >> ${run_converter_result_file}
        else
            converter_result='converter ONNX '${model_name}' failed';echo ${converter_result} >> ${run_converter_result_file};exit 1
        fi
      else
        echo "unsupported framework"; return 1
      fi
      cp ./tmp/custom_0.om ${om_generated_path}/${model_name}_lib_original.om || exit 1

      # atc convert model
      if [[ ${framework} == 'CAFFE' ]]; then
        sed -i 's/\[framework\] 6/\[framework\] 0/g' ./${model_name}_atc.cfg
        sed -i '1 i\[weight] '${models_path}/${model_location}/model/${model_name}'.caffemodel' ./${model_name}_atc.cfg
        sed -i '1 i\[model] '${models_path}/${model_location}/model/${model_name}'.prototxt' ./${model_name}_atc.cfg
      elif [[ ${framework} == 'ONNX' ]]; then
        sed -i 's/\[framework\] 6/\[framework\] 5/g' ./${model_name}_atc.cfg
        sed -i '1 i\[model] '${models_path}/${model_location}/models/${model_name} ./${model_name}_atc.cfg
      fi
      sed -i '$a \[instruction_name] '${om_generated_path}/${model_name}_atc ./${model_name}_atc.cfg
      ./atc ./${model_name}_atc.cfg
      if [ $? = 0 ]; then
          converter_result='atc '${framework}' '${model_name}' pass';echo ${converter_result} >> ${run_converter_result_file}
      else
          converter_result='atc '${framework}' '${model_name}' failed';echo ${converter_result} >> ${run_converter_result_file}; exit 1
      fi
  done < ${models_3403_cfg}
}

# Run converter for DPICO models on x86 platform:
function Run_Converter() {
    cd ${x86_path} || exit 1
    tar -zxf mindspore-lite-${version}-linux-x64.tar.gz || exit 1
    cd ${x86_path}/mindspore-lite-${version}-linux-x64/ || exit 1

    #  atc tool
    cp ${dpico_atc_path}/pico_mapper_0924/bin/atc ./ || exit 1
    chmod +x atc

    cp tools/converter/converter/converter_lite ./ || exit 1
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:./tools/converter/lib/:./runtime/lib/:${dpico_atc_path}/pico_mapper_0924/lib:${dpico_atc_path}/protobuf-3.13.0/lib:${dpico_atc_path}/opencv-4.5.2/lib
    chmod +x ./tools/benchmark/benchmark

    echo ' ' > ${run_converter_log_file}
    rm -rf ${om_generated_path}
    mkdir -p ${om_generated_path}
    chmod +x converter_lite

    Run_Convert_MODELS 'ONNX' ${models_onnx_3403_config}
    Run_convert_onnx_status=$?
    if [[ ${Run_convert_onnx_status} = 0 ]];then
        echo "Run convert onnx success"
    else
        echo "Run convert onnx failed"
        exit 1
    fi

    Run_Convert_MODELS 'CAFFE' ${models_caffe_3403_config}
    Run_convert_caffe_status=$?
    if [[ ${Run_convert_caffe_status} = 0 ]];then
        echo "Run convert caffe success"
    else
        echo "Run convert caffe failed"
        exit 1
    fi
}

function Run_Func_Sim() {
  models_3403_cfg=$1
  while read line; do
      dpico_line_info=${line}
      if [[ $dpico_line_info == \#* || $dpico_line_info == "" ]]; then
        continue
      fi
      model_info=`echo ${dpico_line_info}|awk -F ' ' '{print $2}'`
      model_name=${model_info%%;*}
      input_num=`echo ${dpico_line_info}|awk -F ' ' '{print $3}'`
      input_files=''
      if [[ $input_num != 1 ]]; then
        for i in $(seq 1 $input_num)
        do
          cp ${models_path}'/input_output/input/'${model_name}'.ms.bin_'${i}* ${model_name}'_'${i}'.ms.bin' || exit 1
          input_files=$input_files${model_name}'_'${i}'.ms.bin,'
        done
      else
        cp ${models_path}/input_output/input/${model_name}.ms.bin* ${model_name}'.ms.bin' || exit 1
        input_files=${model_name}'.ms.bin'
      fi
      # generate dump files
      rm -rf ${om_generated_path}/dump_output
      ./func_sim -m ./${model_name}_lib_original.om -i ${input_files} -a
      if [ $? -ne 0 ]; then
        simulation_3403_result='func_sim '${model_name}' failed';echo ${simulation_3403_result} >> ${run_simulation_result_file}; exit 1
      fi
      ./func_sim -m ./${model_name}_atc_original.om -i ${input_files} -a
      if [ $? -ne 0 ]; then
        simulation_3403_result='func_sim '${model_name}' failed';echo ${simulation_3403_result} >> ${run_simulation_result_file}; exit 1
      fi

      # compare dump files
      ls ./dump_output/*lib*/batch_0/layer/*report* || exit 1
      ls ./dump_output/*atc*/batch_0/layer/*report* || exit 1
      lib_files_cnt=$(ls ./dump_output/*lib*/batch_0/layer/*report* | wc -l)
      atc_files_cnt=$(ls ./dump_output/*atc*/batch_0/layer/*report* | wc -l)
      if [[ $lib_files_cnt -ne $atc_files_cnt ]]; then
        echo "generated report files is not equal"; exit 1
      fi
      is_file_equal=1
      for i in $(seq 0 $input_num)
      do
        cmp -s ./dump_output/*lib*/batch_0/layer/*report_0_${i}_*.float ./dump_output/*atc*/batch_0/layer/*report_0_${i}_*.float || is_file_equal=0 && break
      done
      if [[ ${is_file_equal} == 1 ]]; then
        simulation_3403_result='simulation '${model_name}' pass';echo ${simulation_3403_result} >> ${run_simulation_result_file}
      else
        simulation_3403_result='simulation '${model_name}' failed';echo ${simulation_3403_result} >> ${run_simulation_result_file}; exit 1
      fi
  done < ${models_3403_cfg}
}

# Run benchmark on 3403:
function Run_Simulation() {
  cd ${om_generated_path} || exit 1
  cp ${dpico_atc_path}/simulation/func_sim ./ || exit 1
  chmod +x func_sim

  Run_Func_Sim  ${models_onnx_3403_config}
  Run_func_sim_status=$?
  if [[ ${Run_func_sim_status} = 0 ]];then
      echo "Run func_sim onnx success"
  else
      echo "Run func_sim onnx failed"
      exit 1
  fi

  Run_Func_Sim  ${models_caffe_3403_config}
  Run_func_sim_status=$?
  if [[ ${Run_func_sim_status} = 0 ]];then
      echo "Run func_sim caffe success"
  else
      echo "Run func_sim caffe failed"
      exit 1
  fi

}

basepath=$(pwd)
echo ${basepath}

while getopts "r:m:e:l:" opt; do
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
        l)
            level=${OPTARG}
            echo "level is ${OPTARG}"
            ;;
        ?)
        echo "unknown para"
        exit 1;;
    esac
done

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

# get sdk path
if [ "${HISI_SDK_PATH}" ]; then
    hisi_sdk=${HISI_SDK_PATH}
else
    echo "HISI_SDK_PATH env not found"
    exit 1
fi

dpico_atc_path=${hisi_sdk}/sd3403_sdk/dpico_atc_adapter
x86_path=${release_path}/centos_x86

# Set version
cd ${x86_path}
file_name=$(ls *-linux-x64.tar.gz)
IFS="-" read -r -a file_name_array <<< "$file_name"
version=${file_name_array[2]}
cd -

# Set filepath
config_folder="config_level0"
if [[ ${level} == "level1" ]]; then
    config_folder="config_level1"
fi
models_caffe_3403_config=${basepath}/../${config_folder}/models_caffe_3403_simulation.cfg
models_onnx_3403_config=${basepath}/../${config_folder}/models_onnx_3403_simulation.cfg

# Set om generated path
om_generated_path=${basepath}/om_generated

# Write converter result to temp file
run_converter_log_file=${basepath}/run_converter_log.txt
echo ' ' > ${run_converter_log_file}

run_converter_result_file=${basepath}/run_converter_result.txt
echo ' ' > ${run_converter_result_file}

# Run converter
echo "start Run converter for dpico models..."
Run_Converter &
Run_converter_PID=$!
sleep 1

wait ${Run_converter_PID}
Run_converter_status=$?
if [[ ${Run_converter_status} = 0 ]];then
    echo "Run converter for dpico models success"
    Print_Converter_Result
else
    echo "Run converter for dpico models failed"
    cat ${run_converter_log_file}
    Print_Converter_Result
    exit 1
fi

# Write benchmark result to temp file
run_simulation_result_file=${basepath}/run_simulation_3403_result.txt
rm ${run_simulation_result_file}
echo ' ' > ${run_simulation_result_file}

if [[ $backend == "all" || $backend == "simulation_sd3403" ]]; then
    # Run funcsim
    Run_Simulation &
    Run_Simulation_PID=$!
    sleep 1
fi

if [[ $backend == "all" || $backend == "simulation_sd3403" ]]; then
    wait ${Run_Simulation_PID}
    Run_Simulation_status=$?
    if [[ ${Run_Simulation_status} != 0 ]];then
        echo "Run_simulation_3403 failed"
        isFailed=1
    else
        echo "Run_simulation_3403 success"
        isFailed=0
    fi
    MS_PRINT_TESTCASE_END_MSG
    cat ${run_simulation_result_file}
    MS_PRINT_TESTCASE_END_MSG
fi

if [[ $isFailed == 1 ]]; then
    exit 1
fi
exit 0
