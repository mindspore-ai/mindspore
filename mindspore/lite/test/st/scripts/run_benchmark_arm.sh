#!/bin/bash

# Run converter on x86 platform:
function Run_Converter() {
    # Unzip x86 runtime and converter
    cd ${x86_path} || exit 1
    tar -zxf mindspore-lite-${version}-linux-x64.tar.gz || exit 1
    cd ${x86_path}/mindspore-lite-${version}-linux-x64/ || exit 1

    cp tools/converter/converter/converter_lite ./ || exit 1
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:./tools/converter/lib/:./tools/converter/third_party/glog/lib

    rm -rf ${ms_models_path}
    mkdir -p ${ms_models_path}

    # Convert tf models:
    while read line; do
        model_name=${line%;*}
        if [[ $model_name == \#* ]]; then
          continue
        fi
        echo ${model_name} >> "${run_converter_log_file}"
        echo './converter_lite  --fmk=TF --modelFile='${models_path}'/'${model_name}' --outputFile='${ms_models_path}'/'${model_name}'' >> "${run_converter_log_file}"
        ./converter_lite  --fmk=TF --modelFile=$models_path/${model_name} --outputFile=${ms_models_path}/${model_name}
        if [ $? = 0 ]; then
            converter_result='converter tf '${model_name}' pass';echo ${converter_result} >> ${run_converter_result_file}
        else
            converter_result='converter tf '${model_name}' failed';echo ${converter_result} >> ${run_converter_result_file};return 1
        fi
    done < ${models_tf_config}

    # Convert tflite models:
    while read line; do
        model_name=${line}
        if [[ $model_name == \#* ]]; then
          continue
        fi
        echo ${model_name} >> "${run_converter_log_file}"
        echo './converter_lite  --fmk=TFLITE --modelFile='${models_path}'/'${model_name}' --outputFile='${ms_models_path}'/'${model_name}'' >> "${run_converter_log_file}"
        ./converter_lite  --fmk=TFLITE --modelFile=$models_path/${model_name} --outputFile=${ms_models_path}/${model_name}
        if [ $? = 0 ]; then
            converter_result='converter tflite '${model_name}' pass';echo ${converter_result} >> ${run_converter_result_file}
        else
            converter_result='converter tflite '${model_name}' failed';echo ${converter_result} >> ${run_converter_result_file};return 1
        fi
    done < ${models_tflite_config}

    # Convert caffe models:
    while read line; do
        model_name=${line}
        if [[ $model_name == \#* ]]; then
          continue
        fi
        echo ${model_name} >> "${run_converter_log_file}"
        echo './converter_lite  --fmk=CAFFE --modelFile='${models_path}'/'${model_name}'.prototxt --weightFile='${models_path}'/'${model_name}'.caffemodel --outputFile='${ms_models_path}'/'${model_name}'' >> "${run_converter_log_file}"
        ./converter_lite  --fmk=CAFFE --modelFile=${models_path}/${model_name}.prototxt --weightFile=${models_path}/${model_name}.caffemodel --outputFile=${ms_models_path}/${model_name}
        if [ $? = 0 ]; then
            converter_result='converter caffe '${model_name}' pass';echo ${converter_result} >> ${run_converter_result_file}
        else
            converter_result='converter caffe '${model_name}' failed';echo ${converter_result} >> ${run_converter_result_file};return 1
        fi
    done < ${models_caffe_config}

    # Convert onnx models:
    while read line; do
        model_name=${line%;*}
        if [[ $model_name == \#* ]]; then
          continue
        fi
        echo ${model_name} >> "${run_converter_log_file}"
        echo './converter_lite  --fmk=ONNX --modelFile='${models_path}'/'${model_name}' --outputFile='${ms_models_path}'/'${model_name}'' >> "${run_converter_log_file}"
        ./converter_lite  --fmk=ONNX --modelFile=${models_path}/${model_name} --outputFile=${ms_models_path}/${model_name}
        if [ $? = 0 ]; then
            converter_result='converter onnx '${model_name}' pass';echo ${converter_result} >> ${run_converter_result_file}
        else
            converter_result='converter onnx '${model_name}' failed';echo ${converter_result} >> ${run_converter_result_file};return 1
        fi
    done < ${models_onnx_config}

    # Convert mindspore models:
    while read line; do
        mindspore_line_info=${line}
        if [[ $mindspore_line_info == \#* ]]; then
          continue
        fi
        model_name=`echo ${mindspore_line_info}|awk -F ' ' '{print $1}'`
        accuracy_limit=`echo ${mindspore_line_info}|awk -F ' ' '{print $2}'`
        echo ${model_name} >> "${run_converter_log_file}"
        echo './converter_lite  --fmk=MINDIR --modelFile='${models_path}'/'${model_name}' --outputFile='${ms_models_path}'/'${model_name}'' >> "${run_converter_log_file}"
        ./converter_lite  --fmk=MINDIR --modelFile=${models_path}/${model_name} --outputFile=${ms_models_path}/${model_name}
        if [ $? = 0 ]; then
            converter_result='converter mindspore '${model_name}' pass';echo ${converter_result} >> ${run_converter_result_file}
        else
            converter_result='converter mindspore '${model_name}' failed';echo ${converter_result} >> ${run_converter_result_file};return 1
        fi
    done < ${models_mindspore_config}

    # Convert mindspore quant train models:
    while read line; do
        model_name=${line}
        if [[ $model_name == \#* ]]; then
          continue
        fi
        echo ${model_name}'_train' >> "${run_converter_log_file}"
        echo './converter_lite  --fmk=MINDIR --modelFile='${models_path}'/'${model_name}' --outputFile='${ms_models_path}'/'${model_name}'_train  --trainModel=true' >> "${run_converter_log_file}"
        ./converter_lite  --fmk=MINDIR --modelFile=${models_path}/${model_name} --outputFile=${ms_models_path}/${model_name}'_train' --trainModel=true
        if [ $? = 0 ]; then
            converter_result='converter mindspore '${model_name}'_train pass';echo ${converter_result} >> ${run_converter_result_file}
        else
            converter_result='converter mindspore '${model_name}'_train failed';echo ${converter_result} >> ${run_converter_result_file};return 1
        fi
    done < ${models_mindspore_train_config}

    rm -rf ${ms_train_models_path}
    mkdir -p ${ms_train_models_path}
    # Convert mindspore train models:
    while read line; do
        LFS=" " read -r -a line_array <<< ${line}
        WEIGHT_QUANT=""
        model_prefix=${line_array[0]}'_train'
        model_name=${line_array[0]}'_train'
        if [[ $model_name == \#* ]]; then
          continue
        fi
        if [[ "${line_array[1]}" == "weight_quant" ]]; then
            WEIGHT_QUANT="--quantType=WeightQuant --bitNum=8 --quantWeightSize=0 --quantWeightChannel=0"
            model_name=${line_array[0]}'_train_quant'
        fi

        echo ${model_name} >> "${run_converter_log_file}"
        echo './converter_lite  --fmk=MINDIR --modelFile='${train_models_path}'/'${model_prefix}'.mindir --outputFile='${ms_train_models_path}'/'${model_name}' --trainModel=true' ${WEIGHT_QUANT} >> "${run_converter_log_file}"
        ./converter_lite --fmk=MINDIR --modelFile=${train_models_path}/${model_prefix}.mindir --outputFile=${ms_train_models_path}/${model_name} --trainModel=true ${WEIGHT_QUANT}
        if [ $? = 0 ]; then
            converter_result='converter mindspore_train '${model_name}' pass';echo ${converter_result} >> ${run_converter_result_file}
        else
            converter_result='converter mindspore_train '${model_name}' failed';echo ${converter_result} >> ${run_converter_result_file};return 1
        fi
    done < ${models_ms_train_config}

    # Convert TFLite PostTraining models:
    while read line; do
        posttraining_line_info=${line}
        if [[ $posttraining_line_info == \#* ]]; then
          continue
        fi
        model_name=`echo ${posttraining_line_info}|awk -F ' ' '{print $1}'`
        accuracy_limit=`echo ${posttraining_line_info}|awk -F ' ' '{print $2}'`
        echo ${model_name} >> "${run_converter_log_file}"
        echo 'convert mode name: '${model_name}' begin.'
        echo './converter_lite  --fmk=TFLITE --modelFile='${models_path}'/'${model_name}' --outputFile='${ms_models_path}'/'${model_name}_posttraining' --quantType=PostTraining --config_file='${models_path}'/'${model_name}'_posttraining.config' >> "${run_converter_log_file}"
        ./converter_lite  --fmk=TFLITE --modelFile=$models_path/${model_name} --outputFile=${ms_models_path}/${model_name}_posttraining --quantType=PostTraining --configFile=${models_path}/${model_name}_posttraining.config
        if [ $? = 0 ]; then
            converter_result='converter post_training '${model_name}' pass';echo ${converter_result} >> ${run_converter_result_file}
        else
            converter_result='converter post_training '${model_name}' failed';echo ${converter_result} >> ${run_converter_result_file};return 1
        fi
    done < ${models_tflite_posttraining_config}

    # Convert Caffe PostTraining models:
    while read line; do
        posttraining_line_info=${line}
        if [[ $posttraining_line_info == \#* ]]; then
          continue
        fi
        model_name=`echo ${posttraining_line_info}|awk -F ' ' '{print $1}'`
        accuracy_limit=`echo ${posttraining_line_info}|awk -F ' ' '{print $2}'`
        echo ${model_name} >> "${run_converter_log_file}"
        echo 'convert mode name: '${model_name}' begin.'
        echo './converter_lite  --fmk=TFLITE --modelFile='${models_path}'/'${model_name}' --outputFile='${ms_models_path}'/'${model_name}_posttraining' --quantType=PostTraining --config_file='${models_path}'/'${model_name}'_posttraining.config' >> "${run_converter_log_file}"
        ./converter_lite  --fmk=CAFFE --modelFile=$models_path/${model_name}.prototxt --weightFile=$models_path/${model_name}.caffemodel --outputFile=${ms_models_path}/${model_name}_posttraining --quantType=PostTraining --configFile=${models_path}/config.${model_name}
        if [ $? = 0 ]; then
            converter_result='converter post_training '${model_name}' pass';echo ${converter_result} >> ${run_converter_result_file}
        else
            converter_result='converter post_training '${model_name}' failed';echo ${converter_result} >> ${run_converter_result_file};return 1
        fi
    done < ${models_caffe_posttraining_config}

    # Convert TFLite AwareTraining models:
    while read line; do
        model_name=${line}
        if [[ $model_name == \#* ]]; then
          continue
        fi
        echo ${model_name} >> "${run_converter_log_file}"
        echo './converter_lite  --fmk=TFLITE --modelFile='${models_path}'/'${model_name}' --outputFile='${ms_models_path}'/'${model_name}' --inputDataType=FLOAT  --outputDataType=FLOAT' >> "${run_converter_log_file}"
        ./converter_lite  --fmk=TFLITE --modelFile=${models_path}/${model_name} --outputFile=${ms_models_path}/${model_name}   --inputDataType=FLOAT  --outputDataType=FLOAT
        if [ $? = 0 ]; then
            converter_result='converter aware_training '${model_name}' pass';echo ${converter_result} >> ${run_converter_result_file}
        else
            converter_result='converter aware_training '${model_name}' failed';echo ${converter_result} >> ${run_converter_result_file};return 1
        fi
    done < ${models_tflite_awaretraining_config}

    # Convert tflite weightquant models:
    while read line; do
        weight_quant_line_info=${line}
        if [[ $weight_quant_line_info == \#* ]]; then
          continue
        fi
        model_name=`echo ${weight_quant_line_info}|awk -F ' ' '{print $1}'`
        echo ${model_name} >> "${run_converter_log_file}"
        echo './converter_lite  --fmk=TFLITE --modelFile='${models_path}'/'${model_name}' --outputFile='${ms_models_path}'/'${model_name}_weightquant'--quantType=WeightQuant --bitNum=8 --quantWeightChannel=0' >> "${run_converter_log_file}"
        ./converter_lite  --fmk=TFLITE --modelFile=$models_path/${model_name} --outputFile=${ms_models_path}/${model_name}_weightquant --quantType=WeightQuant --bitNum=8 --quantWeightChannel=0
        if [ $? = 0 ]; then
            converter_result='converter weight_quant '${model_name}' pass';echo ${converter_result} >> ${run_converter_result_file}
        else
            converter_result='converter weight_quant '${model_name}' failed';echo ${converter_result} >> ${run_converter_result_file};return 1
        fi
    done < ${models_tflite_weightquant_config}

    # Convert caffe weightquant models:
    while read line; do
        weight_quant_line_info=${line}
        if [[ ${weight_quant_line_info} == \#* ]]; then
          continue
        fi
        model_name=`echo ${weight_quant_line_info}|awk -F ' ' '{print $1}'`
        echo ${model_name} >> "${run_converter_log_file}"
        echo './converter_lite  --fmk=CAFFE --modelFile='${models_path}'/'${model_name}'.prototxt --weightFile='${models_path}'/'${model_name}'.caffemodel --outputFile='${ms_models_path}'/'${model_name}_weightquant'  --quantType=WeightQuant --bitNum=8 --quantWeightChannel=0' >> "${run_converter_log_file}"
        ./converter_lite  --fmk=CAFFE --modelFile=${models_path}/${model_name}.prototxt --weightFile=${models_path}/${model_name}.caffemodel --outputFile=${ms_models_path}/${model_name}_weightquant  --quantType=WeightQuant --bitNum=8 --quantWeightChannel=0
        if [ $? = 0 ]; then
            converter_result='converter caffe_weight_quant '${model_name}' pass';echo ${converter_result} >> ${run_converter_result_file}
        else
            converter_result='converter caffe_weight_quant '${model_name}' failed';echo ${converter_result} >> ${run_converter_result_file};return 1
        fi
    done < ${models_caffe_weightquant_config}

    # Convert mindir weightquant models:
    while read line; do
        weight_quant_line_info=${line}
        if [[ $weight_quant_line_info == \#* ]]; then
          continue
        fi
        model_name=`echo ${weight_quant_line_info}|awk -F ' ' '{print $1}'`
        echo ${model_name} >> "${run_converter_log_file}"
        echo './converter_lite  --fmk=MINDIR --modelFile='${models_path}'/'${model_name}' --outputFile='${ms_models_path}'/'${model_name}' --quantType=WeightQuant --bitNum=8 --quantWeightSize=0 --quantWeightChannel=0' >> "${run_converter_log_file}"
        ./converter_lite  --fmk=MINDIR --modelFile=$models_path/${model_name} --outputFile=${ms_models_path}/${model_name}_weightquant --quantType=WeightQuant --bitNum=8 --quantWeightSize=0 --quantWeightChannel=0
        if [ $? = 0 ]; then
            converter_result='converter weight_quant '${model_name}' pass';echo ${converter_result} >> ${run_converter_result_file}
        else
            converter_result='converter weight_quant '${model_name}' failed';echo ${converter_result} >> ${run_converter_result_file};return 1
        fi
    done < ${models_mindspore_weightquant_config}

    # Convert tf weightquant models:
    while read line; do
        weight_quant_line_info=${line}
        if [[ $weight_quant_line_info == \#* ]]; then
          continue
        fi
        model_name=`echo ${weight_quant_line_info}|awk -F ' ' '{print $1}'`
        echo ${model_name} >> "${run_converter_log_file}"
        echo './converter_lite  --fmk=TF --modelFile='${models_path}'/'${model_name}' --outputFile='${ms_models_path}'/'${model_name}'_weightquant' >> "${run_converter_log_file}"
        ./converter_lite  --fmk=TF --modelFile=$models_path/${model_name} --outputFile=${ms_models_path}/${model_name}_weightquant --quantType=WeightQuant --bitNum=8 --quantWeightChannel=0
        if [ $? = 0 ]; then
            converter_result='converter weight_quant '${model_name}' pass';echo ${converter_result} >> ${run_converter_result_file}
        else
            converter_result='converter weight_quant '${model_name}' failed';echo ${converter_result} >> ${run_converter_result_file};return 1
        fi
    done < ${models_tf_weightquant_config}

    # Convert mindir mixbit weightquant models:
    while read line; do
        line_info=${line}
        if [[ $line_info == \#* ]]; then
          continue
        fi
        model_name=`echo ${line_info}|awk -F ' ' '{print $1}'`

        echo ${model_name}'_7bit' >> "${run_converter_log_file}"
        echo './converter_lite  --fmk=MINDIR --modelFile='${models_path}'/'${model_name}' --outputFile='${ms_models_path}'/'${model_name}'_7bit  --quantType=WeightQuant --bitNum=7 --quantWeightSize=0 --quantWeightChannel=0' >> "${run_converter_log_file}"
        ./converter_lite  --fmk=MINDIR --modelFile=${models_path}/${model_name} --outputFile=${ms_models_path}/${model_name}'_7bit' --quantType=WeightQuant --bitNum=7 --quantWeightSize=0 --quantWeightChannel=0
        if [ $? = 0 ]; then
            converter_result='converter mindspore '${model_name}'_7bit pass';echo ${converter_result} >> ${run_converter_result_file}
        else
            converter_result='converter mindspore '${model_name}'_7bit failed';echo ${converter_result} >> ${run_converter_result_file};return 1
        fi
        echo ${model_name}'_9bit' >> "${run_converter_log_file}"
        echo './converter_lite  --fmk=MINDIR --modelFile='${models_path}'/'${model_name}' --outputFile='${ms_models_path}'/'${model_name}'_9bit  --quantType=WeightQuant --bitNum=9 --quantWeightSize=0 --quantWeightChannel=0' >> "${run_converter_log_file}"
        ./converter_lite  --fmk=MINDIR --modelFile=${models_path}/${model_name} --outputFile=${ms_models_path}/${model_name}'_9bit' --quantType=WeightQuant --bitNum=9 --quantWeightSize=0 --quantWeightChannel=0
        if [ $? = 0 ]; then
            converter_result='converter mindspore '${model_name}'_9bit pass';echo ${converter_result} >> ${run_converter_result_file}
        else
            converter_result='converter mindspore '${model_name}'_9bit failed';echo ${converter_result} >> ${run_converter_result_file};return 1
        fi
    done < ${models_mindspore_mixbit_config}

    # Convert models which has multiple inputs:
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
    done < ${models_with_multiple_inputs_config}

    # Convert models which does not need to be cared about the accuracy:
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
    done < ${models_for_process_only_config}

    # Copy fp16 ms models:
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
    done < ${models_onnx_fp16_config}

    while read line; do
        fp16_line_info=${line}
        if [[ $fp16_line_info == \#* ]]; then
          continue
        fi
        model_name=`echo ${fp16_line_info}|awk -F ' ' '{print $1}'`
        echo 'cp '${ms_models_path}'/'${model_name}'.ms' ${ms_models_path}'/'${model_name}'.fp16.ms'
        cp ${ms_models_path}/${model_name}.ms ${ms_models_path}/${model_name}.fp16.ms
        if [ $? = 0 ]; then
            converter_result='converter fp16 '${model_name}' pass';echo ${converter_result} >> ${run_converter_result_file}
        else
            converter_result='converter fp16 '${model_name}' failed';echo ${converter_result} >> ${run_converter_result_file};return 1
        fi
    done < ${models_caffe_fp16_config}

    while read line; do
        fp16_line_info=${line}
        if [[ $fp16_line_info == \#* ]]; then
          continue
        fi
        model_name=`echo ${fp16_line_info}|awk -F ' ' '{print $1}'`
        echo 'cp '${ms_models_path}'/'${model_name}'.ms' ${ms_models_path}'/'${model_name}'.fp16.ms'
        cp ${ms_models_path}/${model_name}.ms ${ms_models_path}/${model_name}.fp16.ms
        if [ $? = 0 ]; then
            converter_result='converter fp16 '${model_name}' pass';echo ${converter_result} >> ${run_converter_result_file}
        else
            converter_result='converter fp16 '${model_name}' failed';echo ${converter_result} >> ${run_converter_result_file};return 1
        fi
    done < ${models_tflite_fp16_config}

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
    done < ${models_tf_fp16_config}

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
    done < ${models_multiple_inputs_fp16_config}
}

function Run_arm64_codegen() {
    echo "ANDROID_NDK: ${ANDROID_NDK}" >> ${run_arm64_fp32_codegen_log_file}
    cd ${arm64_path} || exit 1
    tar -zxf mindspore-lite-${version}-android-aarch64.tar.gz || exit 1
    local PKG_PATH=${arm64_path}/mindspore-lite-${version}-android-aarch64
    local CODEGEN_PATH=${x86_path}/mindspore-lite-${version}-linux-x64/tools/codegen

    rm -rf ${build_path}
    mkdir -p ${build_path}

    # Run tflite converted models:
    while read line; do
        model_name=${line}
        if [[ $model_name == \#* ]]; then
          continue
        fi

        {
            echo "arm64_codegen: ${model_name}"
            echo "${CODEGEN_PATH}/codegen --codePath=${build_path} --modelPath=${ms_models_path}/${model_name}.ms --target=ARM64"
            ${CODEGEN_PATH}/codegen --codePath=${build_path} --modelPath=${ms_models_path}/${model_name}.ms --target=ARM64
        } >> ${run_arm64_fp32_codegen_log_file}

        rm -rf ${build_path}/benchmark
        mkdir -p ${build_path}/benchmark && cd ${build_path}/benchmark || exit 1

        {
            echo "cmake -DCMAKE_BUILD_TYPE=Release
                  -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake
                  -DANDROID_ABI=arm64-v8a
                  -DANDROID_TOOLCHAIN_NAME=aarch64-linux-android-clang
                  -DANDROID_NATIVE_API_LEVEL=19
                  -DPLATFORM_ARM64=ON
                  -DPKG_PATH=${PKG_PATH} ${build_path}/${model_name}"

            cmake -DCMAKE_BUILD_TYPE=Release \
                  -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK}/build/cmake/android.toolchain.cmake" \
                  -DANDROID_ABI="arm64-v8a" \
                  -DANDROID_TOOLCHAIN_NAME="aarch64-linux-android-clang" \
                  -DANDROID_NATIVE_API_LEVEL="19" \
                  -DPLATFORM_ARM64=ON \
                  -DPKG_PATH=${PKG_PATH} ${build_path}/${model_name}

            make -j4
        } >> ${run_arm64_fp32_codegen_log_file}

        rm -rf ${build_path}/codegen_test
        mkdir ${build_path}/codegen_test && cd ${build_path}/codegen_test || exit 1
        cp -a ${build_path}/benchmark/benchmark ${build_path}/codegen_test/benchmark || exit 1
        cp -a ${build_path}/${model_name}/src/net.bin ${build_path}/codegen_test/net.bin || exit 1

        {
            echo 'ls ${build_path}/codegen_test'
            ls ${build_path}/codegen_test
        } >> ${run_arm64_fp32_codegen_log_file}

        # adb push all needed files to the phone
        adb -s ${device_id} push ${build_path}/codegen_test /data/local/tmp/ > adb_push_log.txt

        {
            echo 'cd  /data/local/tmp/codegen_test'
            echo 'chmod 777 benchmark'
            echo 'chmod 777 net.bin'
            echo 'ls'
            echo './benchmark /data/local/tmp/input_output/input/'${model_name}'.ms.bin ./net.bin 1 /data/local/tmp/input_output/output/'${model_name}'.ms.out'
            echo 'cd .. && rm -rf codegen_test'
        } >> ${run_arm64_fp32_codegen_log_file}

        {
            echo 'cd  /data/local/tmp/codegen_test'
            echo 'chmod 777 benchmark'
            echo 'chmod 777 net.bin'
            echo 'ls'
            echo './benchmark /data/local/tmp/input_output/input/'${model_name}'.ms.bin ./net.bin 1 /data/local/tmp/input_output/output/'${model_name}'.ms.out'
            echo 'cd .. && rm -rf codegen_test'
        } > adb_run_cmd.txt

        adb -s ${device_id} shell < adb_run_cmd.txt >> ${run_arm64_fp32_codegen_log_file}
        if [ $? = 0 ]; then
            run_result='arm64_codegen: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='arm64_codegen: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
    done < ${models_codegen_config}

    rm -rf ${build_path}
}

function Run_arm32_codegen() {
    echo "ANDROID_NDK: ${ANDROID_NDK}" >> ${run_arm32_fp32_codegen_log_file}
    cd ${arm32_path} || exit 1
    tar -zxf mindspore-lite-${version}-android-aarch32.tar.gz || exit 1
    local PKG_PATH=${arm32_path}/mindspore-lite-${version}-android-aarch32
    local CODEGEN_PATH=${x86_path}/mindspore-lite-${version}-linux-x64/tools/codegen

    rm -rf ${build_path}
    mkdir -p ${build_path}

    # Run tflite converted models:
    while read line; do
        model_name=${line}
        if [[ $model_name == \#* ]]; then
          continue
        fi

        {
            echo "arm32_codegen: ${model_name}"
            echo "${CODEGEN_PATH}/codegen --codePath=${build_path} --modelPath=${ms_models_path}/${model_name}.ms --target=ARM32A"
            ${CODEGEN_PATH}/codegen --codePath=${build_path} --modelPath=${ms_models_path}/${model_name}.ms --target=ARM32A
        } >> ${run_arm32_fp32_codegen_log_file}

        rm -rf ${build_path}/benchmark
        mkdir -p ${build_path}/benchmark && cd ${build_path}/benchmark || exit 1

        {
            echo "cmake -DCMAKE_BUILD_TYPE=Release \
                  -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake \
                  -DANDROID_ABI=armeabi-v7a \
                  -DANDROID_TOOLCHAIN_NAME=clang \
                  -DANDROID_NATIVE_API_LEVEL=19 \
                  -DPLATFORM_ARM32=ON \
                  -DPKG_PATH=${PKG_PATH} ${build_path}/${model_name}"

            cmake -DCMAKE_BUILD_TYPE=Release \
                  -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK}/build/cmake/android.toolchain.cmake" \
                  -DANDROID_ABI="armeabi-v7a" \
                  -DANDROID_TOOLCHAIN_NAME="clang" \
                  -DANDROID_NATIVE_API_LEVEL="19" \
                  -DPLATFORM_ARM32=ON \
                  -DPKG_PATH=${PKG_PATH} ${build_path}/${model_name}

            make -j4
        } >> ${run_arm32_fp32_codegen_log_file}

        rm -rf ${build_path}/codegen_test
        mkdir ${build_path}/codegen_test && cd ${build_path}/codegen_test || exit 1
        cp -a ${build_path}/benchmark/benchmark ${build_path}/codegen_test/benchmark || exit 1
        cp -a ${build_path}/${model_name}/src/net.bin ${build_path}/codegen_test/net.bin || exit 1

        {
            echo 'ls ${build_path}/codegen_test'
            ls ${build_path}/codegen_test
        } >> ${run_arm32_fp32_codegen_log_file}

        # adb push all needed files to the phone
        adb -s ${device_id} push ${build_path}/codegen_test /data/local/tmp/ > adb_push_log.txt

        {
            echo 'cd  /data/local/tmp/codegen_test'
            echo 'chmod 777 benchmark'
            echo 'chmod 777 net.bin'
            echo 'ls'
            echo './benchmark /data/local/tmp/input_output/input/'${model_name}'.ms.bin ./net.bin 1 /data/local/tmp/input_output/output/'${model_name}'.ms.out'
            echo 'cd .. && rm -rf codegen_test'
        } >> ${run_arm32_fp32_codegen_log_file}

        {
            echo 'cd  /data/local/tmp/codegen_test'
            echo 'chmod 777 benchmark'
            echo 'chmod 777 net.bin'
            echo 'ls'
            echo './benchmark /data/local/tmp/input_output/input/'${model_name}'.ms.bin ./net.bin 1 /data/local/tmp/input_output/output/'${model_name}'.ms.out'
            echo 'cd .. && rm -rf codegen_test'
        } > adb_run_cmd.txt

        adb -s ${device_id} shell < adb_run_cmd.txt >> ${run_arm32_fp32_codegen_log_file}
        if [ $? = 0 ]; then
            run_result='arm32_codegen: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='arm32_codegen: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
    done < ${models_codegen_config}

    rm -rf ${build_path}
}

# Run on arm64 platform:
function Run_arm64() {
    cd ${arm64_path} || exit 1
    tar -zxf mindspore-lite-${version}-android-aarch64.tar.gz || exit 1

    # If build with minddata, copy the minddata related libs
    cd ${benchmark_test_path} || exit 1
    if [ -f ${arm64_path}/mindspore-lite-${version}-android-aarch64/runtime/lib/libminddata-lite.so ]; then
        cp -a ${arm64_path}/mindspore-lite-${version}-android-aarch64/runtime/lib/libminddata-lite.so ${benchmark_test_path}/libminddata-lite.so || exit 1
        cp -a ${arm64_path}/mindspore-lite-${version}-android-aarch64/runtime/third_party/libjpeg-turbo/lib/libjpeg.so* ${benchmark_test_path}/ || exit 1
        cp -a ${arm64_path}/mindspore-lite-${version}-android-aarch64/runtime/third_party/libjpeg-turbo/lib/libturbojpeg.so* ${benchmark_test_path}/ || exit 1
    fi
    cp -a ${arm64_path}/mindspore-lite-${version}-android-aarch64/runtime/third_party/hiai_ddk/lib/libhiai.so ${benchmark_test_path}/libhiai.so || exit 1
    cp -a ${arm64_path}/mindspore-lite-${version}-android-aarch64/runtime/third_party/hiai_ddk/lib/libhiai_ir.so ${benchmark_test_path}/libhiai_ir.so || exit 1
    cp -a ${arm64_path}/mindspore-lite-${version}-android-aarch64/runtime/third_party/hiai_ddk/lib/libhiai_ir_build.so ${benchmark_test_path}/libhiai_ir_build.so || exit 1

    cp -a ${arm64_path}/mindspore-lite-${version}-android-aarch64/runtime/lib/libmindspore-lite.so ${benchmark_test_path}/libmindspore-lite.so || exit 1
    cp -a ${arm64_path}/mindspore-lite-${version}-android-aarch64/runtime/lib/libmindspore-lite-train.so ${benchmark_test_path}/libmindspore-lite-train.so || exit 1
    cp -a ${arm64_path}/mindspore-lite-${version}-android-aarch64/tools/benchmark/benchmark ${benchmark_test_path}/benchmark || exit 1
    cp -a ${arm64_path}/mindspore-lite-${version}-android-aarch64/tools/benchmark_train/benchmark_train ${benchmark_train_test_path}/benchmark_train || exit 1

    # adb push all needed files to the phone
    adb -s ${device_id} push ${benchmark_test_path} /data/local/tmp/ > adb_push_log.txt 
    cp -a ${benchmark_test_path}/lib*so* ${benchmark_train_test_path}/
    adb -s ${device_id} push ${benchmark_train_test_path} /data/local/tmp/ > adb_push_log.txt 

    # run adb ,run session ,check the result:
    echo 'cd  /data/local/tmp/benchmark_test' > adb_cmd.txt
    echo 'cp  /data/local/tmp/libc++_shared.so ./' >> adb_cmd.txt
    echo 'chmod 777 benchmark' >> adb_cmd.txt
    echo 'cd /data/local/tmp/benchmark_train_test' >> adb_cmd.txt
    echo 'cp  /data/local/tmp/libc++_shared.so ./' >> adb_cmd.txt
    echo 'chmod 777 benchmark_train' >> adb_cmd.txt

    adb -s ${device_id} shell < adb_cmd.txt

    # Run compatibility test models:
    while read line; do
        compat_line_info=${line}
        if [[ $compat_line_info == \#* ]]; then
          continue
        fi
        model_info=`echo ${compat_line_info}|awk -F ' ' '{print $1}'`
        accuracy_limit=`echo ${compat_line_info}|awk -F ' ' '{print $2}'`
        model_name=${model_info%%;*}
        length=${#model_name}
        input_shapes=${model_info:length+1}
        echo "---------------------------------------------------------" >> "${run_arm64_fp32_log_file}"
        echo "compat run: ${model_name}, accuracy limit:${accuracy_limit}" >> "${run_arm64_fp32_log_file}"

        echo 'cd  /data/local/tmp/benchmark_test' > adb_run_cmd.txt
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test' >> adb_run_cmd.txt
        if [ -z "$accuracy_limit" ]; then
          echo './benchmark --modelFile='${model_name}'.ms --warmUpLoopCount=1 --loopCount=2 --inputShapes='${input_shapes} >> "${run_arm64_fp32_log_file}"
          echo './benchmark --modelFile='${model_name}'.ms --warmUpLoopCount=1 --loopCount=2 --inputShapes='${input_shapes} >> adb_run_cmd.txt
        else
          echo './benchmark --modelFile='${model_name}'.ms --inDataFile=/data/local/tmp/input_output/input/'${model_name}'.ms.bin --benchmarkDataFile=/data/local/tmp/input_output/output/'${model_name}'.ms.out --accuracyThreshold='${accuracy_limit} ' --inputShapes='${input_shapes} >> adb_run_cmd.txt
        fi
        cat adb_run_cmd.txt >> "${run_arm64_fp32_log_file}"
        adb -s ${device_id} shell < adb_run_cmd.txt >> "${run_arm64_fp32_log_file}"
        if [ $? = 0 ]; then
            run_result='arm64: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='arm64: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
    done < ${models_compatibility_config}

    # Run tf converted models:
    while read line; do
        model_name=${line%;*}
        length=${#model_name}
        input_shapes=${line:length+1}
        if [[ $model_name == \#* ]]; then
          continue
        fi
        echo ${model_name} >> "${run_arm64_fp32_log_file}"
        echo 'cd  /data/local/tmp/benchmark_test' > adb_run_cmd.txt
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --modelFile='${model_name}'.ms --inDataFile=/data/local/tmp/input_output/input/'${model_name}'.ms.bin --inputShapes='${input_shapes}' --benchmarkDataFile=/data/local/tmp/input_output/output/'${model_name}'.ms.out' >> "${run_arm64_fp32_log_file}"
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --modelFile='${model_name}'.ms --inDataFile=/data/local/tmp/input_output/input/'${model_name}'.ms.bin --inputShapes='${input_shapes}' --benchmarkDataFile=/data/local/tmp/input_output/output/'${model_name}'.ms.out' >> adb_run_cmd.txt
        adb -s ${device_id} shell < adb_run_cmd.txt >> "${run_arm64_fp32_log_file}"
        if [ $? = 0 ]; then
            run_result='arm64: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='arm64: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
        # run benchmark test without clib data
        echo ${model_name} >> "${run_arm64_fp32_log_file}"
        echo 'cd  /data/local/tmp/benchmark_test' > adb_run_cmd.txt
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --modelFile='${model_name}'.ms --inputShapes='${input_shapes}' --warmUpLoopCount=1 --loopCount=2' >> "{run_arm64_fp32_log_file}"
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --modelFile='${model_name}'.ms --inputShapes='${input_shapes}' --warmUpLoopCount=1 --loopCount=2' >> adb_run_cmd.txt
        adb -s ${device_id} shell < adb_run_cmd.txt >> "${run_arm64_fp32_log_file}"
        if [ $? = 0 ]; then
            run_result='arm64: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='arm64: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
    done < ${models_tf_config}

    # Run tflite converted models:
    while read line; do
        model_name=${line}
        if [[ $model_name == \#* ]]; then
          continue
        fi
        echo ${model_name} >> "${run_arm64_fp32_log_file}"
        echo 'cd  /data/local/tmp/benchmark_test' > adb_run_cmd.txt
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --modelFile='${model_name}'.ms --inDataFile=/data/local/tmp/input_output/input/'${model_name}'.ms.bin --benchmarkDataFile=/data/local/tmp/input_output/output/'${model_name}'.ms.out' >> "${run_arm64_fp32_log_file}"
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --modelFile='${model_name}'.ms --inDataFile=/data/local/tmp/input_output/input/'${model_name}'.ms.bin --benchmarkDataFile=/data/local/tmp/input_output/output/'${model_name}'.ms.out' >> adb_run_cmd.txt
        adb -s ${device_id} shell < adb_run_cmd.txt >> "${run_arm64_fp32_log_file}"
        if [ $? = 0 ]; then
            run_result='arm64: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='arm64: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
        # run benchmark test without clib data
        echo 'cd  /data/local/tmp/benchmark_test' > adb_run_cmd.txt
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --modelFile='${model_name}'.ms --warmUpLoopCount=1 --loopCount=2' >> "${run_arm64_fp32_log_file}"
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --modelFile='${model_name}'.ms --warmUpLoopCount=1 --loopCount=2' >> adb_run_cmd.txt
        adb -s ${device_id} shell < adb_run_cmd.txt >> "${run_arm64_fp32_log_file}"
        if [ $? = 0 ]; then
            run_result='arm64: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='arm64: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
    done < ${models_tflite_config}

    # Run caffe converted models:
    while read line; do
        model_name=${line}
        if [[ $model_name == \#* ]]; then
          continue
        fi
        echo ${model_name} >> "${run_arm64_fp32_log_file}"
        echo 'cd  /data/local/tmp/benchmark_test' > adb_run_cmd.txt
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --modelFile='${model_name}'.ms --inDataFile=/data/local/tmp/input_output/input/'${model_name}'.ms.bin --benchmarkDataFile=/data/local/tmp/input_output/output/'${model_name}'.ms.out' >> "${run_arm64_fp32_log_file}"
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --modelFile='${model_name}'.ms --inDataFile=/data/local/tmp/input_output/input/'${model_name}'.ms.bin --benchmarkDataFile=/data/local/tmp/input_output/output/'${model_name}'.ms.out' >> adb_run_cmd.txt
        adb -s ${device_id} shell < adb_run_cmd.txt >> "${run_arm64_fp32_log_file}"
        if [ $? = 0 ]; then
            run_result='arm64: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='arm64: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
        # run benchmark test without clib data
        echo ${model_name} >> "${run_arm64_fp32_log_file}"
        echo 'cd  /data/local/tmp/benchmark_test' > adb_run_cmd.txt
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --modelFile='${model_name}'.ms --warmUpLoopCount=1 --loopCount=2' >> "${run_arm64_fp32_log_file}"
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --modelFile='${model_name}'.ms --warmUpLoopCount=1 --loopCount=2' >> adb_run_cmd.txt
        adb -s ${device_id} shell < adb_run_cmd.txt >> "${run_arm64_fp32_log_file}"
        if [ $? = 0 ]; then
            run_result='arm64: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='arm64: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
    done < ${models_caffe_config}

    # Run onnx converted models:
    while read line; do
        model_name=${line%;*}
        length=${#model_name}
        input_shapes=${line:length+1}
        if [[ $model_name == \#* ]]; then
          continue
        fi
        echo ${model_name} >> "${run_arm64_fp32_log_file}"
        echo 'cd  /data/local/tmp/benchmark_test' > adb_run_cmd.txt
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --modelFile='${model_name}'.ms --inDataFile=/data/local/tmp/input_output/input/'${model_name}'.ms.bin --inputShapes='${input_shapes}' --benchmarkDataFile=/data/local/tmp/input_output/output/'${model_name}'.ms.out' >> "${run_arm64_fp32_log_file}"
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --modelFile='${model_name}'.ms --inDataFile=/data/local/tmp/input_output/input/'${model_name}'.ms.bin --inputShapes='${input_shapes}' --benchmarkDataFile=/data/local/tmp/input_output/output/'${model_name}'.ms.out' >> adb_run_cmd.txt
        adb -s ${device_id} shell < adb_run_cmd.txt >> "${run_arm64_fp32_log_file}"
        if [ $? = 0 ]; then
            run_result='arm64: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='arm64: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
        # run benchmark test without clib data
        echo ${model_name} >> "${run_arm64_fp32_log_file}"
        echo 'cd  /data/local/tmp/benchmark_test' > adb_run_cmd.txt
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --modelFile='${model_name}'.ms --inputShapes='${input_shapes}' --warmUpLoopCount=1 --loopCount=2' >> "{run_arm64_fp32_log_file}"
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --modelFile='${model_name}'.ms --inputShapes='${input_shapes}' --warmUpLoopCount=1 --loopCount=2' >> adb_run_cmd.txt
        adb -s ${device_id} shell < adb_run_cmd.txt >> "${run_arm64_fp32_log_file}"
        if [ $? = 0 ]; then
            run_result='arm64: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='arm64: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
    done < ${models_onnx_config}

    # Run mindir converted models:
    while read line; do
        mindspore_line_info=${line}
        if [[ $mindspore_line_info == \#* ]]; then
          continue
        fi
        model_name=`echo ${mindspore_line_info}|awk -F ' ' '{print $1}'`
        accuracy_limit=`echo ${mindspore_line_info}|awk -F ' ' '{print $2}'`
        echo "mindspore run: ${model_name}, accuracy limit:${accuracy_limit}" >> "${run_arm64_fp32_log_file}"
        echo 'cd  /data/local/tmp/benchmark_test' > adb_run_cmd.txt
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --modelFile='${model_name}'.ms --inDataFile=/data/local/tmp/input_output/input/'${model_name}'.ms.bin --benchmarkDataFile=/data/local/tmp/input_output/output/'${model_name}'.ms.out --accuracyThreshold='${accuracy_limit} >> "${run_arm64_fp32_log_file}"
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --modelFile='${model_name}'.ms --inDataFile=/data/local/tmp/input_output/input/'${model_name}'.ms.bin --benchmarkDataFile=/data/local/tmp/input_output/output/'${model_name}'.ms.out --accuracyThreshold='${accuracy_limit} >> adb_run_cmd.txt
        adb -s ${device_id} shell < adb_run_cmd.txt >> "${run_arm64_fp32_log_file}"
        if [ $? = 0 ]; then
            run_result='arm64: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='arm64: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
        # run benchmark test without clib data
        echo ${model_name} >> "${run_arm64_fp32_log_file}"
        echo 'cd /data/local/tmp/benchmark_test' > adb_run_cmd.txt
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --modelFile='${model_name}'.ms --inDataFile=/data/local/tmp/input_output/input/'${model_name}'.ms.bin --warmUpLoopCount=1 --loopCount=2' >> "{run_arm64_fp32_log_file}"
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --modelFile='${model_name}'.ms --inDataFile=/data/local/tmp/input_output/input/'${model_name}'.ms.bin --warmUpLoopCount=1 --loopCount=2' >> adb_run_cmd.txt
        adb -s ${device_id} shell < adb_run_cmd.txt >> "${run_arm64_fp32_log_file}"
        if [ $? = 0 ]; then
            run_result='arm64: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='arm64: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
    done < ${models_mindspore_config}

    # Run mindir converted train models:
    while read line; do
        model_name=${line}
        if [[ $model_name == \#* ]]; then
          continue
        fi
        echo ${model_name}'_train' >> "${run_arm64_fp32_log_file}"
        echo 'cd /data/local/tmp/benchmark_test' > adb_run_cmd.txt
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --modelFile='${model_name}'_train.ms --inDataFile=/data/local/tmp/input_output/input/'${model_name}'.ms.bin --benchmarkDataFile=/data/local/tmp/input_output/output/'${model_name}'.train.ms.out --accuracyThreshold=1.5' >> "${run_arm64_fp32_log_file}"
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --modelFile='${model_name}'_train.ms --inDataFile=/data/local/tmp/input_output/input/'${model_name}'.ms.bin --benchmarkDataFile=/data/local/tmp/input_output/output/'${model_name}'.train.ms.out --accuracyThreshold=1.5' >> adb_run_cmd.txt
        adb -s ${device_id} shell < adb_run_cmd.txt >> "${run_arm64_fp32_log_file}"
        if [ $? = 0 ]; then
            run_result='arm64: '${model_name}'_train pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='arm64: '${model_name}'_train failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
        # run benchmark test without clib data
        echo ${model_name} >> "${run_arm64_fp32_log_file}"
        echo 'cd /data/local/tmp/benchmark_test' > adb_run_cmd.txt
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --modelFile='${model_name}'_train.ms --warmUpLoopCount=1 --loopCount=2' >> "{run_arm64_fp32_log_file}"
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --modelFile='${model_name}'_train.ms --warmUpLoopCount=1 --loopCount=2' >> adb_run_cmd.txt
        adb -s ${device_id} shell < adb_run_cmd.txt >> "${run_arm64_fp32_log_file}"
        if [ $? = 0 ]; then
            run_result='arm64: '${model_name}'_train pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='arm64: '${model_name}'_train failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
    done < ${models_mindspore_train_config}

    # Run tflite post training quantization converted models:
    while read line; do
        posttraining_line_info=${line}
        if [[ $posttraining_line_info == \#* ]]; then
          continue
        fi
        model_name=`echo ${posttraining_line_info}|awk -F ' ' '{print $1}'`
        accuracy_limit=`echo ${posttraining_line_info}|awk -F ' ' '{print $2}'`
        echo ${model_name} >> "${run_arm64_fp32_log_file}"
        echo 'cd  /data/local/tmp/benchmark_test' > adb_run_cmd.txt
        if [[ $model_name == "transformer_20200831_encoder_fp32.tflite" ]]; then
            echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --modelFile='${model_name}'_posttraining.ms --inDataFile=/data/local/tmp/input_output/input/encoder_buffer_in_0-35.bin --benchmarkDataFile=/data/local/tmp/input_output/output/'${model_name}'_posttraining.ms.out' --accuracyThreshold=${accuracy_limit} >> "${run_arm64_fp32_log_file}"
            echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --modelFile='${model_name}'_posttraining.ms --inDataFile=/data/local/tmp/input_output/input/encoder_buffer_in_35.bin,/data/local/tmp/input_output/input/encoder_buffer_in_0.bin,/data/local/tmp/input_output/input/encoder_buffer_in_1.bin,/data/local/tmp/input_output/input/encoder_buffer_in_4.bin,/data/local/tmp/input_output/input/encoder_buffer_in_2.bin,/data/local/tmp/input_output/input/encoder_buffer_in_3.bin,/data/local/tmp/input_output/input/encoder_buffer_in_7.bin,/data/local/tmp/input_output/input/encoder_buffer_in_5.bin,/data/local/tmp/input_output/input/encoder_buffer_in_6.bin,/data/local/tmp/input_output/input/encoder_buffer_in_10.bin,/data/local/tmp/input_output/input/encoder_buffer_in_8.bin,/data/local/tmp/input_output/input/encoder_buffer_in_9.bin,/data/local/tmp/input_output/input/encoder_buffer_in_11.bin,/data/local/tmp/input_output/input/encoder_buffer_in_12.bin,/data/local/tmp/input_output/input/encoder_buffer_in_15.bin,/data/local/tmp/input_output/input/encoder_buffer_in_13.bin,/data/local/tmp/input_output/input/encoder_buffer_in_14.bin,/data/local/tmp/input_output/input/encoder_buffer_in_18.bin,/data/local/tmp/input_output/input/encoder_buffer_in_16.bin,/data/local/tmp/input_output/input/encoder_buffer_in_17.bin,/data/local/tmp/input_output/input/encoder_buffer_in_21.bin,/data/local/tmp/input_output/input/encoder_buffer_in_19.bin,/data/local/tmp/input_output/input/encoder_buffer_in_20.bin,/data/local/tmp/input_output/input/encoder_buffer_in_22.bin,/data/local/tmp/input_output/input/encoder_buffer_in_23.bin,/data/local/tmp/input_output/input/encoder_buffer_in_26.bin,/data/local/tmp/input_output/input/encoder_buffer_in_24.bin,/data/local/tmp/input_output/input/encoder_buffer_in_25.bin,/data/local/tmp/input_output/input/encoder_buffer_in_29.bin,/data/local/tmp/input_output/input/encoder_buffer_in_27.bin,/data/local/tmp/input_output/input/encoder_buffer_in_28.bin,/data/local/tmp/input_output/input/encoder_buffer_in_32.bin,/data/local/tmp/input_output/input/encoder_buffer_in_30.bin,/data/local/tmp/input_output/input/encoder_buffer_in_31.bin,/data/local/tmp/input_output/input/encoder_buffer_in_33.bin,/data/local/tmp/input_output/input/encoder_buffer_in_34.bin --benchmarkDataFile=/data/local/tmp/input_output/output/'${model_name}'_posttraining.ms.out' --accuracyThreshold=${accuracy_limit} >> adb_run_cmd.txt
        fi
        if [[ $model_name == "transformer_20200831_decoder_fp32.tflite" ]]; then
            echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --modelFile='${model_name}'_posttraining.ms --inDataFile=/data/local/tmp/input_output/input/decoder_buffer_in_0-10.bin --benchmarkDataFile=/data/local/tmp/input_output/output/'${model_name}'_posttraining.ms.out' --accuracyThreshold=${accuracy_limit} >> "${run_arm64_fp32_log_file}"
            echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --modelFile='${model_name}'_posttraining.ms --inDataFile=/data/local/tmp/input_output/input/decoder_buffer_in_9.bin,/data/local/tmp/input_output/input/decoder_buffer_in_2.bin,/data/local/tmp/input_output/input/decoder_buffer_in_0.bin,/data/local/tmp/input_output/input/decoder_buffer_in_1.bin,/data/local/tmp/input_output/input/decoder_buffer_in_5.bin,/data/local/tmp/input_output/input/decoder_buffer_in_3.bin,/data/local/tmp/input_output/input/decoder_buffer_in_4.bin,/data/local/tmp/input_output/input/decoder_buffer_in_8.bin,/data/local/tmp/input_output/input/decoder_buffer_in_6.bin,/data/local/tmp/input_output/input/decoder_buffer_in_7.bin,/data/local/tmp/input_output/input/decoder_buffer_in_10.bin --benchmarkDataFile=/data/local/tmp/input_output/output/'${model_name}'_posttraining.ms.out' --accuracyThreshold=${accuracy_limit} >> adb_run_cmd.txt
        fi
        adb -s ${device_id} shell < adb_run_cmd.txt >> "${run_arm64_fp32_log_file}"
        if [ $? = 0 ]; then
            run_result='arm64: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='arm64: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
        # run benchmark test without clib data
        echo ${model_name} >> "${run_arm64_fp32_log_file}"
        echo 'cd  /data/local/tmp/benchmark_test' > adb_run_cmd.txt
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --modelFile='${model_name}'_posttraining.ms --warmUpLoopCount=1 --loopCount=2' >> "${run_arm64_fp32_log_file}"
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --modelFile='${model_name}'_posttraining.ms --warmUpLoopCount=1 --loopCount=2' >> adb_run_cmd.txt
        adb -s ${device_id} shell < adb_run_cmd.txt >> "${run_arm64_fp32_log_file}"
        if [ $? = 0 ]; then
            run_result='arm64: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='arm64: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
    done < ${models_tflite_posttraining_config}

    # Run caffe posttraining models:
    while read line; do
        posttraining_line_info=${line}
        if [[ $posttraining_line_info == \#* ]]; then
          continue
        fi
        model_name=`echo ${posttraining_line_info}|awk -F ' ' '{print $1}'`
        accuracy_limit=`echo ${posttraining_line_info}|awk -F ' ' '{print $2}'`
        echo ${model_name} >> "${run_arm64_fp32_log_file}"
        echo 'cd  /data/local/tmp/benchmark_test' > adb_run_cmd.txt
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --modelFile='${model_name}'_posttraining.ms --inDataFile=/data/local/tmp/input_output/input/'${model_name}'_posttraining.ms.bin --benchmarkDataFile=/data/local/tmp/input_output/output/'${model_name}'_posttraining.ms.out' --accuracyThreshold=${accuracy_limit} >> "${run_arm64_fp32_log_file}"
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --modelFile='${model_name}'_posttraining.ms --inDataFile=/data/local/tmp/input_output/input/'${model_name}'_posttraining.ms.bin --benchmarkDataFile=/data/local/tmp/input_output/output/'${model_name}'_posttraining.ms.out' --accuracyThreshold=${accuracy_limit} >> adb_run_cmd.txt
        adb -s ${device_id} shell < adb_run_cmd.txt >> "${run_arm64_fp32_log_file}"
        if [ $? = 0 ]; then
            run_result='arm64: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='arm64: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
        # run benchmark test without clib data
        echo ${model_name} >> "${run_arm64_fp32_log_file}"
        echo 'cd  /data/local/tmp/benchmark_test' > adb_run_cmd.txt
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --modelFile='${model_name}'_posttraining.ms --warmUpLoopCount=1 --loopCount=2' >> "${run_arm64_fp32_log_file}"
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --modelFile='${model_name}'_posttraining.ms --warmUpLoopCount=1 --loopCount=2' >> adb_run_cmd.txt
        adb -s ${device_id} shell < adb_run_cmd.txt >> "${run_arm64_fp32_log_file}"
        if [ $? = 0 ]; then
            run_result='arm64: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='arm64: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
    done < ${models_caffe_posttraining_config}

    # Run tflite aware training quantization converted models:
    while read line; do
        model_name=${line}
        if [[ $model_name == \#* ]]; then
          continue
        fi
        echo ${model_name} >> "${run_arm64_fp32_log_file}"
        echo 'cd  /data/local/tmp/benchmark_test' > adb_run_cmd.txt
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --modelFile='${model_name}'.ms --inDataFile=/data/local/tmp/input_output/input/'${model_name}'.ms.bin --benchmarkDataFile=/data/local/tmp/input_output/output/'${model_name}'.ms.out' >> "${run_arm64_fp32_log_file}"
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --modelFile='${model_name}'.ms --inDataFile=/data/local/tmp/input_output/input/'${model_name}'.ms.bin --benchmarkDataFile=/data/local/tmp/input_output/output/'${model_name}'.ms.out' >> adb_run_cmd.txt
        adb -s ${device_id} shell < adb_run_cmd.txt >> "${run_arm64_fp32_log_file}"
        if [ $? = 0 ]; then
            run_result='arm64: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='arm64: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
        # run benchmark test without clib data
        echo ${model_name} >> "${run_arm64_fp32_log_file}"
        echo 'cd  /data/local/tmp/benchmark_test' > adb_run_cmd.txt
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --modelFile='${model_name}'.ms --warmUpLoopCount=1 --loopCount=2' >> "${run_arm64_fp32_log_file}"
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --modelFile='${model_name}'.ms --warmUpLoopCount=1 --loopCount=2' >> adb_run_cmd.txt
        adb -s ${device_id} shell < adb_run_cmd.txt >> "${run_arm64_fp32_log_file}"
        if [ $? = 0 ]; then
            run_result='arm64_awq: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='arm64_awq: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
    done < ${models_tflite_awaretraining_config}

    # Run mindir weightquant converted train models:
    while read line; do
        weight_quant_line_info=${line}
        if [[ $weight_quant_line_info == \#* ]]; then
          continue
        fi
        model_name=`echo ${weight_quant_line_info}|awk -F ' ' '{print $1}'`
        accuracy_limit=`echo ${weight_quant_line_info}|awk -F ' ' '{print $2}'`
        echo ${model_name} >> "${run_arm64_fp32_log_file}"
        echo 'cd /data/local/tmp/benchmark_test' > adb_run_cmd.txt
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --modelFile='${model_name}'_weightquant.ms --inDataFile=/data/local/tmp/input_output/input/'${model_name}'.ms.bin --benchmarkDataFile=/data/local/tmp/input_output/output/'${model_name}'.ms.out --loopCount=1 --accuracyThreshold='${accuracy_limit} >> "${run_arm64_fp32_log_file}"
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --modelFile='${model_name}'_weightquant.ms --inDataFile=/data/local/tmp/input_output/input/'${model_name}'.ms.bin --benchmarkDataFile=/data/local/tmp/input_output/output/'${model_name}'.ms.out --loopCount=1 --accuracyThreshold='${accuracy_limit} >> adb_run_cmd.txt
        adb -s ${device_id} shell < adb_run_cmd.txt >> "${run_arm64_fp32_log_file}"
        if [ $? = 0 ]; then
            run_result='arm64: '${model_name}'_weightquant pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='arm64: '${model_name}'_weightquant failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
    done < ${models_mindspore_weightquant_config}

    # Run converted models which has multiple inputs:
    while read line; do
        model_name=${line%%;*}
        if [[ $model_name == \#* ]]; then
          continue
        fi
        model_name=`echo ${line} | awk -F ';' '{print $1}'`
        input_num=`echo ${line} | awk -F ';' '{print $2}'`
        input_shapes=`echo ${line} | awk -F ';' '{print $3}'`
        input_files=''
        output_file=''
        data_path="/data/local/tmp/input_output/"
        for i in $(seq 1 $input_num)
        do
          input_files=$input_files${data_path}'input/'$model_name'.ms.bin_'$i','
        done
        output_file=${data_path}'output/'${model_name}'.ms.out'
        if [[ ${model_name##*.} == "caffemodel" ]]; then
          model_name=${model_name%.*}
        fi
        echo ${model_name} >> "${run_arm64_fp32_log_file}"
        echo 'cd  /data/local/tmp/benchmark_test' > adb_run_cmd.txt
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --modelFile='${model_name}'.ms --inDataFile='${input_files}' --inputShapes='${input_shapes}' --benchmarkDataFile='${output_file} >> "${run_arm64_fp32_log_file}"
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --modelFile='${model_name}'.ms --inDataFile='${input_files}' --inputShapes='${input_shapes}' --benchmarkDataFile='${output_file} >> adb_run_cmd.txt
        adb -s ${device_id} shell < adb_run_cmd.txt >> "${run_arm64_fp32_log_file}"
        if [ $? = 0 ]; then
            run_result='arm64: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='arm64: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
        # run benchmark test without clib data
        echo ${model_name} >> "${run_arm64_fp32_log_file}"
        echo 'cd  /data/local/tmp/benchmark_test' > adb_run_cmd.txt
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --modelFile='${model_name}'.ms --inDataFile='${input_files}' --inputShapes='${input_shapes} ' --warmUpLoopCount=1 --loopCount=2' >> "${run_arm64_fp32_log_file}"
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --modelFile='${model_name}'.ms --inDataFile='${input_files}' --inputShapes='${input_shapes} ' --warmUpLoopCount=1 --loopCount=2' >> adb_run_cmd.txt
        adb -s ${device_id} shell < adb_run_cmd.txt >> "${run_arm64_fp32_log_file}"
        if [ $? = 0 ]; then
            run_result='arm64: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='arm64: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
    done < ${models_with_multiple_inputs_config}

    # Run converted models which does not need to be cared about the accuracy:
    while read line; do
        model_name=${line%%;*}
        if [[ $model_name == \#* ]]; then
          continue
        fi
        model_name=`echo ${line} | awk -F ';' '{print $1}'`
        input_num=`echo ${line} | awk -F ';' '{print $2}'`
        input_shapes=`echo ${line} | awk -F ';' '{print $3}'`
        if [[ ${model_name##*.} == "caffemodel" ]]; then
          model_name=${model_name%.*}
        fi
        echo ${model_name} >> "${run_arm64_fp32_log_file}"
        echo 'cd  /data/local/tmp/benchmark_test' > adb_run_cmd.txt
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --modelFile='${model_name}'.ms --inputShapes='${input_shapes}' --warmUpLoopCount=0 --loopCount=2' >> "${run_arm64_fp32_log_file}"
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --modelFile='${model_name}'.ms --inputShapes='${input_shapes}' --warmUpLoopCount=0 --loopCount=2' >> adb_run_cmd.txt
        adb -s ${device_id} shell < adb_run_cmd.txt >> "${run_arm64_fp32_log_file}"
        if [ $? = 0 ]; then
            run_result='arm64: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='arm64: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
    done < ${models_for_process_only_config}
    fail=0
    # Run mindir converted train models:
    tmp_dir=/data/local/tmp/benchmark_train_test
    while read line; do
        LFS=" " read -r -a line_array <<< ${line}
        model_prefix=${line_array[0]}
        model_name=${line_array[0]}'_train'
        accuracy_limit=0.5
        enable_fp16="false"
        virtual_batch="false"
        suffix_print=""
        if [[ $model_name == \#* ]]; then
            continue
        fi
        if [[ "${line_array[1]}" == "weight_quant" ]]; then
            model_name=${line_array[0]}'_train_quant'
            accuracy_limit=${line_array[2]}
        elif [[ "${line_array[1]}" == "fp16" ]]; then
            enable_fp16="true"
            suffix_print="_fp16"
            accuracy_limit=${line_array[2]}
        elif [[ "${line_array[1]}" == "vb" ]]; then
            virtual_batch="true"
            suffix_print="_virtual_batch"
            accuracy_limit=${line_array[2]}
        fi
        export_file="${tmp_dir}/${model_name}_tod"
        inference_file="${tmp_dir}/${model_name}_infer"
        # run benchmark_train test with clib data
        echo ${model_name} >> "${run_arm64_fp32_log_file}"
        adb -s ${device_id} push ${train_io_path}/${model_prefix}_input*.bin ${train_io_path}/${model_prefix}_output*.bin /data/local/tmp/benchmark_train_test >> adb_push_log.txt
        echo 'cd /data/local/tmp/benchmark_train_test' > adb_run_cmd.txt
        echo 'chmod 777 benchmark_train' >> adb_run_cmd.txt
        adb -s ${device_id} shell < adb_run_cmd.txt >> ${run_arm64_fp32_log_file}
        echo "rm -f ${export_file}* ${inference_file}*" >> ${run_arm64_fp32_log_file}
        echo "rm -f ${export_file}* ${inference_file}*" >> adb_run_cmd.txt
        adb -s ${device_id} shell < adb_run_cmd.txt >> ${run_arm64_fp32_log_file}
        adb_cmd=$(cat <<-ENDM
        export LD_LIBRARY_PATH=./:/data/local/tmp/:/data/local/tmp/benchmark_train_test;./benchmark_train \
        --epochs=${epoch_num} \
        --modelFile=${model_name}.ms \
        --inDataFile=${tmp_dir}/${model_prefix}_input \
        --expectedDataFile=${tmp_dir}/${model_prefix}_output \
        --numThreads=${threads} \
        --accuracyThreshold=${accuracy_limit} \
        --enableFp16=${enable_fp16} \
        --inferenceFile=${inference_file} \
        --exportFile=${export_file} \
        --virtualBatch=${virtual_batch}
ENDM
        )
        echo "${adb_cmd}" >> ${run_arm64_fp32_log_file}
        echo "${adb_cmd}" >> adb_run_cmd.txt
        adb -s ${device_id} shell < adb_run_cmd.txt >> ${run_arm64_fp32_log_file}
        # TODO: change to arm_type
        if [ $? = 0 ]; then
            run_result='arm64_train: '${model_name}''${suffix_print}' pass'; echo ${run_result} >> ${run_benchmark_train_result_file}
        else
            run_result='arm64_train: '${model_name}''${suffix_print}' failed'; echo ${run_result} >> ${run_benchmark_train_result_file};
            fail=1
        fi
    done < ${models_ms_train_config}
    return ${fail}
}

# Run on arm32 platform:
function Run_arm32() {
    cd ${arm32_path} || exit 1
    tar -zxf mindspore-lite-${version}-android-aarch32.tar.gz || exit 1

    # If build with minddata, copy the minddata related libs
    cd ${benchmark_test_path} || exit 1
    if [ -f ${arm32_path}/mindspore-lite-${version}-android-aarch32/runtime/lib/libminddata-lite.so ]; then
        cp -a ${arm32_path}/mindspore-lite-${version}-android-aarch32/runtime/lib/libminddata-lite.so ${benchmark_test_path}/libminddata-lite.so || exit 1
        cp -a ${arm32_path}/mindspore-lite-${version}-android-aarch32/runtime/third_party/libjpeg-turbo/lib/libturbojpeg.so* ${benchmark_test_path}/ || exit 1
        cp -a ${arm32_path}/mindspore-lite-${version}-android-aarch32/runtime/third_party/libjpeg-turbo/lib/libjpeg.so* ${benchmark_test_path}/ || exit 1
    fi
    cp -a ${arm32_path}/mindspore-lite-${version}-android-aarch32/runtime/third_party/hiai_ddk/lib/libhiai.so ${benchmark_test_path}/libhiai.so || exit 1
    cp -a ${arm32_path}/mindspore-lite-${version}-android-aarch32/runtime/third_party/hiai_ddk/lib/libhiai_ir.so ${benchmark_test_path}/libhiai_ir.so || exit 1
    cp -a ${arm32_path}/mindspore-lite-${version}-android-aarch32/runtime/third_party/hiai_ddk/lib/libhiai_ir_build.so ${benchmark_test_path}/libhiai_ir_build.so || exit 1
    cp -a ${arm32_path}/mindspore-lite-${version}-android-aarch32/runtime/lib/libmindspore-lite.so ${benchmark_test_path}/libmindspore-lite.so || exit 1
    cp -a ${arm32_path}/mindspore-lite-${version}-android-aarch32/runtime/lib/libmindspore-lite-train.so ${benchmark_test_path}/libmindspore-lite-train.so || exit 1
    cp -a ${arm32_path}/mindspore-lite-${version}-android-aarch32/tools/benchmark/benchmark ${benchmark_test_path}/benchmark || exit 1
    cp -a ${arm32_path}/mindspore-lite-${version}-android-aarch32/tools/benchmark_train/benchmark_train ${benchmark_train_test_path}/benchmark_train || exit 1

    # adb push all needed files to the phone
    adb -s ${device_id} push ${benchmark_test_path} /data/local/tmp/ > adb_push_log.txt
    # train ms file may be same,need push diff folder
    cp -a ${benchmark_test_path}/lib*so* ${benchmark_train_test_path}/
    adb -s ${device_id} push ${benchmark_train_test_path} /data/local/tmp/ > adb_push_log.txt

    # run adb ,run session ,check the result:
    echo 'cd  /data/local/tmp/benchmark_test' > adb_cmd.txt
    echo 'cp  /data/local/tmp/arm32/libc++_shared.so ./' >> adb_cmd.txt
    echo 'chmod 777 benchmark' >> adb_cmd.txt
    echo 'cd /data/local/tmp/benchmark_train_test' >> adb_cmd.txt
    echo 'cp /data/local/tmp/arm32/libc++_shared.so ./' >> adb_cmd.txt
    echo 'chmod 777 benchmark_train' >> adb_cmd.txt

    adb -s ${device_id} shell < adb_cmd.txt

    # Run fp32 models:
    while read line; do
        model_name=${line}
        if [[ $model_name == \#* ]]; then
          continue
        fi
        echo ${model_name} >> "${run_arm32_log_file}"
        echo 'cd  /data/local/tmp/benchmark_test' > adb_run_cmd.txt
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --modelFile='${model_name}'.ms --inDataFile=/data/local/tmp/input_output/input/'${model_name}'.ms.bin --benchmarkDataFile=/data/local/tmp/input_output/output/'${model_name}'.ms.out' >> "${run_arm32_log_file}"
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --modelFile='${model_name}'.ms --inDataFile=/data/local/tmp/input_output/input/'${model_name}'.ms.bin --benchmarkDataFile=/data/local/tmp/input_output/output/'${model_name}'.ms.out' >> adb_run_cmd.txt
        adb -s ${device_id} shell < adb_run_cmd.txt >> "${run_arm32_log_file}"
        if [ $? = 0 ]; then
            run_result='arm32: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='arm32: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
        # run benchmark test without clib data
        echo ${model_name} >> "${run_arm32_log_file}"
        echo 'cd  /data/local/tmp/benchmark_test' > adb_run_cmd.txt
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --modelFile='${model_name}'.ms --warmUpLoopCount=1 --loopCount=2' >> "${run_arm32_log_file}"
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --modelFile='${model_name}'.ms --warmUpLoopCount=1 --loopCount=2' >> adb_run_cmd.txt
        adb -s ${device_id} shell < adb_run_cmd.txt >> "${run_arm32_log_file}"
        if [ $? = 0 ]; then
            run_result='arm32: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='arm32: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
    done < ${models_arm32_config}

    fail=0
    # Run mindir converted train models:
    tmp_dir=/data/local/tmp/benchmark_train_test
    while read line; do
        LFS=" " read -r -a line_array <<< ${line}
        model_prefix=${line_array[0]}
        model_name=${line_array[0]}'_train'
        accuracy_limit=0.5
        if [[ $model_name == \#* ]]; then
            continue
        fi
        if [[ "${line_array[1]}" == "weight_quant" ]]; then
            model_name=${line_array[0]}'_train_quant'
            accuracy_limit=${line_array[2]}
        elif [[ "${line_array[1]}" != "" ]]; then
            continue
        fi
        export_file="${tmp_dir}/${model_name}_tod"
        inference_file="${tmp_dir}/${model_name}_infer"
        # run benchmark_train test without clib data
        echo ${model_name} >> "${run_arm32_log_file}"
        adb -s ${device_id} push ${train_io_path}/${model_prefix}_input*.bin ${train_io_path}/${model_prefix}_output*.bin  /data/local/tmp/benchmark_train_test >> adb_push_log.txt
        echo 'cd /data/local/tmp/benchmark_train_test' > adb_run_cmd.txt
        echo 'chmod 777 benchmark_train' >> adb_run_cmd.txt
        adb -s ${device_id} shell < adb_run_cmd.txt >> ${run_arm32_log_file}
        echo "rm -f ${export_file}* ${inference_file}*" >> ${run_arm32_log_file}
        echo "rm -f ${export_file}* ${inference_file}*" >> adb_run_cmd.txt
        adb -s ${device_id} shell < adb_run_cmd.txt >> ${run_arm32_log_file}
        adb_cmd=$(cat <<-ENDM
        export LD_LIBRARY_PATH=./:/data/local/tmp/:/data/local/tmp/benchmark_train_test;./benchmark_train \
        --epochs=${epoch_num} \
        --modelFile=${model_name}.ms \
        --inDataFile=${tmp_dir}/${model_prefix}_input \
        --expectedDataFile=${tmp_dir}/${model_prefix}_output \
        --numThreads=${threads} \
        --accuracyThreshold=${accuracy_limit} \
        --inferenceFile=${inference_file} \
        --exportFile=${export_file}
ENDM
        )
        echo "${adb_cmd}" >> ${run_arm32_log_file}
        echo "${adb_cmd}" >> adb_run_cmd.txt
        adb -s ${device_id} shell < adb_run_cmd.txt >> ${run_arm32_log_file}
        # TODO: change to arm_type
        if [ $? = 0 ]; then
            run_result='arm32_train: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_train_result_file}
        else
            run_result='arm32_train: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_train_result_file};
            fail=1
        fi
    done < ${models_ms_train_config}
    return ${fail}
}

# Run on arm64-fp16 platform:
function Run_arm64_fp16() {
    cd ${arm64_path} || exit 1
    tar -zxf mindspore-lite-${version}-android-aarch64.tar.gz || exit 1

    # If build with minddata, copy the minddata related libs
    cd ${benchmark_test_path} || exit 1
    if [ -f ${arm64_path}/mindspore-lite-${version}-android-aarch64/runtime/lib/libminddata-lite.so ]; then
        cp -a ${arm64_path}/mindspore-lite-${version}-android-aarch64/runtime/lib/libminddata-lite.so ${benchmark_test_path}/libminddata-lite.so || exit 1
    fi
    cp -a ${arm64_path}/mindspore-lite-${version}-android-aarch64/runtime/third_party/hiai_ddk/lib/libhiai.so ${benchmark_test_path}/libhiai.so || exit 1
    cp -a ${arm64_path}/mindspore-lite-${version}-android-aarch64/runtime/third_party/hiai_ddk/lib/libhiai_ir.so ${benchmark_test_path}/libhiai_ir.so || exit 1
    cp -a ${arm64_path}/mindspore-lite-${version}-android-aarch64/runtime/third_party/hiai_ddk/lib/libhiai_ir_build.so ${benchmark_test_path}/libhiai_ir_build.so || exit 1

    cp -a ${arm64_path}/mindspore-lite-${version}-android-aarch64/runtime/lib/libmindspore-lite.so ${benchmark_test_path}/libmindspore-lite.so || exit 1
    cp -a ${arm64_path}/mindspore-lite-${version}-android-aarch64/tools/benchmark/benchmark ${benchmark_test_path}/benchmark || exit 1

    # adb push all needed files to the phone
    adb -s ${device_id} push ${benchmark_test_path} /data/local/tmp/ > adb_push_log.txt

    # run adb ,run session ,check the result:
    echo 'cd  /data/local/tmp/benchmark_test' > adb_cmd.txt
    echo 'cp  /data/local/tmp/libc++_shared.so ./' >> adb_cmd.txt
    echo 'chmod 777 benchmark' >> adb_cmd.txt

    adb -s ${device_id} shell < adb_cmd.txt

    # Run fp16 converted models:
    while read line; do
        fp16_line_info=${line}
        if [[ $fp16_line_info == \#* ]]; then
          continue
        fi
        model_info=`echo ${fp16_line_info}|awk -F ' ' '{print $1}'`
        accuracy_limit=`echo ${fp16_line_info}|awk -F ' ' '{print $2}'`
        model_name=${model_info%%;*}
        length=${#model_name}
        input_shapes=${model_info:length+1}
        echo "---------------------------------------------------------" >> "${run_arm64_fp16_log_file}"
        echo "fp16 run: ${model_name}, accuracy limit:${accuracy_limit}" >> "${run_arm64_fp16_log_file}"

        echo 'cd  /data/local/tmp/benchmark_test' > adb_run_cmd.txt
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test' >> adb_run_cmd.txt
        if [[ $accuracy_limit == "-1" ]]; then
          echo './benchmark --modelFile='${model_name}'.fp16.ms --inDataFile=/data/local/tmp/input_output/input/'${model_name}'.ms.bin --enableFp16=true --inputShapes='${input_shapes} >> adb_run_cmd.txt
        else
          echo './benchmark --modelFile='${model_name}'.fp16.ms --inDataFile=/data/local/tmp/input_output/input/'${model_name}'.ms.bin --benchmarkDataFile=/data/local/tmp/input_output/output/'${model_name}'.ms.out --enableFp16=true --accuracyThreshold='${accuracy_limit} ' --inputShapes='${input_shapes} >> adb_run_cmd.txt
        fi
        cat adb_run_cmd.txt >> "${run_arm64_fp16_log_file}"
        adb -s ${device_id} shell < adb_run_cmd.txt >> "${run_arm64_fp16_log_file}"
        if [ $? = 0 ]; then
            run_result='arm64_fp16: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='arm64_fp16: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
    done < ${models_onnx_fp16_config}

    while read line; do
        fp16_line_info=${line}
        if [[ $fp16_line_info == \#* ]]; then
          continue
        fi
        model_name=`echo ${fp16_line_info}|awk -F ' ' '{print $1}'`
        accuracy_limit=`echo ${fp16_line_info}|awk -F ' ' '{print $2}'`
        echo "---------------------------------------------------------" >> "${run_arm64_fp16_log_file}"
        echo "fp16 run: ${model_name}, accuracy limit:${accuracy_limit}" >> "${run_arm64_fp16_log_file}"

        echo 'cd  /data/local/tmp/benchmark_test' > adb_run_cmd.txt
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test' >> adb_run_cmd.txt
        echo './benchmark --modelFile='${model_name}'.fp16.ms --inDataFile=/data/local/tmp/input_output/input/'${model_name}'.ms.bin --benchmarkDataFile=/data/local/tmp/input_output/output/'${model_name}'.ms.out --enableFp16=true --accuracyThreshold='${accuracy_limit} >> adb_run_cmd.txt

        cat adb_run_cmd.txt >> "${run_arm64_fp16_log_file}"
        adb -s ${device_id} shell < adb_run_cmd.txt >> "${run_arm64_fp16_log_file}"
        if [ $? = 0 ]; then
            run_result='arm64_fp16: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='arm64_fp16: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
    done < ${models_caffe_fp16_config}

    while read line; do
        fp16_line_info=${line}
        if [[ $fp16_line_info == \#* ]]; then
          continue
        fi
        model_name=`echo ${fp16_line_info}|awk -F ' ' '{print $1}'`
        accuracy_limit=`echo ${fp16_line_info}|awk -F ' ' '{print $2}'`
        echo "---------------------------------------------------------" >> "${run_arm64_fp16_log_file}"
        echo "fp16 run: ${model_name}, accuracy limit:${accuracy_limit}" >> "${run_arm64_fp16_log_file}"

        echo 'cd  /data/local/tmp/benchmark_test' > adb_run_cmd.txt
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test' >> adb_run_cmd.txt
        echo './benchmark --modelFile='${model_name}'.fp16.ms --inDataFile=/data/local/tmp/input_output/input/'${model_name}'.ms.bin --benchmarkDataFile=/data/local/tmp/input_output/output/'${model_name}'.ms.out --enableFp16=true --accuracyThreshold='${accuracy_limit} >> adb_run_cmd.txt

        cat adb_run_cmd.txt >> "${run_arm64_fp16_log_file}"
        adb -s ${device_id} shell < adb_run_cmd.txt >> "${run_arm64_fp16_log_file}"
        if [ $? = 0 ]; then
            run_result='arm64_fp16: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='arm64_fp16: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
    done < ${models_tflite_fp16_config}

    # Run fp16 converted models:
    while read line; do
        fp16_line_info=${line}
        if [[ $fp16_line_info == \#* ]]; then
          continue
        fi
        model_info=`echo ${fp16_line_info}|awk -F ' ' '{print $1}'`
        accuracy_limit=`echo ${fp16_line_info}|awk -F ' ' '{print $2}'`
        model_name=${model_info%%;*}
        length=${#model_name}
        input_shapes=${model_info:length+1}
        echo "---------------------------------------------------------" >> "${run_arm64_fp16_log_file}"
        echo "fp16 run: ${model_name}, accuracy limit:${accuracy_limit}" >> "${run_arm64_fp16_log_file}"

        echo 'cd  /data/local/tmp/benchmark_test' > adb_run_cmd.txt
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test' >> adb_run_cmd.txt
        if [[ $accuracy_limit == "-1" ]]; then
          echo './benchmark --modelFile='${model_name}'.fp16.ms --inDataFile=/data/local/tmp/input_output/input/'${model_name}'.ms.bin --enableFp16=true --inputShapes='${input_shapes} >> adb_run_cmd.txt
        else
          echo './benchmark --modelFile='${model_name}'.fp16.ms --inDataFile=/data/local/tmp/input_output/input/'${model_name}'.ms.bin --benchmarkDataFile=/data/local/tmp/input_output/output/'${model_name}'.ms.out --enableFp16=true --accuracyThreshold='${accuracy_limit} ' --inputShapes='${input_shapes} >> adb_run_cmd.txt
        fi
        cat adb_run_cmd.txt >> "${run_arm64_fp16_log_file}"
        adb -s ${device_id} shell < adb_run_cmd.txt >> "${run_arm64_fp16_log_file}"
        if [ $? = 0 ]; then
            run_result='arm64_fp16: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='arm64_fp16: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
    done < ${models_tf_fp16_config}

    # Run converted models which has multiple inputs in fp16 mode:
    while read line; do
        fp16_line_info=${line}
        if [[ $fp16_line_info == \#* ]]; then
          continue
        fi
        model_info=`echo ${fp16_line_info}|awk -F ' ' '{print $1}'`
        accuracy_limit=`echo ${fp16_line_info}|awk -F ' ' '{print $2}'`
        model_name=`echo ${model_info}|awk -F ';' '{print $1}'`
        input_num=`echo ${model_info} | awk -F ';' '{print $2}'`
        input_shapes=`echo ${model_info} | awk -F ';' '{print $3}'`
        input_files=''
        output_file=''
        data_path="/data/local/tmp/input_output/"
        for i in $(seq 1 $input_num)
        do
          input_files=$input_files${data_path}'input/'$model_name'.ms.bin_'$i','
        done
        output_file=${data_path}'output/'${model_name}'.ms.out'
        if [[ ${model_name##*.} == "caffemodel" ]]; then
          model_name=${model_name%.*}
        fi
        echo "---------------------------------------------------------" >> "${run_arm64_fp16_log_file}"
        echo "fp16 run: ${model_name}, accuracy limit:${accuracy_limit}" >> "${run_arm64_fp16_log_file}"

        echo 'cd  /data/local/tmp/benchmark_test' > adb_run_cmd.txt
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test' >> adb_run_cmd.txt
        echo './benchmark --modelFile='${model_name}'.fp16.ms --inDataFile='${input_files}' --inputShapes='${input_shapes}' --benchmarkDataFile='${output_file} '--enableFp16=true --accuracyThreshold='${accuracy_limit} >> adb_run_cmd.txt

        cat adb_run_cmd.txt >> "${run_arm64_fp16_log_file}"
        adb -s ${device_id} shell < adb_run_cmd.txt >> "${run_arm64_fp16_log_file}"
        if [ $? = 0 ]; then
            run_result='arm64_fp16: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='arm64_fp16: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
    done < ${models_multiple_inputs_fp16_config}
}

# Run on armv8.2-a32-fp16 platform:
function Run_armv82_a32_fp16() {
    cd ${armv82_path} || exit 1
    tar -zxf mindspore-lite-${version}-android-aarch32.tar.gz || exit 1

    # If build with minddata, copy the minddata related libs
    cd ${benchmark_test_path} || exit 1
    if [ -f ${armv82_path}/mindspore-lite-${version}-android-aarch32/runtime/minddata/lib/libminddata-lite.so ]; then
        cp -a ${armv82_path}/mindspore-lite-${version}-android-aarch32/runtime/minddata/lib/libminddata-lite.so ${benchmark_test_path}/libminddata-lite.so || exit 1
    fi
    cp -a ${armv82_path}/mindspore-lite-${version}-android-aarch32/runtime/third_party/hiai_ddk/lib/libhiai.so ${benchmark_test_path}/libhiai.so || exit 1
    cp -a ${armv82_path}/mindspore-lite-${version}-android-aarch32/runtime/third_party/hiai_ddk/lib/libhiai_ir.so ${benchmark_test_path}/libhiai_ir.so || exit 1
    cp -a ${armv82_path}/mindspore-lite-${version}-android-aarch32/runtime/third_party/hiai_ddk/lib/libhiai_ir_build.so ${benchmark_test_path}/libhiai_ir_build.so || exit 1
    cp -a ${armv82_path}/mindspore-lite-${version}-android-aarch32/runtime/lib/libmindspore-lite.so ${benchmark_test_path}/libmindspore-lite.so || exit 1
    cp -a ${armv82_path}/mindspore-lite-${version}-android-aarch32/tools/benchmark/benchmark ${benchmark_test_path}/benchmark || exit 1

    # adb push all needed files to the phone
    adb -s ${device_id} push ${benchmark_test_path} /data/local/tmp/ > adb_push_log.txt

    # run adb ,run session ,check the result:
    echo 'cd  /data/local/tmp/benchmark_test' > adb_cmd.txt
    echo 'cp  /data/local/tmp/arm32/libc++_shared.so ./' >> adb_cmd.txt
    echo 'chmod 777 benchmark' >> adb_cmd.txt

    adb -s ${device_id} shell < adb_cmd.txt

    # Run fp16 converted models:
    while read line; do
        fp16_line_info=${line}
        column_num=`echo ${fp16_line_info} | awk -F ' ' '{print NF}'`
        if [[ ${fp16_line_info} == \#* || ${column_num} -lt 3 ]]; then
          continue
        fi
        model_info=`echo ${fp16_line_info}|awk -F ' ' '{print $1}'`
        accuracy_limit=`echo ${fp16_line_info}|awk -F ' ' '{print $3}'`
        model_name=${model_info%%;*}
        length=${#model_name}
        input_shapes=${model_info:length+1}
        echo "---------------------------------------------------------" >> "${run_armv82_a32_fp16_log_file}"
        echo "fp16 run: ${model_name}, accuracy limit:${accuracy_limit}" >> "${run_armv82_a32_fp16_log_file}"

        echo 'cd  /data/local/tmp/benchmark_test' > adb_run_cmd.txt
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test' >> adb_run_cmd.txt
        if [[ $accuracy_limit == "-1" ]]; then
          echo './benchmark --modelFile='${model_name}'.fp16.ms --inDataFile=/data/local/tmp/input_output/input/'${model_name}'.ms.bin --enableFp16=true --inputShapes='${input_shapes} >> adb_run_cmd.txt
        else
          echo './benchmark --modelFile='${model_name}'.fp16.ms --inDataFile=/data/local/tmp/input_output/input/'${model_name}'.ms.bin --benchmarkDataFile=/data/local/tmp/input_output/output/'${model_name}'.ms.out --enableFp16=true --accuracyThreshold='${accuracy_limit} ' --inputShapes='${input_shapes} >> adb_run_cmd.txt
        fi
        cat adb_run_cmd.txt >> "${run_armv82_a32_fp16_log_file}"
        adb -s ${device_id} shell < adb_run_cmd.txt >> "${run_armv82_a32_fp16_log_file}"
        if [ $? = 0 ]; then
            run_result='armv82_a32_fp16: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='armv82_a32_fp16: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
    done < ${models_onnx_fp16_config}

    while read line; do
        fp16_line_info=${line}
        column_num=`echo ${fp16_line_info} | awk -F ' ' '{print NF}'`
        if [[ ${fp16_line_info} == \#* || ${column_num} -lt 3 ]]; then
          continue
        fi
        model_name=`echo ${fp16_line_info}|awk -F ' ' '{print $1}'`
        accuracy_limit=`echo ${fp16_line_info}|awk -F ' ' '{print $3}'`
        echo "---------------------------------------------------------" >> "${run_armv82_a32_fp16_log_file}"
        echo "fp16 run: ${model_name}, accuracy limit:${accuracy_limit}" >> "${run_armv82_a32_fp16_log_file}"

        echo 'cd  /data/local/tmp/benchmark_test' > adb_run_cmd.txt
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test' >> adb_run_cmd.txt
        echo './benchmark --modelFile='${model_name}'.fp16.ms --inDataFile=/data/local/tmp/input_output/input/'${model_name}'.ms.bin --benchmarkDataFile=/data/local/tmp/input_output/output/'${model_name}'.ms.out --enableFp16=true --accuracyThreshold='${accuracy_limit} >> adb_run_cmd.txt

        cat adb_run_cmd.txt >> "${run_armv82_a32_fp16_log_file}"
        adb -s ${device_id} shell < adb_run_cmd.txt >> "${run_armv82_a32_fp16_log_file}"
        if [ $? = 0 ]; then
            run_result='armv82_a32_fp16: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='armv82_a32_fp16: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
    done < ${models_caffe_fp16_config}

    while read line; do
        fp16_line_info=${line}
        column_num=`echo ${fp16_line_info} | awk -F ' ' '{print NF}'`
        if [[ ${fp16_line_info} == \#* || ${column_num} -lt 3 ]]; then
          continue
        fi
        model_name=`echo ${fp16_line_info}|awk -F ' ' '{print $1}'`
        accuracy_limit=`echo ${fp16_line_info}|awk -F ' ' '{print $3}'`
        echo "---------------------------------------------------------" >> "${run_armv82_a32_fp16_log_file}"
        echo "fp16 run: ${model_name}, accuracy limit:${accuracy_limit}" >> "${run_armv82_a32_fp16_log_file}"

        echo 'cd  /data/local/tmp/benchmark_test' > adb_run_cmd.txt
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test' >> adb_run_cmd.txt
        echo './benchmark --modelFile='${model_name}'.fp16.ms --inDataFile=/data/local/tmp/input_output/input/'${model_name}'.ms.bin --benchmarkDataFile=/data/local/tmp/input_output/output/'${model_name}'.ms.out --enableFp16=true --accuracyThreshold='${accuracy_limit} >> adb_run_cmd.txt

        cat adb_run_cmd.txt >> "${run_armv82_a32_fp16_log_file}"
        adb -s ${device_id} shell < adb_run_cmd.txt >> "${run_armv82_a32_fp16_log_file}"
        if [ $? = 0 ]; then
            run_result='armv82_a32_fp16: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='armv82_a32_fp16: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
    done < ${models_tflite_fp16_config}

    # Run fp16 converted models:
    while read line; do
        fp16_line_info=${line}
        column_num=`echo ${fp16_line_info} | awk -F ' ' '{print NF}'`
        if [[ ${fp16_line_info} == \#* || ${column_num} -lt 3 ]]; then
          continue
        fi
        model_info=`echo ${fp16_line_info}|awk -F ' ' '{print $1}'`
        accuracy_limit=`echo ${fp16_line_info}|awk -F ' ' '{print $3}'`
        model_name=${model_info%%;*}
        length=${#model_name}
        input_shapes=${model_info:length+1}
        echo "---------------------------------------------------------" >> "${run_armv82_a32_fp16_log_file}"
        echo "fp16 run: ${model_name}, accuracy limit:${accuracy_limit}" >> "${run_armv82_a32_fp16_log_file}"

        echo 'cd  /data/local/tmp/benchmark_test' > adb_run_cmd.txt
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test' >> adb_run_cmd.txt
        if [[ $accuracy_limit == "-1" ]]; then
          echo './benchmark --modelFile='${model_name}'.fp16.ms --inDataFile=/data/local/tmp/input_output/input/'${model_name}'.ms.bin --enableFp16=true --inputShapes='${input_shapes} >> adb_run_cmd.txt
        else
          echo './benchmark --modelFile='${model_name}'.fp16.ms --inDataFile=/data/local/tmp/input_output/input/'${model_name}'.ms.bin --benchmarkDataFile=/data/local/tmp/input_output/output/'${model_name}'.ms.out --enableFp16=true --accuracyThreshold='${accuracy_limit} ' --inputShapes='${input_shapes} >> adb_run_cmd.txt
        fi
        cat adb_run_cmd.txt >> "${run_armv82_a32_fp16_log_file}"
        adb -s ${device_id} shell < adb_run_cmd.txt >> "${run_armv82_a32_fp16_log_file}"
        if [ $? = 0 ]; then
            run_result='armv82_a32_fp16: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='armv82_a32_fp16: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
    done < ${models_tf_fp16_config}

    # Run converted models which has multiple inputs in fp16 mode:
    while read line; do
        fp16_line_info=${line}
        column_num=`echo ${fp16_line_info} | awk -F ' ' '{print NF}'`
        if [[ ${fp16_line_info} == \#* || ${column_num} -lt 3 ]]; then
          continue
        fi
        model_info=`echo ${fp16_line_info}|awk -F ' ' '{print $1}'`
        accuracy_limit=`echo ${fp16_line_info}|awk -F ' ' '{print $3}'`
        model_name=`echo ${model_info}|awk -F ';' '{print $1}'`
        input_num=`echo ${model_info} | awk -F ';' '{print $2}'`
        input_shapes=`echo ${model_info} | awk -F ';' '{print $3}'`
        input_files=''
        output_file=''
        data_path="/data/local/tmp/input_output/"
        for i in $(seq 1 $input_num)
        do
          input_files=$input_files${data_path}'input/'$model_name'.ms.bin_'$i','
        done
        output_file=${data_path}'output/'${model_name}'.ms.out'
        if [[ ${model_name##*.} == "caffemodel" ]]; then
          model_name=${model_name%.*}
        fi
        echo "---------------------------------------------------------" >> "${run_armv82_a32_fp16_log_file}"
        echo "fp16 run: ${model_name}, accuracy limit:${accuracy_limit}" >> "${run_armv82_a32_fp16_log_file}"

        echo 'cd  /data/local/tmp/benchmark_test' > adb_run_cmd.txt
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test' >> adb_run_cmd.txt
        echo './benchmark --modelFile='${model_name}'.fp16.ms --inDataFile='${input_files}' --inputShapes='${input_shapes}' --benchmarkDataFile='${output_file} '--enableFp16=true --accuracyThreshold='${accuracy_limit} >> adb_run_cmd.txt

        cat adb_run_cmd.txt >> "${run_armv82_a32_fp16_log_file}"
        adb -s ${device_id} shell < adb_run_cmd.txt >> "${run_armv82_a32_fp16_log_file}"
        if [ $? = 0 ]; then
            run_result='armv82_a32_fp16: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='armv82_a32_fp16: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
    done < ${models_multiple_inputs_fp16_config}
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

function Print_Benchmark_Train_Result() {
    MS_PRINT_TESTCASE_START_MSG
    while read line; do
        arr=("${line}")
        printf "%-20s %-100s %-7s\n" ${arr[0]} ${arr[1]} ${arr[2]}
    done < ${run_benchmark_train_result_file}
    MS_PRINT_TESTCASE_END_MSG
}

basepath=$(pwd)
echo ${basepath}
#set -e
epoch_num=1
train_models_path=""
train_io_path=""
threads=2
# Example:sh run_benchmark_arm.sh -r /home/temp_test -m /home/temp_test/models -d "8KE5T19620002408" -e arm_cpu
while getopts "r:m:d:e:M:q:i:t:" opt; do
    case ${opt} in
        r)
            release_path=${OPTARG}
            echo "release_path is ${OPTARG}"
            ;;
        m)
            models_path=${OPTARG}
            echo "models_path is ${OPTARG}"
            ;;
        i)
            train_io_path=${OPTARG}
            echo "train_io_path is ${OPTARG}"
            ;;
        M)
            train_models_path=${OPTARG}
            echo "train_models_path is ${models_path}"
            ;;
        d)
            device_id=${OPTARG}
            echo "device_id is ${OPTARG}"
            ;;
        e)
            backend=${OPTARG}
            echo "backend is ${OPTARG}"
            ;;
        t)
            epoch_num=${OPTARG}
            echo "train epoch num is ${epoch_num}"
            ;;
        q)
           threads=${OPTARG}
           echo "threads=${threads}"
           ;;
        ?)
        echo "unknown para"
        exit 1;;
    esac
done

# mkdir train

x86_path=${release_path}/ubuntu_x86
file_name=$(ls ${x86_path}/*linux-x64.tar.gz)
IFS="-" read -r -a file_name_array <<< "$file_name"
version=${file_name_array[2]}

# Set models config filepath
models_tflite_config=${basepath}/../config/models_tflite.cfg
models_tf_config=${basepath}/../config/models_tf.cfg
models_caffe_config=${basepath}/../config/models_caffe.cfg
models_caffe_weightquant_config=${basepath}/../config/models_caffe_weightquant.cfg
models_tflite_awaretraining_config=${basepath}/../config/models_tflite_awaretraining.cfg
models_tflite_posttraining_config=${basepath}/../config/models_tflite_posttraining.cfg
models_caffe_posttraining_config=${basepath}/../config/models_caffe_posttraining.cfg
models_tflite_weightquant_config=${basepath}/../config/models_tflite_weightquant.cfg
models_onnx_config=${basepath}/../config/models_onnx.cfg
models_onnx_fp16_config=${basepath}/../config/models_onnx_fp16.cfg
models_caffe_fp16_config=${basepath}/../config/models_caffe_fp16.cfg
models_tflite_fp16_config=${basepath}/../config/models_tflite_fp16.cfg
models_tf_fp16_config=${basepath}/../config/models_tf_fp16.cfg
models_multiple_inputs_fp16_config=${basepath}/../config/models_with_multiple_inputs_fp16.cfg
models_mindspore_config=${basepath}/../config/models_mindspore.cfg
models_mindspore_train_config=${basepath}/../config/models_mindspore_train.cfg
models_mindspore_mixbit_config=${basepath}/../config/models_mindspore_mixbit.cfg
models_mindspore_weightquant_config=${basepath}/../config/models_mindspore_weightquant.cfg
models_arm32_config=${basepath}/../config/models_arm32.cfg
models_compatibility_config=${basepath}/../config/models_compatibility.cfg
models_with_multiple_inputs_config=${basepath}/../config/models_with_multiple_inputs.cfg
models_for_process_only_config=${basepath}/../config/models_for_process_only.cfg
models_tf_weightquant_config=${basepath}/../config/models_tf_weightquant.cfg
models_codegen_config=${basepath}/../codegen/models_codegen.cfg
models_ms_train_config=${basepath}/../config/models_ms_train.cfg

ms_models_path=${basepath}/ms_models
ms_train_models_path=${basepath}/ms_train_models
rm -rf ${ms_train_models_path}
mkdir -p ${ms_train_models_path}
build_path=${basepath}/codegen_build
logs_path=${basepath}/logs_train
rm -rf ${logs_path}
mkdir -p ${logs_path}
run_benchmark_train_result_file=${logs_path}/run_benchmark_train_result.txt
echo ' ' > ${run_benchmark_train_result_file}

if [[ $train_models_path == "" ]]
then
  echo "train_io path is empty"
  train_models_path="${models_path}/../../models_train"
fi
echo $train_models_path
if [[ $train_io_path == "" ]]
then
  echo "train_io path is empty"
  train_io_path=${train_models_path}/input_output
fi
echo $train_io_path

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

run_arm64_fp32_log_file=${basepath}/run_arm64_fp32_log.txt
echo 'run arm64_fp32 logs: ' > ${run_arm64_fp32_log_file}

run_arm64_fp32_codegen_log_file=${basepath}/run_arm64_fp32_codegen_log.txt
echo 'run arm64_codegen logs: ' > ${run_arm64_fp32_codegen_log_file}

run_arm64_train_fp32_log_file=${basepath}/run_arm64_train_fp32_log.txt
echo 'run arm64_train_fp32 logs: ' > ${run_arm64_train_fp32_log_file}

run_arm32_fp32_codegen_log_file=${basepath}/run_arm32_fp32_codegen_log.txt
echo 'run arm32_codegen logs: ' > ${run_arm32_fp32_codegen_log_file}

run_arm64_fp16_log_file=${basepath}/run_arm64_fp16_log.txt
echo 'run arm64_fp16 logs: ' > ${run_arm64_fp16_log_file}

run_armv82_a32_fp16_log_file=${basepath}/run_armv82_a32_fp16_log.txt
echo 'run arm82_a32_fp16 logs: ' > ${run_armv82_a32_fp16_log_file}

run_arm32_log_file=${basepath}/run_arm32_log.txt
echo 'run arm32 logs: ' > ${run_arm32_log_file}

# Copy the MindSpore models:
echo "Push files to the arm and run benchmark"
benchmark_test_path=${basepath}/benchmark_test
rm -rf ${benchmark_test_path}
mkdir -p ${benchmark_test_path}
benchmark_train_test_path=${basepath}/benchmark_train_test
rm -rf ${benchmark_train_test_path}
mkdir -p ${benchmark_train_test_path}
cp -a ${ms_models_path}/*.ms ${benchmark_test_path} || exit 1
# Copy models converted using old release of mslite converter for compatibility test
cp -a ${models_path}/compatibility_test/*.ms ${benchmark_test_path} || exit 1
cp -a ${ms_train_models_path}/*.ms ${benchmark_train_test_path} || exit 1

backend=${backend:-"all"}
isFailed=0

if [[ $backend == "all" || $backend == "arm64_cpu" || $backend == "arm64_codegen" ]]; then
    # Run on arm64
    arm64_path=${release_path}/android_aarch64
    file_name=$(ls ${arm64_path}/*android-aarch64.tar.gz)
    IFS="-" read -r -a file_name_array <<< "$file_name"
    version=${file_name_array[2]}

    echo "start Run arm64 codegen ..."
    Run_arm64_codegen
    Run_arm64_codegen_status=$?
    sleep 1
fi

if [[ $backend == "all" || $backend == "arm32_cpu" || $backend == "arm32_codegen" ]]; then
    # Run on arm32
    arm32_path=${release_path}/android_aarch32
    file_name=$(ls ${arm32_path}/*android-aarch32.tar.gz)
    IFS="-" read -r -a file_name_array <<< "$file_name"
    version=${file_name_array[2]}

    echo "start Run arm32 codegen ..."
    Run_arm32_codegen
    Run_arm32_codegen_status=$?
    sleep 1
fi

if [[ $backend == "all" || $backend == "arm32_cpu" || $backend == "arm32_fp16" ]]; then
    # Run on armv82-a32-fp16
    armv82_path=${release_path}/android_aarch32
    file_name=$(ls ${armv82_path}/*android-aarch32.tar.gz)
    IFS="-" read -r -a file_name_array <<< "$file_name"
    version=${file_name_array[2]}

    echo "start Run armv82-a32-fp16 ..."
    Run_armv82_a32_fp16
    Run_armv82_a32_fp16_status=$?
    sleep 1
fi

if [[ $backend == "all" || $backend == "arm32_cpu" || $backend == "arm32_fp32" ]]; then
    # Run on arm32
    arm32_path=${release_path}/android_aarch32
    # mv ${arm32_path}/*train-android-aarch32* ./train
    file_name=$(ls ${arm32_path}/*android-aarch32.tar.gz)
    IFS="-" read -r -a file_name_array <<< "$file_name"
    version=${file_name_array[2]}

    echo "start Run arm32 ..."
    Run_arm32
    Run_arm32_status=$?
    sleep 1
fi

if [[ $backend == "all" || $backend == "arm64_cpu" || $backend == "arm64_fp32" ]]; then
    # Run on arm64
    arm64_path=${release_path}/android_aarch64
    # mv ${arm64_path}/*train-android-aarch64* ./train
    file_name=$(ls ${arm64_path}/*android-aarch64.tar.gz)
    IFS="-" read -r -a file_name_array <<< "$file_name"
    version=${file_name_array[2]}

    echo "start Run arm64 ..."
    Run_arm64
    Run_arm64_fp32_status=$?
    sleep 1
fi

if [[ $backend == "all" || $backend == "arm64_cpu" || $backend == "arm64_fp16" ]]; then
    # Run on arm64-fp16
    arm64_path=${release_path}/android_aarch64
    # mv ${arm64_path}/*train-android-aarch64* ./train
    file_name=$(ls ${arm64_path}/*android-aarch64.tar.gz)
    IFS="-" read -r -a file_name_array <<< "$file_name"
    version=${file_name_array[2]}

    echo "start Run arm64-fp16 ..."
    Run_arm64_fp16
    Run_arm64_fp16_status=$?
    sleep 1
fi


if [[ $backend == "all" || $backend == "arm64_cpu" || $backend == "arm64_fp32" ]]; then
    if [[ ${Run_arm64_fp32_status} != 0 ]];then
        echo "Run_arm64_fp32 failed"
        cat ${run_arm64_fp32_log_file}
        isFailed=1
    fi
fi
if [[ $backend == "all" || $backend == "arm64_cpu" || $backend == "arm64_codegen" ]]; then
    if [[ ${Run_arm64_codegen_status} != 0 ]];then
        echo "Run_arm64_codegen failed"
        cat ${run_arm64_fp32_codegen_log_file}
        isFailed=1
    fi
fi
if [[ $backend == "all" || $backend == "arm32_cpu" || $backend == "arm32_codegen" ]]; then
    if [[ ${Run_arm32_codegen_status} != 0 ]];then
        echo "Run_arm32 codegen failed"
        cat ${run_arm32_fp32_codegen_log_file}
        isFailed=1
    fi
fi
if [[ $backend == "all" || $backend == "arm64_cpu" || $backend == "arm64_fp16" ]]; then
    if [[ ${Run_arm64_fp16_status} != 0 ]];then
        echo "Run_arm64_fp16 failed"
        cat ${run_arm64_fp16_log_file}
        isFailed=1
    fi
fi
if [[ $backend == "all" || $backend == "arm32_cpu" || $backend == "arm32_fp16" ]]; then
    if [[ ${Run_armv82_a32_fp16_status} != 0 ]];then
        echo "Run_armv82_a32_fp16 failed"
        cat ${run_armv82_a32_fp16_log_file}
        isFailed=1
    fi
fi
if [[ $backend == "all" || $backend == "arm32_cpu" || $backend == "arm32_fp32" ]]; then
    if [[ ${Run_arm32_status} != 0 ]];then
        echo "Run_arm32 failed"
        cat ${run_arm32_log_file}
        isFailed=1
    fi
fi

echo "Run_arm64_fp32 and Run_arm64_fp16 and Run_arm32_fp32 and Run_armv82_a32_fp16 is ended"
Print_Benchmark_Result
if [[ $isFailed == 1 ]]; then
    exit 1
fi
echo "Run Arm Train is ended"
Print_Benchmark_Train_Result
if [[ $isFailed == 1 ]]; then
    exit 1
fi
exit 0
