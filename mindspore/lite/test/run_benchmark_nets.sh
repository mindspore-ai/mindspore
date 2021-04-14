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

    # Convert mindspore train models:
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
        echo './converter_lite  --fmk=TFLITE --modelFile='${models_path}'/'${model_name}' --outputFile='${ms_models_path}'/'${model_name}'--quantType=WeightQuant --bitNum=8 --quantWeightChannel=0' >> "${run_converter_log_file}"
        ./converter_lite  --fmk=TFLITE --modelFile=$models_path/${model_name} --outputFile=${ms_models_path}/${model_name}_weightquant --quantType=WeightQuant --bitNum=8 --quantWeightChannel=0
        if [ $? = 0 ]; then
            converter_result='converter weight_quant '${model_name}' pass';echo ${converter_result} >> ${run_converter_result_file}
        else
            converter_result='converter weight_quant '${model_name}' failed';echo ${converter_result} >> ${run_converter_result_file};return 1
        fi
    done < ${models_tflite_weightquant_config}

    # Convert mindir weightquant models:
    while read line; do
        weight_quant_line_info=${line}
        if [[ $weight_quant_line_info == \#* ]]; then
          continue
        fi
        model_name=`echo ${weight_quant_line_info}|awk -F ' ' '{print $1}'`
        echo ${model_name} >> "${run_converter_log_file}"
        echo './converter_lite  --fmk=MINDIR --modelFile='${models_path}'/'${model_name}' --outputFile='${ms_models_path}'/'${model_name}' --quantType=WeightQuant --bitNum=8 --quantWeightSize=500 --quantWeightChannel=16' >> "${run_converter_log_file}"
        ./converter_lite  --fmk=MINDIR --modelFile=$models_path/${model_name} --outputFile=${ms_models_path}/${model_name}_weightquant --quantType=WeightQuant --bitNum=8 --quantWeightSize=500 --quantWeightChannel=16
        if [ $? = 0 ]; then
            converter_result='converter weight_quant '${model_name}' pass';echo ${converter_result} >> ${run_converter_result_file}
        else
            converter_result='converter weight_quant '${model_name}' failed';echo ${converter_result} >> ${run_converter_result_file};return 1
        fi
    done < ${models_mindspore_weightquant_config}

    # Convert mindir mixbit weightquant models:
    while read line; do
        line_info=${line}
        if [[ $line_info == \#* ]]; then
          continue
        fi
        model_name=`echo ${line_info}|awk -F ' ' '{print $1}'`

        echo ${model_name}'_7bit' >> "${run_converter_log_file}"
        echo './converter_lite  --fmk=MINDIR --modelFile='${models_path}'/'${model_name}' --outputFile='${ms_models_path}'/'${model_name}'_7bit  --quantType=WeightQuant --bitNum=7 --quantWeightSize=500 --quantWeightChannel=16' >> "${run_converter_log_file}"
        ./converter_lite  --fmk=MINDIR --modelFile=${models_path}/${model_name} --outputFile=${ms_models_path}/${model_name}'_7bit' --quantType=WeightQuant --bitNum=7 --quantWeightSize=500 --quantWeightChannel=16
        if [ $? = 0 ]; then
            converter_result='converter mindspore '${model_name}'_7bit pass';echo ${converter_result} >> ${run_converter_result_file}
        else
            converter_result='converter mindspore '${model_name}'_7bit failed';echo ${converter_result} >> ${run_converter_result_file};return 1
        fi
        echo ${model_name}'_9bit' >> "${run_converter_log_file}"
        echo './converter_lite  --fmk=MINDIR --modelFile='${models_path}'/'${model_name}' --outputFile='${ms_models_path}'/'${model_name}'_9bit  --quantType=WeightQuant --bitNum=9 --quantWeightSize=500 --quantWeightChannel=16' >> "${run_converter_log_file}"
        ./converter_lite  --fmk=MINDIR --modelFile=${models_path}/${model_name} --outputFile=${ms_models_path}/${model_name}'_9bit' --quantType=WeightQuant --bitNum=9 --quantWeightSize=500 --quantWeightChannel=16
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

# Run on x86 platform:
function Run_x86() {
    echo 'cd  '${x86_path}'/mindspore-lite-'${version}'-inference-linux-x64' >> "${run_x86_log_file}"
    cd ${x86_path}/mindspore-lite-${version}-inference-linux-x64 || return 1
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./inference/lib:./inference/minddata/lib
    cp tools/benchmark/benchmark ./ || exit 1

    # Run tf converted models:
    while read line; do
        model_name=${line%;*}
        length=${#model_name}
        input_shapes=${line:length+1}
        if [[ $model_name == \#* ]]; then
          continue
        fi
        echo ${model_name} >> "${run_x86_log_file}"
        echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'.ms --inDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/input/'${model_name}'.ms.bin --inputShapes='${input_shapes}' --benchmarkDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/'${model_name}'.ms.out' >> "${run_x86_log_file}"
        ./benchmark --modelFile=${ms_models_path}/${model_name}.ms --inDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/input/${model_name}.ms.bin --inputShapes=${input_shapes} --benchmarkDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/${model_name}.ms.out >> "${run_x86_log_file}"
        if [ $? = 0 ]; then
            run_result='x86: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='x86: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
    done < ${models_tf_config}

    # Run tflite converted models:
    while read line; do
        model_name=${line}
        if [[ $model_name == \#* ]]; then
          continue
        fi
        echo ${model_name} >> "${run_x86_log_file}"
        echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'.ms --inDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/input/'${model_name}'.ms.bin --benchmarkDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/'${model_name}'.ms.out' >> "${run_x86_log_file}"
        ./benchmark --modelFile=${ms_models_path}/${model_name}.ms --inDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/input/${model_name}.ms.bin --benchmarkDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/${model_name}.ms.out >> "${run_x86_log_file}"
        if [ $? = 0 ]; then
            run_result='x86: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='x86: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
    done < ${models_tflite_config}

    # Run caffe converted models:
    while read line; do
        model_name=${line}
        if [[ $model_name == \#* ]]; then
          continue
        fi
        echo ${model_name} >> "${run_x86_log_file}"
        echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'.ms --inDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/input/'${model_name}'.ms.bin --benchmarkDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/'${model_name}'.ms.out' >> "${run_x86_log_file}"
        ./benchmark --modelFile=${ms_models_path}/${model_name}.ms --inDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/input/${model_name}.ms.bin --benchmarkDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/${model_name}.ms.out >> "${run_x86_log_file}"
        if [ $? = 0 ]; then
            run_result='x86: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='x86: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
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
        echo ${model_name} >> "${run_x86_log_file}"
        echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'.ms --inDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/input/'${model_name}'.ms.bin --inputShapes='${input_shapes}' --benchmarkDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/'${model_name}'.ms.out' >> "${run_x86_log_file}"
        ./benchmark --modelFile=${ms_models_path}/${model_name}.ms --inDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/input/${model_name}.ms.bin --inputShapes=${input_shapes} --benchmarkDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/${model_name}.ms.out >> "${run_x86_log_file}"
        if [ $? = 0 ]; then
            run_result='x86: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='x86: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
    done < ${models_onnx_config}

    # Run mindspore converted models:
    while read line; do
        mindspore_line_info=${line}
        if [[ $mindspore_line_info == \#* ]]; then
          continue
        fi
        model_name=`echo ${mindspore_line_info}|awk -F ' ' '{print $1}'`
        accuracy_limit=`echo ${mindspore_line_info}|awk -F ' ' '{print $2}'`
        echo "---------------------------------------------------------" >> "${run_x86_log_file}"
        echo "mindspore run: ${model_name}, accuracy limit:${accuracy_limit}" >> "${run_x86_log_file}"
        echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'.ms --inDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/input/'${model_name}'.ms.bin --benchmarkDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/'${model_name}'.ms.out' >> "${run_x86_log_file}"
        ./benchmark --modelFile=${ms_models_path}/${model_name}.ms --inDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/input/${model_name}.ms.bin --benchmarkDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/${model_name}.ms.out --accuracyThreshold=${accuracy_limit} >> "${run_x86_log_file}"
        if [ $? = 0 ]; then
            run_result='x86: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='x86: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
    done < ${models_mindspore_config}

    # Run mindspore converted train models:
    while read line; do
        model_name=${line}
        if [[ $model_name == \#* ]]; then
          continue
        fi
        echo ${model_name}'_train' >> "${run_x86_log_file}"
        echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'_train.ms --inDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/input/'${model_name}'.ms.bin --benchmarkDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/'${model_name}'.train.ms.out' >> "${run_x86_log_file}"
        ./benchmark --modelFile=${ms_models_path}/${model_name}'_train'.ms --inDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/input/${model_name}.ms.bin --benchmarkDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/${model_name}.train.ms.out --accuracyThreshold=1.5 >> "${run_x86_log_file}"
        if [ $? = 0 ]; then
            run_result='x86: '${model_name}'_train pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='x86: '${model_name}'_train failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
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
        transformer_data_path="/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/input"
        echo ${model_name} >> "${run_x86_log_file}"
        if [[ $model_name == "mobilenet.tflite" ]]; then
            echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'_posttraining.ms --inDataFile=/home/workspace/mindspore_dataset/mslite/quantTraining/mnist_calibration_data/00099.bin --benchmarkDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/'${model_name}'_posttraining.ms.out' --accuracyThreshold=${accuracy_limit} >> "${run_x86_log_file}"
            ./benchmark --modelFile=${ms_models_path}/${model_name}_posttraining.ms --inDataFile=/home/workspace/mindspore_dataset/mslite/quantTraining/mnist_calibration_data/00099.bin --benchmarkDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/${model_name}_posttraining.ms.out --accuracyThreshold=${accuracy_limit} >> "${run_x86_log_file}"
        fi
        if [[ $model_name == "transformer_20200831_encoder_fp32.tflite" ]]; then
            echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'_posttraining.ms --inDataFile=${transformer_data_path}/encoder_buffer_in_0-35.bin --benchmarkDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/'${model_name}'_posttraining.ms.out' --accuracyThreshold=${accuracy_limit} >> "${run_x86_log_file}"
            ./benchmark --modelFile=${ms_models_path}/${model_name}_posttraining.ms --inDataFile=${transformer_data_path}/encoder_buffer_in_35.bin,${transformer_data_path}/encoder_buffer_in_0.bin,${transformer_data_path}/encoder_buffer_in_1.bin,${transformer_data_path}/encoder_buffer_in_4.bin,${transformer_data_path}/encoder_buffer_in_2.bin,${transformer_data_path}/encoder_buffer_in_3.bin,${transformer_data_path}/encoder_buffer_in_7.bin,${transformer_data_path}/encoder_buffer_in_5.bin,${transformer_data_path}/encoder_buffer_in_6.bin,${transformer_data_path}/encoder_buffer_in_10.bin,${transformer_data_path}/encoder_buffer_in_8.bin,${transformer_data_path}/encoder_buffer_in_9.bin,${transformer_data_path}/encoder_buffer_in_11.bin,${transformer_data_path}/encoder_buffer_in_12.bin,${transformer_data_path}/encoder_buffer_in_15.bin,${transformer_data_path}/encoder_buffer_in_13.bin,${transformer_data_path}/encoder_buffer_in_14.bin,${transformer_data_path}/encoder_buffer_in_18.bin,${transformer_data_path}/encoder_buffer_in_16.bin,${transformer_data_path}/encoder_buffer_in_17.bin,${transformer_data_path}/encoder_buffer_in_21.bin,${transformer_data_path}/encoder_buffer_in_19.bin,${transformer_data_path}/encoder_buffer_in_20.bin,${transformer_data_path}/encoder_buffer_in_22.bin,${transformer_data_path}/encoder_buffer_in_23.bin,${transformer_data_path}/encoder_buffer_in_26.bin,${transformer_data_path}/encoder_buffer_in_24.bin,${transformer_data_path}/encoder_buffer_in_25.bin,${transformer_data_path}/encoder_buffer_in_29.bin,${transformer_data_path}/encoder_buffer_in_27.bin,${transformer_data_path}/encoder_buffer_in_28.bin,${transformer_data_path}/encoder_buffer_in_32.bin,${transformer_data_path}/encoder_buffer_in_30.bin,${transformer_data_path}/encoder_buffer_in_31.bin,${transformer_data_path}/encoder_buffer_in_33.bin,${transformer_data_path}/encoder_buffer_in_34.bin --benchmarkDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/${model_name}_posttraining.ms.out --accuracyThreshold=${accuracy_limit} >> "${run_x86_log_file}"
        fi
        if [[ $model_name == "transformer_20200831_decoder_fp32.tflite" ]]; then
            echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'_posttraining.ms --inDataFile=${transformer_data_path}/decoder_buffer_in_0-10.bin --benchmarkDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/'${model_name}'_posttraining.ms.out' --accuracyThreshold=${accuracy_limit} >> "${run_x86_log_file}"
            ./benchmark --modelFile=${ms_models_path}/${model_name}_posttraining.ms --inDataFile=${transformer_data_path}/decoder_buffer_in_9.bin,${transformer_data_path}/decoder_buffer_in_2.bin,${transformer_data_path}/decoder_buffer_in_0.bin,${transformer_data_path}/decoder_buffer_in_1.bin,${transformer_data_path}/decoder_buffer_in_5.bin,${transformer_data_path}/decoder_buffer_in_3.bin,${transformer_data_path}/decoder_buffer_in_4.bin,${transformer_data_path}/decoder_buffer_in_8.bin,${transformer_data_path}/decoder_buffer_in_6.bin,${transformer_data_path}/decoder_buffer_in_7.bin,${transformer_data_path}/decoder_buffer_in_10.bin --benchmarkDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/${model_name}_posttraining.ms.out --accuracyThreshold=${accuracy_limit} >> "${run_x86_log_file}"
        fi
        if [ $? = 0 ]; then
            run_result='x86: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='x86: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
    done < ${models_tflite_posttraining_config}

    # Run caffe post training quantization converted models:
    while read line; do
        posttraining_line_info=${line}
        if [[ $posttraining_line_info == \#* ]]; then
          continue
        fi
        model_name=`echo ${posttraining_line_info}|awk -F ' ' '{print $1}'`
        accuracy_limit=`echo ${posttraining_line_info}|awk -F ' ' '{print $2}'`
        echo ${model_name} >> "${run_x86_log_file}"
        echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'_posttraining.ms --inDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/input/'${model_name}'_posttraining.ms.bin --benchmarkDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/'${model_name}'_posttraining.ms.out'  --accuracyThreshold=${accuracy_limit} >> "${run_x86_log_file}"
        ./benchmark --modelFile=${ms_models_path}/${model_name}_posttraining.ms --inDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/input/${model_name}_posttraining.ms.bin --benchmarkDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/${model_name}_posttraining.ms.out --accuracyThreshold=${accuracy_limit} >> "${run_x86_log_file}"
        if [ $? = 0 ]; then
            run_result='x86: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='x86: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
    done < ${models_caffe_posttraining_config}

    # Run tflite aware training quantization converted models:
    while read line; do
        model_name=${line}
        if [[ $model_name == \#* ]]; then
          continue
        fi
        echo ${model_name} >> "${run_x86_log_file}"
        echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'.ms --inDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/input/'${model_name}'.ms.bin --benchmarkDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/'${model_name}'.ms.out' >> "${run_x86_log_file}"
        ./benchmark --modelFile=${ms_models_path}/${model_name}.ms --inDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/input/${model_name}.ms.bin --benchmarkDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/${model_name}.ms.out >> "${run_x86_log_file}"
        if [ $? = 0 ]; then
            run_result='x86: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='x86: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
    done < ${models_tflite_awaretraining_config}

    # Run tflite weight quantization converted models:
    while read line; do
        weight_quant_line_info=${line}
        if [[ $weight_quant_line_info == \#* ]]; then
          continue
        fi
        model_name=`echo ${weight_quant_line_info}|awk -F ' ' '{print $1}'`
        accuracy_limit=`echo ${weight_quant_line_info}|awk -F ' ' '{print $2}'`
        echo ${model_name} >> "${run_x86_log_file}"
        echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'.ms --inDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/input/'${model_name}'.ms.bin --benchmarkDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/'${model_name}'.ms.out --accuracyThreshold=${accuracy_limit}' >> "${run_x86_log_file}"
        ./benchmark --modelFile=${ms_models_path}/${model_name}_weightquant.ms --inDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/input/${model_name}.ms.bin --benchmarkDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/${model_name}.ms.out --accuracyThreshold=${accuracy_limit}>> "${run_x86_log_file}"
        if [ $? = 0 ]; then
            run_result='x86: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='x86: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
    done < ${models_tflite_weightquant_config}

    # Run mindir weight quantization converted models:
    while read line; do
        weight_quant_line_info=${line}
        if [[ $weight_quant_line_info == \#* ]]; then
          continue
        fi
        model_name=`echo ${weight_quant_line_info}|awk -F ' ' '{print $1}'`
        accuracy_limit=`echo ${weight_quant_line_info}|awk -F ' ' '{print $2}'`
        echo ${model_name} >> "${run_x86_log_file}"
        echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'.ms --inDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/input/'${model_name}'.ms.bin --benchmarkDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/'${model_name}'.ms.out --accuracyThreshold=${accuracy_limit}' >> "${run_x86_log_file}"
        ./benchmark --modelFile=${ms_models_path}/${model_name}_weightquant.ms --inDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/input/${model_name}.ms.bin --benchmarkDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/${model_name}.ms.out --accuracyThreshold=${accuracy_limit} >> "${run_x86_log_file}"
        if [ $? = 0 ]; then
            run_result='x86: '${model_name}'[weight_quant] pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='x86: '${model_name}'[weight_quant] failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
    done < ${models_mindspore_weightquant_config}

    # Run mindir mixbit weight quantization converted models:
    while read line; do
        line_info=${line}
        if [[ $line_info == \#* ]]; then
          continue
        fi
        model_name=`echo ${line_info}|awk -F ' ' '{print $1}'`
        accuracy_limit_7bit=`echo ${line_info}|awk -F ' ' '{print $2}'`
        accuracy_limit_9bit=`echo ${line_info}|awk -F ' ' '{print $3}'`

        echo ${model_name} >> "${run_x86_log_file}"
        echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'_7bit.ms --inDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/input/'${model_name}'.ms.bin --benchmarkDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/'${model_name}'.ms.out --accuracyThreshold=${accuracy_limit_7bit}' >> "${run_x86_log_file}"
        ./benchmark --modelFile=${ms_models_path}/${model_name}_7bit.ms --inDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/input/${model_name}.ms.bin --benchmarkDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/${model_name}.ms.out --accuracyThreshold=${accuracy_limit_7bit} >> "${run_x86_log_file}"
        if [ $? = 0 ]; then
            run_result='x86: '${model_name}'_7bit pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='x86: '${model_name}'_7bit failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
        echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'_9bit.ms --inDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/input/'${model_name}'.ms.bin --benchmarkDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/'${model_name}'.ms.out --accuracyThreshold=${accuracy_limit_9bit}' >> "${run_x86_log_file}"
        ./benchmark --modelFile=${ms_models_path}/${model_name}_9bit.ms --inDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/input/${model_name}.ms.bin --benchmarkDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/${model_name}.ms.out --accuracyThreshold=${accuracy_limit_9bit} >> "${run_x86_log_file}"
        if [ $? = 0 ]; then
            run_result='x86: '${model_name}'_9bit pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='x86: '${model_name}'_9bit failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
    done < ${models_mindspore_mixbit_config}

    # Run converted models which has multiple inputs:
    while read line; do
        if [[ $line == \#* ]]; then
          continue
        fi
        model_name=`echo ${line} | awk -F ';' '{print $1}'`
        input_num=`echo ${line} | awk -F ';' '{print $2}'`
        input_shapes=`echo ${line} | awk -F ';' '{print $3}'`
        input_files=''
        output_file=''
        data_path="/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/"
        for i in $(seq 1 $input_num)
        do
          input_files=$input_files${data_path}'input/'$model_name'.ms.bin_'$i','
        done
        output_file=${data_path}'output/'${model_name}'.ms.out'
        if [[ ${model_name##*.} == "caffemodel" ]]; then
          model_name=${model_name%.*}
        fi
        echo ${model_name} >> "${run_x86_log_file}"
        echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'.ms --inDataFile='${input_files}' --inputShapes='${input_shapes}' --benchmarkDataFile='${output_file} >> "${run_x86_log_file}"
        ./benchmark --modelFile=${ms_models_path}/${model_name}.ms --inDataFile=${input_files} --inputShapes=${input_shapes} --benchmarkDataFile=${output_file} >> "${run_x86_log_file}"
        if [ $? = 0 ]; then
            run_result='x86: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='x86: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
    done < ${models_with_multiple_inputs_config}

    # Run converted models which does not need to be cared about the accuracy:
    while read line; do
        if [[ $line == \#* ]]; then
          continue
        fi
        model_name=`echo ${line} | awk -F ';' '{print $1}'`
        input_num=`echo ${line} | awk -F ';' '{print $2}'`
        input_shapes=`echo ${line} | awk -F ';' '{print $3}'`
        input_files=''
        output_file=''
        data_path="/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/"
        if [[ ${model_name##*.} == "caffemodel" ]]; then
          model_name=${model_name%.*}
        fi
        echo ${model_name} >> "${run_x86_log_file}"
        echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'.ms --inputShapes='${input_shapes}' --loopCount=2 --warmUpLoopCount=0' >> "${run_x86_log_file}"
        ./benchmark --modelFile=${ms_models_path}/${model_name}.ms --inputShapes=${input_shapes} --loopCount=2 --warmUpLoopCount=0 >> "${run_x86_log_file}"
        if [ $? = 0 ]; then
            run_result='x86: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='x86: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
    done < ${models_for_process_only_config}
}

# Run on x86 sse platform:
function Run_x86_sse() {
    cd ${x86_path}/sse || exit 1
    tar -zxf mindspore-lite-${version}-inference-linux-x64.tar.gz || exit 1
    cd ${x86_path}/sse/mindspore-lite-${version}-inference-linux-x64 || return 1
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./inference/lib:./inference/minddata/lib
    cp tools/benchmark/benchmark ./ || exit 1

    # Run tflite converted models:
    while read line; do
        model_name=${line}
        if [[ $model_name == \#* ]]; then
          continue
        fi
        echo ${model_name} >> "${run_x86_sse_log_file}"
        echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'.ms --inDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/input/'${model_name}'.ms.bin --benchmarkDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/'${model_name}'.ms.out' >> "${run_x86_sse_log_file}"
        ./benchmark --modelFile=${ms_models_path}/${model_name}.ms --inDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/input/${model_name}.ms.bin --benchmarkDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/${model_name}.ms.out >> "${run_x86_sse_log_file}"
        if [ $? = 0 ]; then
            run_result='x86_sse: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='x86_sse: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
    done < ${models_tflite_config}

    # Run caffe converted models:
    while read line; do
        model_name=${line}
        if [[ $model_name == \#* ]]; then
          continue
        fi
        echo ${model_name} >> "${run_x86_sse_log_file}"
        echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'.ms --inDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/input/'${model_name}'.ms.bin --benchmarkDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/'${model_name}'.ms.out' >> "${run_x86_sse_log_file}"
        ./benchmark --modelFile=${ms_models_path}/${model_name}.ms --inDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/input/${model_name}.ms.bin --benchmarkDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/${model_name}.ms.out >> "${run_x86_sse_log_file}"
        if [ $? = 0 ]; then
            run_result='x86_sse: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='x86_sse: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
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
        echo ${model_name} >> "${run_x86_sse_log_file}"
        echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'.ms --inDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/input/'${model_name}'.ms.bin --inputShapes='${input_shapes}' --benchmarkDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/'${model_name}'.ms.out' >> "${run_x86_sse_log_file}"
        ./benchmark --modelFile=${ms_models_path}/${model_name}.ms --inDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/input/${model_name}.ms.bin --inputShapes=${input_shapes} --benchmarkDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/${model_name}.ms.out >> "${run_x86_sse_log_file}"
        if [ $? = 0 ]; then
            run_result='x86_sse: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='x86_sse: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
    done < ${models_onnx_config}

    # Run mindspore converted models:
    while read line; do
        mindspore_line_info=${line}
        if [[ $mindspore_line_info == \#* ]]; then
          continue
        fi
        model_name=`echo ${mindspore_line_info}|awk -F ' ' '{print $1}'`
        accuracy_limit=`echo ${mindspore_line_info}|awk -F ' ' '{print $2}'`
        echo "---------------------------------------------------------" >> "${run_x86_sse_log_file}"
        echo "mindspore run: ${model_name}, accuracy limit:${accuracy_limit}" >> "${run_x86_sse_log_file}"
        echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'.ms --inDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/input/'${model_name}'.ms.bin --benchmarkDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/'${model_name}'.ms.out' >> "${run_x86_sse_log_file}"
        ./benchmark --modelFile=${ms_models_path}/${model_name}.ms --inDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/input/${model_name}.ms.bin --benchmarkDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/${model_name}.ms.out --accuracyThreshold=${accuracy_limit} >> "${run_x86_sse_log_file}"
        if [ $? = 0 ]; then
            run_result='x86_sse: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='x86_sse: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
    done < ${models_mindspore_config}

    # Run mindspore converted train models:
    while read line; do
        model_name=${line}
        if [[ $model_name == \#* ]]; then
          continue
        fi
        echo ${model_name}'_train' >> "${run_x86_sse_log_file}"
        echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'_train.ms --inDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/input/'${model_name}'.ms.bin --benchmarkDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/'${model_name}'.train.ms.out' >> "${run_x86_sse_log_file}"
        ./benchmark --modelFile=${ms_models_path}/${model_name}'_train'.ms --inDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/input/${model_name}.ms.bin --benchmarkDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/${model_name}.train.ms.out --accuracyThreshold=1.5 >> "${run_x86_sse_log_file}"
        if [ $? = 0 ]; then
            run_result='x86_sse: '${model_name}'_train pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='x86_sse: '${model_name}'_train failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
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
        transformer_data_path="/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/input"
        echo ${model_name} >> "${run_x86_sse_log_file}"
        if [[ $model_name == "mobilenet.tflite" ]]; then
            echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'_posttraining.ms --inDataFile=/home/workspace/mindspore_dataset/mslite/quantTraining/mnist_calibration_data/00099.bin --benchmarkDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/'${model_name}'_posttraining.ms.out' --accuracyThreshold=${accuracy_limit} >> "${run_x86_sse_log_file}"
            ./benchmark --modelFile=${ms_models_path}/${model_name}_posttraining.ms --inDataFile=/home/workspace/mindspore_dataset/mslite/quantTraining/mnist_calibration_data/00099.bin --benchmarkDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/${model_name}_posttraining.ms.out --accuracyThreshold=${accuracy_limit} >> "${run_x86_sse_log_file}"
        fi
        if [[ $model_name == "transformer_20200831_encoder_fp32.tflite" ]]; then
            echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'_posttraining.ms --inDataFile=${transformer_data_path}/encoder_buffer_in_0-35.bin --benchmarkDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/'${model_name}'_posttraining.ms.out' --accuracyThreshold=${accuracy_limit} >> "${run_x86_sse_log_file}"
            ./benchmark --modelFile=${ms_models_path}/${model_name}_posttraining.ms --inDataFile=${transformer_data_path}/encoder_buffer_in_35.bin,${transformer_data_path}/encoder_buffer_in_0.bin,${transformer_data_path}/encoder_buffer_in_1.bin,${transformer_data_path}/encoder_buffer_in_4.bin,${transformer_data_path}/encoder_buffer_in_2.bin,${transformer_data_path}/encoder_buffer_in_3.bin,${transformer_data_path}/encoder_buffer_in_7.bin,${transformer_data_path}/encoder_buffer_in_5.bin,${transformer_data_path}/encoder_buffer_in_6.bin,${transformer_data_path}/encoder_buffer_in_10.bin,${transformer_data_path}/encoder_buffer_in_8.bin,${transformer_data_path}/encoder_buffer_in_9.bin,${transformer_data_path}/encoder_buffer_in_11.bin,${transformer_data_path}/encoder_buffer_in_12.bin,${transformer_data_path}/encoder_buffer_in_15.bin,${transformer_data_path}/encoder_buffer_in_13.bin,${transformer_data_path}/encoder_buffer_in_14.bin,${transformer_data_path}/encoder_buffer_in_18.bin,${transformer_data_path}/encoder_buffer_in_16.bin,${transformer_data_path}/encoder_buffer_in_17.bin,${transformer_data_path}/encoder_buffer_in_21.bin,${transformer_data_path}/encoder_buffer_in_19.bin,${transformer_data_path}/encoder_buffer_in_20.bin,${transformer_data_path}/encoder_buffer_in_22.bin,${transformer_data_path}/encoder_buffer_in_23.bin,${transformer_data_path}/encoder_buffer_in_26.bin,${transformer_data_path}/encoder_buffer_in_24.bin,${transformer_data_path}/encoder_buffer_in_25.bin,${transformer_data_path}/encoder_buffer_in_29.bin,${transformer_data_path}/encoder_buffer_in_27.bin,${transformer_data_path}/encoder_buffer_in_28.bin,${transformer_data_path}/encoder_buffer_in_32.bin,${transformer_data_path}/encoder_buffer_in_30.bin,${transformer_data_path}/encoder_buffer_in_31.bin,${transformer_data_path}/encoder_buffer_in_33.bin,${transformer_data_path}/encoder_buffer_in_34.bin --benchmarkDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/${model_name}_posttraining.ms.out --accuracyThreshold=${accuracy_limit} >> "${run_x86_sse_log_file}"
        fi
        if [[ $model_name == "transformer_20200831_decoder_fp32.tflite" ]]; then
            echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'_posttraining.ms --inDataFile=${transformer_data_path}/encoder_buffer_in_0-10.bin --benchmarkDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/'${model_name}'_posttraining.ms.out' --accuracyThreshold=${accuracy_limit} >> "${run_x86_sse_log_file}"
            ./benchmark --modelFile=${ms_models_path}/${model_name}_posttraining.ms --inDataFile=${transformer_data_path}/decoder_buffer_in_9.bin,${transformer_data_path}/decoder_buffer_in_2.bin,${transformer_data_path}/decoder_buffer_in_0.bin,${transformer_data_path}/decoder_buffer_in_1.bin,${transformer_data_path}/decoder_buffer_in_5.bin,${transformer_data_path}/decoder_buffer_in_3.bin,${transformer_data_path}/decoder_buffer_in_4.bin,${transformer_data_path}/decoder_buffer_in_8.bin,${transformer_data_path}/decoder_buffer_in_6.bin,${transformer_data_path}/decoder_buffer_in_7.bin,${transformer_data_path}/decoder_buffer_in_10.bin --benchmarkDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/${model_name}_posttraining.ms.out --accuracyThreshold=${accuracy_limit} >> "${run_x86_sse_log_file}"
        fi
        if [ $? = 0 ]; then
            run_result='x86_sse: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='x86_sse: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
    done < ${models_tflite_posttraining_config}

    # Run caffe post training quantization converted models:
    while read line; do
        posttraining_line_info=${line}
        if [[ $posttraining_line_info == \#* ]]; then
          continue
        fi
        model_name=`echo ${posttraining_line_info}|awk -F ' ' '{print $1}'`
        accuracy_limit=`echo ${posttraining_line_info}|awk -F ' ' '{print $2}'`
        echo ${model_name} >> "${run_x86_sse_log_file}"
        echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'_posttraining.ms --inDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/input/'${model_name}'_posttraining.ms.bin --benchmarkDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/'${model_name}'_posttraining.ms.out'  --accuracyThreshold=${accuracy_limit} >> "${run_x86_sse_log_file}"
        ./benchmark --modelFile=${ms_models_path}/${model_name}_posttraining.ms --inDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/input/${model_name}_posttraining.ms.bin --benchmarkDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/${model_name}_posttraining.ms.out --accuracyThreshold=${accuracy_limit} >> "${run_x86_sse_log_file}"
        if [ $? = 0 ]; then
            run_result='x86_sse: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='x86_sse: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
    done < ${models_caffe_posttraining_config}

    # Run tflite aware training quantization converted models:
    while read line; do
        model_name=${line}
        if [[ $model_name == \#* ]]; then
          continue
        fi
        echo ${model_name} >> "${run_x86_sse_log_file}"
        echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'.ms --inDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/input/'${model_name}'.ms.bin --benchmarkDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/'${model_name}'.ms.out' >> "${run_x86_sse_log_file}"
        ./benchmark --modelFile=${ms_models_path}/${model_name}.ms --inDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/input/${model_name}.ms.bin --benchmarkDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/${model_name}.ms.out >> "${run_x86_sse_log_file}"
        if [ $? = 0 ]; then
            run_result='x86_sse: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='x86_sse: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
    done < ${models_tflite_awaretraining_config}

    # Run tflite weight quantization converted models:
    while read line; do
        weight_quant_line_info=${line}
        if [[ $weight_quant_line_info == \#* ]]; then
          continue
        fi
        model_name=`echo ${weight_quant_line_info}|awk -F ' ' '{print $1}'`
        accuracy_limit=`echo ${weight_quant_line_info}|awk -F ' ' '{print $2}'`
        echo ${model_name} >> "${run_x86_sse_log_file}"
        echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'.ms --inDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/input/'${model_name}'.ms.bin --benchmarkDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/'${model_name}'.ms.out --accuracyThreshold=${accuracy_limit}' >> "${run_x86_sse_log_file}"
        ./benchmark --modelFile=${ms_models_path}/${model_name}_weightquant.ms --inDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/input/${model_name}.ms.bin --benchmarkDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/${model_name}.ms.out --accuracyThreshold=${accuracy_limit} >> "${run_x86_sse_log_file}"
        if [ $? = 0 ]; then
            run_result='x86_sse: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='x86_sse: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
    done < ${models_tflite_weightquant_config}

    # Run mindir weight quantization converted models:
    while read line; do
        weight_quant_line_info=${line}
        if [[ $weight_quant_line_info == \#* ]]; then
          continue
        fi
        model_name=`echo ${weight_quant_line_info}|awk -F ' ' '{print $1}'`
        accuracy_limit=`echo ${weight_quant_line_info}|awk -F ' ' '{print $2}'`
        echo ${model_name} >> "${run_x86_sse_log_file}"
        echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'.ms --inDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/input/'${model_name}'.ms.bin --benchmarkDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/'${model_name}'.ms.out --accuracyThreshold=${accuracy_limit}' >> "${run_x86_sse_log_file}"
        ./benchmark --modelFile=${ms_models_path}/${model_name}_weightquant.ms --inDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/input/${model_name}.ms.bin --benchmarkDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/${model_name}.ms.out --accuracyThreshold=${accuracy_limit} >> "${run_x86_sse_log_file}"
        if [ $? = 0 ]; then
            run_result='x86_sse: '${model_name}'[weight quant] pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='x86_sse: '${model_name}'[weight quant] failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
    done < ${models_mindspore_weightquant_config}

    # Run mindir mixbit weight quantization converted models:
    while read line; do
        line_info=${line}
        if [[ $line_info == \#* ]]; then
          continue
        fi
        model_name=`echo ${line_info}|awk -F ' ' '{print $1}'`
        accuracy_limit_7bit=`echo ${line_info}|awk -F ' ' '{print $2}'`
        accuracy_limit_9bit=`echo ${line_info}|awk -F ' ' '{print $3}'`

        echo ${model_name} >> "${run_x86_sse_log_file}"
        echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'_7bit.ms --inDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/input/'${model_name}'.ms.bin --benchmarkDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/'${model_name}'.ms.out --accuracyThreshold=${accuracy_limit_7bit}' >> "${run_x86_sse_log_file}"
        ./benchmark --modelFile=${ms_models_path}/${model_name}_7bit.ms --inDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/input/${model_name}.ms.bin --benchmarkDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/${model_name}.ms.out --accuracyThreshold=${accuracy_limit_7bit} >> "${run_x86_sse_log_file}"
        if [ $? = 0 ]; then
            run_result='x86_sse: '${model_name}'_7bit pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='x86_sse: '${model_name}'_7bit failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
        echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'_9bit.ms --inDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/input/'${model_name}'.ms.bin --benchmarkDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/'${model_name}'.ms.out --accuracyThreshold=${accuracy_limit_9bit}' >> "${run_x86_sse_log_file}"
        ./benchmark --modelFile=${ms_models_path}/${model_name}_9bit.ms --inDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/input/${model_name}.ms.bin --benchmarkDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/${model_name}.ms.out --accuracyThreshold=${accuracy_limit_9bit} >> "${run_x86_sse_log_file}"
        if [ $? = 0 ]; then
            run_result='x86_sse: '${model_name}'_9bit pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='x86_sse: '${model_name}'_9bit failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
    done < ${models_mindspore_mixbit_config}

    # Run converted models with multiple inputs:
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
        data_path="/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/"
        for i in $(seq 1 $input_num)
        do
          input_files=$input_files${data_path}'input/'$model_name'.ms.bin_'$i','
        done
        output_file=${data_path}'output/'${model_name}'.ms.out'
        if [[ ${model_name##*.} == "caffemodel" ]]; then
          model_name=${model_name%.*}
        fi
        echo ${model_name} >> "${run_x86_sse_log_file}"
        echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'.ms --inDataFile='${input_files}' --inputShapes='${input_shapes}' --benchmarkDataFile='${output_file} >> "${run_x86_sse_log_file}"
        ./benchmark --modelFile=${ms_models_path}/${model_name}.ms --inDataFile=${input_files} --inputShapes=${input_shapes} --benchmarkDataFile=${output_file} >> "${run_x86_sse_log_file}"
        if [ $? = 0 ]; then
            run_result='x86_sse: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='x86_sse: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
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
        echo ${model_name} >> "${run_x86_sse_log_file}"
        echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'.ms --inputShapes='${input_shapes}' --loopCount=2 --warmUpLoopCount=0' >> "${run_x86_sse_log_file}"
        ./benchmark --modelFile=${ms_models_path}/${model_name}.ms --inputShapes=${input_shapes} --loopCount=2 --warmUpLoopCount=0 >> "${run_x86_sse_log_file}"
        if [ $? = 0 ]; then
            run_result='x86_sse: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='x86_sse: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
    done < ${models_for_process_only_config}
}

# Run on x86 avx platform:
function Run_x86_avx() {
    cd ${x86_path}/avx || exit 1
    tar -zxf mindspore-lite-${version}-inference-linux-x64.tar.gz || exit 1
    cd ${x86_path}/avx/mindspore-lite-${version}-inference-linux-x64 || return 1
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./inference/lib:./inference/minddata/lib
    cp tools/benchmark/benchmark ./ || exit 1

    # Run tflite converted models:
    while read line; do
        model_name=${line}
        if [[ $model_name == \#* ]]; then
          continue
        fi
        echo ${model_name} >> "${run_x86_avx_log_file}"
        echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'.ms --inDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/input/'${model_name}'.ms.bin --benchmarkDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/'${model_name}'.ms.out' >> "${run_x86_avx_log_file}"
        ./benchmark --modelFile=${ms_models_path}/${model_name}.ms --inDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/input/${model_name}.ms.bin --benchmarkDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/${model_name}.ms.out >> "${run_x86_avx_log_file}"
        if [ $? = 0 ]; then
            run_result='x86_avx: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='x86_avx: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
    done < ${models_tflite_config}

    # Run caffe converted models:
    while read line; do
        model_name=${line}
        if [[ $model_name == \#* ]]; then
          continue
        fi
        echo ${model_name} >> "${run_x86_avx_log_file}"
        echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'.ms --inDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/input/'${model_name}'.ms.bin --benchmarkDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/'${model_name}'.ms.out' >> "${run_x86_avx_log_file}"
        ./benchmark --modelFile=${ms_models_path}/${model_name}.ms --inDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/input/${model_name}.ms.bin --benchmarkDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/${model_name}.ms.out >> "${run_x86_avx_log_file}"
        if [ $? = 0 ]; then
            run_result='x86_avx: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='x86_avx: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
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
        echo ${model_name} >> "${run_x86_avx_log_file}"
        echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'.ms --inDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/input/'${model_name}'.ms.bin --inputShapes='${input_shapes}' --benchmarkDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/'${model_name}'.ms.out' >> "${run_x86_avx_log_file}"
        ./benchmark --modelFile=${ms_models_path}/${model_name}.ms --inDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/input/${model_name}.ms.bin --inputShapes=${input_shapes} --benchmarkDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/${model_name}.ms.out >> "${run_x86_avx_log_file}"
        if [ $? = 0 ]; then
            run_result='x86_avx: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='x86_avx: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
    done < ${models_onnx_config}

    # Run mindspore converted models:
    while read line; do
        mindspore_line_info=${line}
        if [[ $mindspore_line_info == \#* ]]; then
          continue
        fi
        model_name=`echo ${mindspore_line_info}|awk -F ' ' '{print $1}'`
        accuracy_limit=`echo ${mindspore_line_info}|awk -F ' ' '{print $2}'`
        echo "---------------------------------------------------------" >> "${run_x86_avx_log_file}"
        echo "mindspore run: ${model_name}, accuracy limit:${accuracy_limit}" >> "${run_x86_avx_log_file}"
        echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'.ms --inDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/input/'${model_name}'.ms.bin --benchmarkDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/'${model_name}'.ms.out' >> "${run_x86_avx_log_file}"
        ./benchmark --modelFile=${ms_models_path}/${model_name}.ms --inDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/input/${model_name}.ms.bin --benchmarkDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/${model_name}.ms.out --accuracyThreshold=${accuracy_limit} >> "${run_x86_avx_log_file}"
        if [ $? = 0 ]; then
            run_result='x86_avx: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='x86_avx: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
    done < ${models_mindspore_config}

    # Run mindspore converted train models:
    while read line; do
        model_name=${line}
        if [[ $model_name == \#* ]]; then
          continue
        fi
        echo ${model_name}'_train' >> "${run_x86_avx_log_file}"
        echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'_train.ms --inDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/input/'${model_name}'.ms.bin --benchmarkDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/'${model_name}'.train.ms.out' >> "${run_x86_avx_log_file}"
        ./benchmark --modelFile=${ms_models_path}/${model_name}'_train'.ms --inDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/input/${model_name}.ms.bin --benchmarkDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/${model_name}.train.ms.out --accuracyThreshold=1.5 >> "${run_x86_avx_log_file}"
        if [ $? = 0 ]; then
            run_result='x86_avx: '${model_name}'_train pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='x86_avx: '${model_name}'_train failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
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
        transformer_data_path="/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/input"
        echo ${model_name} >> "${run_x86_avx_log_file}"
        if [[ $model_name == "mobilenet.tflite" ]]; then
            echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'_posttraining.ms --inDataFile=/home/workspace/mindspore_dataset/mslite/quantTraining/mnist_calibration_data/00099.bin --benchmarkDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/'${model_name}'_posttraining.ms.out' --accuracyThreshold=${accuracy_limit} >> "${run_x86_avx_log_file}"
            ./benchmark --modelFile=${ms_models_path}/${model_name}_posttraining.ms --inDataFile=/home/workspace/mindspore_dataset/mslite/quantTraining/mnist_calibration_data/00099.bin --benchmarkDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/${model_name}_posttraining.ms.out --accuracyThreshold=${accuracy_limit} >> "${run_x86_avx_log_file}"
        fi
        if [[ $model_name == "transformer_20200831_encoder_fp32.tflite" ]]; then
            echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'_posttraining.ms --inDataFile=${transformer_data_path}/encoder_buffer_in_0-35.bin --benchmarkDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/'${model_name}'_posttraining.ms.out' --accuracyThreshold=${accuracy_limit} >> "${run_x86_avx_log_file}"
            ./benchmark --modelFile=${ms_models_path}/${model_name}_posttraining.ms --inDataFile=${transformer_data_path}/encoder_buffer_in_35.bin,${transformer_data_path}/encoder_buffer_in_0.bin,${transformer_data_path}/encoder_buffer_in_1.bin,${transformer_data_path}/encoder_buffer_in_4.bin,${transformer_data_path}/encoder_buffer_in_2.bin,${transformer_data_path}/encoder_buffer_in_3.bin,${transformer_data_path}/encoder_buffer_in_7.bin,${transformer_data_path}/encoder_buffer_in_5.bin,${transformer_data_path}/encoder_buffer_in_6.bin,${transformer_data_path}/encoder_buffer_in_10.bin,${transformer_data_path}/encoder_buffer_in_8.bin,${transformer_data_path}/encoder_buffer_in_9.bin,${transformer_data_path}/encoder_buffer_in_11.bin,${transformer_data_path}/encoder_buffer_in_12.bin,${transformer_data_path}/encoder_buffer_in_15.bin,${transformer_data_path}/encoder_buffer_in_13.bin,${transformer_data_path}/encoder_buffer_in_14.bin,${transformer_data_path}/encoder_buffer_in_18.bin,${transformer_data_path}/encoder_buffer_in_16.bin,${transformer_data_path}/encoder_buffer_in_17.bin,${transformer_data_path}/encoder_buffer_in_21.bin,${transformer_data_path}/encoder_buffer_in_19.bin,${transformer_data_path}/encoder_buffer_in_20.bin,${transformer_data_path}/encoder_buffer_in_22.bin,${transformer_data_path}/encoder_buffer_in_23.bin,${transformer_data_path}/encoder_buffer_in_26.bin,${transformer_data_path}/encoder_buffer_in_24.bin,${transformer_data_path}/encoder_buffer_in_25.bin,${transformer_data_path}/encoder_buffer_in_29.bin,${transformer_data_path}/encoder_buffer_in_27.bin,${transformer_data_path}/encoder_buffer_in_28.bin,${transformer_data_path}/encoder_buffer_in_32.bin,${transformer_data_path}/encoder_buffer_in_30.bin,${transformer_data_path}/encoder_buffer_in_31.bin,${transformer_data_path}/encoder_buffer_in_33.bin,${transformer_data_path}/encoder_buffer_in_34.bin --benchmarkDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/${model_name}_posttraining.ms.out --accuracyThreshold=${accuracy_limit} >> "${run_x86_avx_log_file}"
        fi
        if [[ $model_name == "transformer_20200831_decoder_fp32.tflite" ]]; then
            echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'_posttraining.ms --inDataFile=${transformer_data_path}/encoder_buffer_in_0-10.bin --benchmarkDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/'${model_name}'_posttraining.ms.out' --accuracyThreshold=${accuracy_limit} >> "${run_x86_avx_log_file}"
            ./benchmark --modelFile=${ms_models_path}/${model_name}_posttraining.ms --inDataFile=${transformer_data_path}/decoder_buffer_in_9.bin,${transformer_data_path}/decoder_buffer_in_2.bin,${transformer_data_path}/decoder_buffer_in_0.bin,${transformer_data_path}/decoder_buffer_in_1.bin,${transformer_data_path}/decoder_buffer_in_5.bin,${transformer_data_path}/decoder_buffer_in_3.bin,${transformer_data_path}/decoder_buffer_in_4.bin,${transformer_data_path}/decoder_buffer_in_8.bin,${transformer_data_path}/decoder_buffer_in_6.bin,${transformer_data_path}/decoder_buffer_in_7.bin,${transformer_data_path}/decoder_buffer_in_10.bin --benchmarkDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/${model_name}_posttraining.ms.out --accuracyThreshold=${accuracy_limit} >> "${run_x86_avx_log_file}"
        fi
        if [ $? = 0 ]; then
            run_result='x86_avx: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='x86_avx: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
    done < ${models_tflite_posttraining_config}

    # Run caffe post training quantization converted models:
    while read line; do
        posttraining_line_info=${line}
        if [[ $posttraining_line_info == \#* ]]; then
          continue
        fi
        model_name=`echo ${posttraining_line_info}|awk -F ' ' '{print $1}'`
        accuracy_limit=`echo ${posttraining_line_info}|awk -F ' ' '{print $2}'`
        echo ${model_name} >> "${run_x86_avx_log_file}"
        echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'_posttraining.ms --inDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/input/'${model_name}'_posttraining.ms.bin --benchmarkDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/'${model_name}'_posttraining.ms.out'  --accuracyThreshold=${accuracy_limit} >> "${run_x86_avx_log_file}"
        ./benchmark --modelFile=${ms_models_path}/${model_name}_posttraining.ms --inDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/input/${model_name}_posttraining.ms.bin --benchmarkDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/${model_name}_posttraining.ms.out --accuracyThreshold=${accuracy_limit} >> "${run_x86_avx_log_file}"
        if [ $? = 0 ]; then
            run_result='x86_avx: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='x86_avx: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
    done < ${models_caffe_posttraining_config}

    # Run tflite aware training quantization converted models:
    while read line; do
        model_name=${line}
        if [[ $model_name == \#* ]]; then
          continue
        fi
        echo ${model_name} >> "${run_x86_avx_log_file}"
        echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'.ms --inDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/input/'${model_name}'.ms.bin --benchmarkDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/'${model_name}'.ms.out' >> "${run_x86_avx_log_file}"
        ./benchmark --modelFile=${ms_models_path}/${model_name}.ms --inDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/input/${model_name}.ms.bin --benchmarkDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/${model_name}.ms.out >> "${run_x86_avx_log_file}"
        if [ $? = 0 ]; then
            run_result='x86_avx: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='x86_avx: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
    done < ${models_tflite_awaretraining_config}

    # Run tflite weight quantization converted models:
    while read line; do
        weight_quant_line_info=${line}
        if [[ $weight_quant_line_info == \#* ]]; then
          continue
        fi
        model_name=`echo ${weight_quant_line_info}|awk -F ' ' '{print $1}'`
        accuracy_limit=`echo ${weight_quant_line_info}|awk -F ' ' '{print $2}'`
        echo ${model_name} >> "${run_x86_avx_log_file}"
        echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'.ms --inDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/input/'${model_name}'.ms.bin --benchmarkDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/'${model_name}'.ms.out --accuracyThreshold=${accuracy_limit}' >> "${run_x86_avx_log_file}"
        ./benchmark --modelFile=${ms_models_path}/${model_name}_weightquant.ms --inDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/input/${model_name}.ms.bin --benchmarkDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/${model_name}.ms.out --accuracyThreshold=${accuracy_limit} >> "${run_x86_avx_log_file}"
        if [ $? = 0 ]; then
            run_result='x86_avx: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='x86_avx: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
    done < ${models_tflite_weightquant_config}

    # Run mindir weight quantization converted models:
    while read line; do
        weight_quant_line_info=${line}
        if [[ $weight_quant_line_info == \#* ]]; then
          continue
        fi
        model_name=`echo ${weight_quant_line_info}|awk -F ' ' '{print $1}'`
        accuracy_limit=`echo ${weight_quant_line_info}|awk -F ' ' '{print $2}'`
        echo ${model_name} >> "${run_x86_avx_log_file}"
        echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'.ms --inDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/input/'${model_name}'.ms.bin --benchmarkDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/'${model_name}'.ms.out --accuracyThreshold=${accuracy_limit}' >> "${run_x86_avx_log_file}"
        ./benchmark --modelFile=${ms_models_path}/${model_name}_weightquant.ms --inDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/input/${model_name}.ms.bin --benchmarkDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/${model_name}.ms.out --accuracyThreshold=${accuracy_limit} >> "${run_x86_avx_log_file}"
        if [ $? = 0 ]; then
            run_result='x86_avx: '${model_name}'[weight quant] pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='x86_avx: '${model_name}'[weight quant] failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
    done < ${models_mindspore_weightquant_config}

    # Run mindir mixbit weight quantization converted models:
    while read line; do
        line_info=${line}
        if [[ $line_info == \#* ]]; then
          continue
        fi
        model_name=`echo ${line_info}|awk -F ' ' '{print $1}'`
        accuracy_limit_7bit=`echo ${line_info}|awk -F ' ' '{print $2}'`
        accuracy_limit_9bit=`echo ${line_info}|awk -F ' ' '{print $3}'`

        echo ${model_name} >> "${run_x86_avx_log_file}"
        echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'_7bit.ms --inDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/input/'${model_name}'.ms.bin --benchmarkDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/'${model_name}'.ms.out --accuracyThreshold=${accuracy_limit_7bit}' >> "${run_x86_avx_log_file}"
        ./benchmark --modelFile=${ms_models_path}/${model_name}_7bit.ms --inDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/input/${model_name}.ms.bin --benchmarkDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/${model_name}.ms.out --accuracyThreshold=${accuracy_limit_7bit} >> "${run_x86_avx_log_file}"
        if [ $? = 0 ]; then
            run_result='x86_avx: '${model_name}'_7bit pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='x86_avx: '${model_name}'_7bit failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
        echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'_9bit.ms --inDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/input/'${model_name}'.ms.bin --benchmarkDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/'${model_name}'.ms.out --accuracyThreshold=${accuracy_limit_9bit}' >> "${run_x86_avx_log_file}"
        ./benchmark --modelFile=${ms_models_path}/${model_name}_9bit.ms --inDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/input/${model_name}.ms.bin --benchmarkDataFile=/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/output/${model_name}.ms.out --accuracyThreshold=${accuracy_limit_9bit} >> "${run_x86_avx_log_file}"
        if [ $? = 0 ]; then
            run_result='x86_avx: '${model_name}'_9bit pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='x86_avx: '${model_name}'_9bit failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
    done < ${models_mindspore_mixbit_config}

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
        data_path="/home/workspace/mindspore_dataset/mslite/models/hiai/input_output/"
        for i in $(seq 1 $input_num)
        do
          input_files=$input_files${data_path}'input/'$model_name'.ms.bin_'$i','
        done
        output_file=${data_path}'output/'${model_name}'.ms.out'
        if [[ ${model_name##*.} == "caffemodel" ]]; then
          model_name=${model_name%.*}
        fi
        echo ${model_name} >> "${run_x86_avx_log_file}"
        echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'.ms --inDataFile='${input_files}' --inputShapes='${input_shapes}' --benchmarkDataFile='${output_file} >> "${run_x86_avx_log_file}"
        ./benchmark --modelFile=${ms_models_path}/${model_name}.ms --inDataFile=${input_files} --inputShapes=${input_shapes} --benchmarkDataFile=${output_file} >> "${run_x86_avx_log_file}"
        if [ $? = 0 ]; then
            run_result='x86_avx: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='x86_avx: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
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
        echo ${model_name} >> "${run_x86_avx_log_file}"
        echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'.ms --inputShapes='${input_shapes}' --loopCount=2 --warmUpLoopCount=0' >> "${run_x86_avx_log_file}"
        ./benchmark --modelFile=${ms_models_path}/${model_name}.ms --inputShapes=${input_shapes} --loopCount=2 --warmUpLoopCount=0 >> "${run_x86_avx_log_file}"
        if [ $? = 0 ]; then
            run_result='x86_avx: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='x86_avx: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
    done < ${models_for_process_only_config}
}

# Run on arm64 platform:
function Run_arm64() {
    cd ${arm64_path} || exit 1
    tar -zxf mindspore-lite-${version}-inference-android-aarch64.tar.gz || exit 1

    # If build with minddata, copy the minddata related libs
    cd ${benchmark_test_path} || exit 1
    if [ -f ${arm64_path}/mindspore-lite-${version}-inference-android-aarch64/inference/minddata/lib/libminddata-lite.so ]; then
        cp -a ${arm64_path}/mindspore-lite-${version}-inference-android-aarch64/inference/minddata/lib/libminddata-lite.so ${benchmark_test_path}/libminddata-lite.so || exit 1
    fi
    cp -a ${arm64_path}/mindspore-lite-${version}-inference-android-aarch64/inference/third_party/hiai_ddk/lib/libhiai.so ${benchmark_test_path}/libhiai.so || exit 1
    cp -a ${arm64_path}/mindspore-lite-${version}-inference-android-aarch64/inference/third_party/hiai_ddk/lib/libhiai_ir.so ${benchmark_test_path}/libhiai_ir.so || exit 1
    cp -a ${arm64_path}/mindspore-lite-${version}-inference-android-aarch64/inference/third_party/hiai_ddk/lib/libhiai_ir_build.so ${benchmark_test_path}/libhiai_ir_build.so || exit 1

    cp -a ${arm64_path}/mindspore-lite-${version}-inference-android-aarch64/inference/lib/libmindspore-lite.so ${benchmark_test_path}/libmindspore-lite.so || exit 1
    cp -a ${arm64_path}/mindspore-lite-${version}-inference-android-aarch64/tools/benchmark/benchmark ${benchmark_test_path}/benchmark || exit 1

    # adb push all needed files to the phone
    adb -s ${device_id} push ${benchmark_test_path} /data/local/tmp/ > adb_push_log.txt

    # run adb ,run session ,check the result:
    echo 'cd  /data/local/tmp/benchmark_test' > adb_cmd.txt
    echo 'cp  /data/local/tmp/libc++_shared.so ./' >> adb_cmd.txt
    echo 'chmod 777 benchmark' >> adb_cmd.txt

    adb -s ${device_id} shell < adb_cmd.txt

    # Run compatibility test models:
    while read line; do
        model_name=${line}
        if [[ $model_name == \#* ]]; then
          continue
        fi
        echo ${model_name} >> "${run_arm64_fp32_log_file}"
        echo 'cd  /data/local/tmp/benchmark_test' > adb_run_cmd.txt
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --modelFile='${model_name}'.ms' >> "${run_arm64_fp32_log_file}"
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --modelFile='${model_name}'.ms' >> adb_run_cmd.txt
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
            run_result='arm64: '${model_name}'[weightQuant] pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='arm64: '${model_name}'[weightQuant] failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
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
}

# Run on arm32 platform:
function Run_arm32() {
    cd ${arm32_path} || exit 1
    tar -zxf mindspore-lite-${version}-inference-android-aarch32.tar.gz || exit 1

    # If build with minddata, copy the minddata related libs
    cd ${benchmark_test_path} || exit 1
    if [ -f ${arm32_path}/mindspore-lite-${version}-inference-android-aarch32/inference/minddata/lib/libminddata-lite.so ]; then
        cp -a ${arm32_path}/mindspore-lite-${version}-inference-android-aarch32/inference/minddata/lib/libminddata-lite.so ${benchmark_test_path}/libminddata-lite.so || exit 1
    fi

    cp -a ${arm32_path}/mindspore-lite-${version}-inference-android-aarch32/inference/lib/libmindspore-lite.so ${benchmark_test_path}/libmindspore-lite.so || exit 1
    cp -a ${arm32_path}/mindspore-lite-${version}-inference-android-aarch32/tools/benchmark/benchmark ${benchmark_test_path}/benchmark || exit 1

    # adb push all needed files to the phone
    adb -s ${device_id} push ${benchmark_test_path} /data/local/tmp/ > adb_push_log.txt

    # run adb ,run session ,check the result:
    echo 'cd  /data/local/tmp/benchmark_test' > adb_cmd.txt
    echo 'cp  /data/local/tmp/arm32/libc++_shared.so ./' >> adb_cmd.txt
    echo 'chmod 777 benchmark' >> adb_cmd.txt

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
}

# Run on arm64-fp16 platform:
function Run_arm64_fp16() {
    cd ${arm64_path} || exit 1
    tar -zxf mindspore-lite-${version}-inference-android-aarch64.tar.gz || exit 1

    # If build with minddata, copy the minddata related libs
    cd ${benchmark_test_path} || exit 1
    if [ -f ${arm64_path}/mindspore-lite-${version}-inference-android-aarch64/inference/minddata/lib/libminddata-lite.so ]; then
        cp -a ${arm64_path}/mindspore-lite-${version}-inference-android-aarch64/inference/minddata/lib/libminddata-lite.so ${benchmark_test_path}/libminddata-lite.so || exit 1
    fi
    cp -a ${arm64_path}/mindspore-lite-${version}-inference-android-aarch64/inference/third_party/hiai_ddk/lib/libhiai.so ${benchmark_test_path}/libhiai.so || exit 1
    cp -a ${arm64_path}/mindspore-lite-${version}-inference-android-aarch64/inference/third_party/hiai_ddk/lib/libhiai_ir.so ${benchmark_test_path}/libhiai_ir.so || exit 1
    cp -a ${arm64_path}/mindspore-lite-${version}-inference-android-aarch64/inference/third_party/hiai_ddk/lib/libhiai_ir_build.so ${benchmark_test_path}/libhiai_ir_build.so || exit 1

    cp -a ${arm64_path}/mindspore-lite-${version}-inference-android-aarch64/inference/lib/libmindspore-lite.so ${benchmark_test_path}/libmindspore-lite.so || exit 1
    cp -a ${arm64_path}/mindspore-lite-${version}-inference-android-aarch64/tools/benchmark/benchmark ${benchmark_test_path}/benchmark || exit 1

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
# Run on gpu platform:
function Run_gpu() {
    cd ${arm64_path} || exit 1
    tar -zxf mindspore-lite-${version}-inference-android-aarch64.tar.gz || exit 1

    # If build with minddata, copy the minddata related libs
    cd ${benchmark_test_path} || exit 1
    if [ -f ${arm64_path}/mindspore-lite-${version}-inference-android-aarch64/inference/minddata/lib/libminddata-lite.so ]; then
        cp -a ${arm64_path}/mindspore-lite-${version}-inference-android-aarch64/inference/minddata/lib/libminddata-lite.so ${benchmark_test_path}/libminddata-lite.so || exit 1
    fi
    cp -a ${arm64_path}/mindspore-lite-${version}-inference-android-aarch64/inference/third_party/hiai_ddk/lib/libhiai.so ${benchmark_test_path}/libhiai.so || exit 1
    cp -a ${arm64_path}/mindspore-lite-${version}-inference-android-aarch64/inference/third_party/hiai_ddk/lib/libhiai_ir.so ${benchmark_test_path}/libhiai_ir.so || exit 1
    cp -a ${arm64_path}/mindspore-lite-${version}-inference-android-aarch64/inference/third_party/hiai_ddk/lib/libhiai_ir_build.so ${benchmark_test_path}/libhiai_ir_build.so || exit 1

    cp -a ${arm64_path}/mindspore-lite-${version}-inference-android-aarch64/inference/lib/libmindspore-lite.so ${benchmark_test_path}/libmindspore-lite.so || exit 1
    cp -a ${arm64_path}/mindspore-lite-${version}-inference-android-aarch64/tools/benchmark/benchmark ${benchmark_test_path}/benchmark || exit 1

    # adb push all needed files to the phone
    adb -s ${device_id} push ${benchmark_test_path} /data/local/tmp/ > adb_push_log.txt

    # run adb ,run session ,check the result:
    echo 'cd  /data/local/tmp/benchmark_test' > adb_cmd.txt
    echo 'cp  /data/local/tmp/libc++_shared.so ./' >> adb_cmd.txt
    echo 'chmod 777 benchmark' >> adb_cmd.txt

    adb -s ${device_id} shell < adb_cmd.txt

    # Run gpu tflite converted models:
    while read line; do
        model_name=${line}
        if [[ $model_name == \#* ]]; then
          continue
        fi
        echo ${model_name} >> "${run_gpu_log_file}"
        echo 'cd  /data/local/tmp/benchmark_test' > adb_run_cmd.txt
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --device=GPU --modelFile='${model_name}'.ms --inDataFile=/data/local/tmp/input_output/input/'${model_name}'.ms.bin --benchmarkDataFile=/data/local/tmp/input_output/output/'${model_name}'.ms.out' >> "${run_gpu_log_file}"
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --device=GPU --modelFile='${model_name}'.ms --inDataFile=/data/local/tmp/input_output/input/'${model_name}'.ms.bin --benchmarkDataFile=/data/local/tmp/input_output/output/'${model_name}'.ms.out' >> adb_run_cmd.txt
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
        echo ${model_name} >> "${run_gpu_log_file}"
        echo 'cd  /data/local/tmp/benchmark_test' > adb_run_cmd.txt
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --device=GPU --modelFile='${model_name}'.ms --inDataFile=/data/local/tmp/input_output/input/'${model_name}'.ms.bin --benchmarkDataFile=/data/local/tmp/input_output/output/'${model_name}'.ms.out --enableFp16=true --accuracyThreshold=5' >> "${run_gpu_log_file}"
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --device=GPU --modelFile='${model_name}'.ms --inDataFile=/data/local/tmp/input_output/input/'${model_name}'.ms.bin --benchmarkDataFile=/data/local/tmp/input_output/output/'${model_name}'.ms.out --enableFp16=true --accuracyThreshold=5' >> adb_run_cmd.txt
        adb -s ${device_id} shell < adb_run_cmd.txt >> "${run_gpu_log_file}"
        if [ $? = 0 ]; then
            run_result='arm64_gpu_fp16: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='arm64_gpu_fp16: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
    #sleep 1
    done < ${models_gpu_fp16_config}

    # Run GPU weightquant converted models:
    while read line; do
        model_name=${line}
        if [[ $model_name == \#* ]]; then
          continue
        fi
        echo ${model_name} >> "${run_gpu_log_file}"
        echo 'cd  /data/local/tmp/benchmark_test' > adb_run_cmd.txt
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --device=GPU --modelFile='${model_name}'_weightquant.ms --inDataFile=/data/local/tmp/input_output/input/'${model_name}'.ms.bin --benchmarkDataFile=/data/local/tmp/input_output/output/'${model_name}'.ms.out --enableFp16=true --accuracyThreshold=5' >> "${run_gpu_log_file}"
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --device=GPU --modelFile='${model_name}'_weightquant.ms --inDataFile=/data/local/tmp/input_output/input/'${model_name}'.ms.bin --benchmarkDataFile=/data/local/tmp/input_output/output/'${model_name}'.ms.out --enableFp16=true --accuracyThreshold=5' >> adb_run_cmd.txt
        adb -s ${device_id} shell < adb_run_cmd.txt >> "${run_gpu_log_file}"
        if [ $? = 0 ]; then
            run_result='arm64_gpu_weightquant: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='arm64_gpu_weightquant: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
    #sleep 1
    done < ${models_gpu_weightquant_config}
}

# Run on npu platform:
function Run_npu() {
    cd ${arm64_path} || exit 1
    tar -zxf mindspore-lite-${version}-inference-android-aarch64.tar.gz || exit 1

    # If build with minddata, copy the minddata related libs
    cd ${benchmark_test_path} || exit 1
    if [ -f ${arm64_path}/mindspore-lite-${version}-inference-android-aarch64/inference/minddata/lib/libminddata-lite.so ]; then
        cp -a ${arm64_path}/mindspore-lite-${version}-inference-android-aarch64/inference/minddata/lib/libminddata-lite.so ${benchmark_test_path}/libminddata-lite.so || exit 1
    fi
    cp -a ${arm64_path}/mindspore-lite-${version}-inference-android-aarch64/inference/third_party/hiai_ddk/lib/libhiai.so ${benchmark_test_path}/libhiai.so || exit 1
    cp -a ${arm64_path}/mindspore-lite-${version}-inference-android-aarch64/inference/third_party/hiai_ddk/lib/libhiai_ir.so ${benchmark_test_path}/libhiai_ir.so || exit 1
    cp -a ${arm64_path}/mindspore-lite-${version}-inference-android-aarch64/inference/third_party/hiai_ddk/lib/libhiai_ir_build.so ${benchmark_test_path}/libhiai_ir_build.so || exit 1

    cp -a ${arm64_path}/mindspore-lite-${version}-inference-android-aarch64/inference/lib/libmindspore-lite.so ${benchmark_test_path}/libmindspore-lite.so || exit 1
    cp -a ${arm64_path}/mindspore-lite-${version}-inference-android-aarch64/tools/benchmark/benchmark ${benchmark_test_path}/benchmark || exit 1

    # adb push all needed files to the phone
    adb -s ${device_id} push ${benchmark_test_path} /data/local/tmp/ > adb_push_log.txt

    # run adb ,run session ,check the result:
    echo 'cd  /data/local/tmp/benchmark_test' > adb_cmd.txt
    echo 'cp  /data/local/tmp/libc++_shared.so ./' >> adb_cmd.txt
    echo 'chmod 777 benchmark' >> adb_cmd.txt

    adb -s ${device_id} shell < adb_cmd.txt

    # Run npu converted models:
    while read line; do
        model_line_info=${line}
        if [[ $model_line_info == \#* ]]; then
          continue
        fi
        model_name=`echo ${line}|awk -F ' ' '{print $1}'`
        accuracy_limit=`echo ${line}|awk -F ' ' '{print $2}'`
        input_num=`echo ${line}|awk -F ' ' '{print $3}'`
        data_path="/data/local/tmp/input_output/"
        input_files=''
        if [[ -z "$input_num" || $input_num == 1 ]]; then
          input_files=${data_path}'input/'$model_name'.ms.bin'
        elif [[ ! -z "$input_num" && $input_num -gt 1 ]]; then
          for i in $(seq 1 $input_num)
          do
            input_files=$input_files${data_path}'input/'$model_name'.ms.bin_'$i','
          done
        fi
        echo "mindspore run npu: ${model_name}, accuracy limit:${accuracy_limit}" >> "${run_npu_log_file}"
        echo 'cd  /data/local/tmp/benchmark_test' > adb_run_cmd.txt
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --device=NPU --modelFile='${model_name}'.ms --inDataFile='${input_files}' --benchmarkDataFile=/data/local/tmp/input_output/output/'${model_name}'.ms.out --accuracyThreshold='${accuracy_limit} >> "${run_npu_log_file}"
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --device=NPU --modelFile='${model_name}'.ms --inDataFile='${input_files}' --benchmarkDataFile=/data/local/tmp/input_output/output/'${model_name}'.ms.out --accuracyThreshold='${accuracy_limit} >> adb_run_cmd.txt
        adb -s ${device_id} shell < adb_run_cmd.txt >> "${run_npu_log_file}"
        if [ $? = 0 ]; then
            run_result='arm64_npu: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='arm64_npu: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
    done < ${models_npu_config}
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

# Example:sh run_benchmark_nets.sh -r /home/temp_test -m /home/temp_test/models -d "8KE5T19620002408" -e arm_cpu
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
models_tflite_config=${basepath}/models_tflite.cfg
models_tf_config=${basepath}/models_tf.cfg
models_caffe_config=${basepath}/models_caffe.cfg
models_tflite_awaretraining_config=${basepath}/models_tflite_awaretraining.cfg
models_tflite_posttraining_config=${basepath}/models_tflite_posttraining.cfg
models_caffe_posttraining_config=${basepath}/models_caffe_posttraining.cfg
models_tflite_weightquant_config=${basepath}/models_tflite_weightquant.cfg
models_onnx_config=${basepath}/models_onnx.cfg
models_onnx_fp16_config=${basepath}/models_onnx_fp16.cfg
models_caffe_fp16_config=${basepath}/models_caffe_fp16.cfg
models_tflite_fp16_config=${basepath}/models_tflite_fp16.cfg
models_tf_fp16_config=${basepath}/models_tf_fp16.cfg
models_multiple_inputs_fp16_config=${basepath}/models_with_multiple_inputs_fp16.cfg
models_mindspore_config=${basepath}/models_mindspore.cfg
models_mindspore_train_config=${basepath}/models_mindspore_train.cfg
models_mindspore_mixbit_config=${basepath}/models_mindspore_mixbit.cfg
models_gpu_fp32_config=${basepath}/models_gpu_fp32.cfg
models_gpu_fp16_config=${basepath}/models_gpu_fp16.cfg
models_gpu_weightquant_config=${basepath}/models_gpu_weightquant.cfg
models_mindspore_weightquant_config=${basepath}/models_mindspore_weightquant.cfg
models_arm32_config=${basepath}/models_arm32.cfg
models_npu_config=${basepath}/models_npu.cfg
models_compatibility_config=${basepath}/models_compatibility.cfg
models_with_multiple_inputs_config=${basepath}/models_with_multiple_inputs.cfg
models_for_process_only_config=${basepath}/models_for_process_only.cfg

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

run_x86_log_file=${basepath}/run_x86_log.txt
echo 'run x86 logs: ' > ${run_x86_log_file}

run_x86_sse_log_file=${basepath}/run_x86_sse_log.txt
echo 'run x86 sse logs: ' > ${run_x86_sse_log_file}

run_x86_avx_log_file=${basepath}/run_x86_avx_log.txt
echo 'run x86 avx logs: ' > ${run_x86_avx_log_file}

run_arm64_fp32_log_file=${basepath}/run_arm64_fp32_log.txt
echo 'run arm64_fp32 logs: ' > ${run_arm64_fp32_log_file}

run_arm64_fp16_log_file=${basepath}/run_arm64_fp16_log.txt
echo 'run arm64_fp16 logs: ' > ${run_arm64_fp16_log_file}

run_arm32_log_file=${basepath}/run_arm32_log.txt
echo 'run arm32 logs: ' > ${run_arm32_log_file}

run_gpu_log_file=${basepath}/run_gpu_log.txt
echo 'run gpu logs: ' > ${run_gpu_log_file}

run_npu_log_file=${basepath}/run_npu_log.txt
echo 'run npu logs: ' > ${run_npu_log_file}

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

if [[ $backend == "all" || $backend == "x86-all" || $backend == "x86" ]]; then
    # Run on x86
    echo "start Run x86 ..."
    Run_x86 &
    Run_x86_PID=$!
    sleep 1
fi
if [[ $backend == "all" || $backend == "x86-all" || $backend == "x86-sse" ]]; then
    # Run on x86-sse
    echo "start Run x86 sse ..."
    Run_x86_sse &
    Run_x86_sse_PID=$!
    sleep 1
fi
if [[ $backend == "all" || $backend == "x86-all" || $backend == "x86-avx" ]]; then
    # Run on x86-avx
    echo "start Run x86 avx ..."
    Run_x86_avx &
    Run_x86_avx_PID=$!
    sleep 1
fi

if [[ $backend == "all" || $backend == "arm_cpu" || $backend == "arm64_fp32" ]]; then
    # Run on arm64
    arm64_path=${release_path}/android_aarch64
    # mv ${arm64_path}/*train-android-aarch64* ./train
    file_name=$(ls ${arm64_path}/*inference-android-aarch64.tar.gz)
    IFS="-" read -r -a file_name_array <<< "$file_name"
    version=${file_name_array[2]}

    echo "start Run arm64 ..."
    Run_arm64
    Run_arm64_fp32_status=$?
    sleep 1
fi

if [[ $backend == "all" || $backend == "arm_cpu" || $backend == "arm64_fp16" ]]; then
    # Run on arm64-fp16
    arm64_path=${release_path}/android_aarch64
    # mv ${arm64_path}/*train-android-aarch64* ./train
    file_name=$(ls ${arm64_path}/*inference-android-aarch64.tar.gz)
    IFS="-" read -r -a file_name_array <<< "$file_name"
    version=${file_name_array[2]}

    echo "start Run arm64-fp16 ..."
    Run_arm64_fp16
    Run_arm64_fp16_status=$?
    sleep 1
fi

if [[ $backend == "all" || $backend == "arm_cpu" || $backend == "arm32" ]]; then
    # Run on arm32
    arm32_path=${release_path}/android_aarch32
    # mv ${arm32_path}/*train-android-aarch32* ./train
    file_name=$(ls ${arm32_path}/*inference-android-aarch32.tar.gz)
    IFS="-" read -r -a file_name_array <<< "$file_name"
    version=${file_name_array[2]}

    echo "start Run arm32 ..."
    Run_arm32
    Run_arm32_status=$?
    sleep 1
fi

if [[ $backend == "all" || $backend == "gpu_npu" || $backend == "gpu" ]]; then
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
if [[ $backend == "all" || $backend == "gpu_npu" || $backend == "npu" ]]; then
    # Run on npu
    arm64_path=${release_path}/android_aarch64
    # mv ${arm64_path}/*train-android-aarch64* ./train
    file_name=$(ls ${arm64_path}/*inference-android-aarch64.tar.gz)
    IFS="-" read -r -a file_name_array <<< "$file_name"
    version=${file_name_array[2]}

    echo "start Run npu ..."
    Run_npu
    Run_npu_status=$?
    sleep 1
fi

if [[ $backend == "all" || $backend == "x86-all" || $backend == "x86-sse" ]]; then
    wait ${Run_x86_sse_PID}
    Run_x86_sse_status=$?

    if [[ ${Run_x86_sse_status} != 0 ]];then
        echo "Run_x86 sse failed"
        cat ${run_x86_sse_log_file}
        isFailed=1
    fi
fi
if [[ $backend == "all" || $backend == "x86-all" || $backend == "x86-avx" ]]; then
    wait ${Run_x86_avx_PID}
    Run_x86_avx_status=$?

    if [[ ${Run_x86_avx_status} != 0 ]];then
        echo "Run_x86 avx failed"
        cat ${run_x86_avx_log_file}
        isFailed=1
    fi
fi

if [[ $backend == "all" || $backend == "x86-all" || $backend == "x86" ]]; then
    wait ${Run_x86_PID}
    Run_x86_status=$?

    # Check benchmark result and return value
    if [[ ${Run_x86_status} != 0 ]];then
        echo "Run_x86 failed"
        cat ${run_x86_log_file}
        isFailed=1
    fi
fi

if [[ $backend == "all" || $backend == "arm_cpu" || $backend == "arm64_fp32" ]]; then
    if [[ ${Run_arm64_fp32_status} != 0 ]];then
        echo "Run_arm64_fp32 failed"
        cat ${run_arm64_fp32_log_file}
        isFailed=1
    fi
fi
if [[ $backend == "all" || $backend == "arm_cpu" || $backend == "arm64_fp16" ]]; then
    if [[ ${Run_arm64_fp16_status} != 0 ]];then
        echo "Run_arm64_fp16 failed"
        cat ${run_arm64_fp16_log_file}
        isFailed=1
    fi
fi
if [[ $backend == "all" || $backend == "arm_cpu" || $backend == "arm32" ]]; then
    if [[ ${Run_arm32_status} != 0 ]];then
        echo "Run_arm32 failed"
        cat ${run_arm32_log_file}
        isFailed=1
    fi
fi
if [[ $backend == "all" || $backend == "gpu_npu" || $backend == "gpu" ]]; then
    if [[ ${Run_gpu_status} != 0 ]];then
        echo "Run_gpu failed"
        cat ${run_gpu_log_file}
        isFailed=1
    fi
fi
if [[ $backend == "all" || $backend == "gpu_npu" || $backend == "npu" ]]; then
    if [[ ${Run_npu_status} != 0 ]];then
        echo "Run_npu failed"
        cat ${run_npu_log_file}
        isFailed=1
    fi
fi

echo "Run_x86 and Run_x86_sse and Run_arm64_fp32 and Run_arm64_fp16 and Run_arm32 and Run_gpu and Run_npu is ended"
Print_Benchmark_Result
if [[ $isFailed == 1 ]]; then
    exit 1
fi
exit 0
