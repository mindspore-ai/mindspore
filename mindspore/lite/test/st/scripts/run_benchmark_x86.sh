#!/bin/bash

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
        if [[ $model_name == \#* ]]; then
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
        echo './converter_lite  --fmk=CAFFE --modelFile='${models_path}'/'${model_name}'.prototxt --weightFile='${models_path}'/'${model_name}'.caffemodel --outputFile='${ms_models_path}'/'${model_name}_weightquant' --quantType=WeightQuant --bitNum=8 --quantWeightChannel=0' >> "${run_converter_log_file}"
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
}

# Run on x86 platform:
function Run_x86() {
    echo 'cd  '${x86_path}'/mindspore-lite-'${version}'-linux-x64' >> "${run_x86_log_file}"
    cd ${x86_path}/mindspore-lite-${version}-linux-x64 || return 1
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./runtime/lib:./runtime/third_party/libjpeg-turbo/lib
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
        echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'.ms --inDataFile='${models_path}'/input_output/input/'${model_name}'.ms.bin --inputShapes='${input_shapes}' --benchmarkDataFile='${models_path}'/input_output/output/'${model_name}'.ms.out' >> "${run_x86_log_file}"
        ./benchmark --modelFile=${ms_models_path}/${model_name}.ms --inDataFile=${models_path}/input_output/input/${model_name}.ms.bin --inputShapes=${input_shapes} --benchmarkDataFile=${models_path}/input_output/output/${model_name}.ms.out >> "${run_x86_log_file}"
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
        echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'.ms --inDataFile='${models_path}'/input_output/input/'${model_name}'.ms.bin --benchmarkDataFile='${models_path}'/input_output/output/'${model_name}'.ms.out' >> "${run_x86_log_file}"
        ./benchmark --modelFile=${ms_models_path}/${model_name}.ms --inDataFile=${models_path}/input_output/input/${model_name}.ms.bin --benchmarkDataFile=${models_path}/input_output/output/${model_name}.ms.out >> "${run_x86_log_file}"
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
        echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'.ms --inDataFile='${models_path}'/input_output/input/'${model_name}'.ms.bin --benchmarkDataFile='${models_path}'/input_output/output/'${model_name}'.ms.out' >> "${run_x86_log_file}"
        ./benchmark --modelFile=${ms_models_path}/${model_name}.ms --inDataFile=${models_path}/input_output/input/${model_name}.ms.bin --benchmarkDataFile=${models_path}/input_output/output/${model_name}.ms.out >> "${run_x86_log_file}"
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
        echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'.ms --inDataFile='${models_path}'/input_output/input/'${model_name}'.ms.bin --inputShapes='${input_shapes}' --benchmarkDataFile='${models_path}'/input_output/output/'${model_name}'.ms.out' >> "${run_x86_log_file}"
        ./benchmark --modelFile=${ms_models_path}/${model_name}.ms --inDataFile=${models_path}/input_output/input/${model_name}.ms.bin --inputShapes=${input_shapes} --benchmarkDataFile=${models_path}/input_output/output/${model_name}.ms.out >> "${run_x86_log_file}"
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
        echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'.ms --inDataFile='${models_path}'/input_output/input/'${model_name}'.ms.bin --benchmarkDataFile='${models_path}'/input_output/output/'${model_name}'.ms.out' >> "${run_x86_log_file}"
        ./benchmark --modelFile=${ms_models_path}/${model_name}.ms --inDataFile=${models_path}/input_output/input/${model_name}.ms.bin --benchmarkDataFile=${models_path}/input_output/output/${model_name}.ms.out --accuracyThreshold=${accuracy_limit} >> "${run_x86_log_file}"
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
        echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'_train.ms --inDataFile='${models_path}'/input_output/input/'${model_name}'.ms.bin --benchmarkDataFile=${models_path}/input_output/output/'${model_name}'.train.ms.out' >> "${run_x86_log_file}"
        ./benchmark --modelFile=${ms_models_path}/${model_name}'_train'.ms --inDataFile=${models_path}/input_output/input/${model_name}.ms.bin --benchmarkDataFile=${models_path}/input_output/output/${model_name}.train.ms.out --accuracyThreshold=1.5 >> "${run_x86_log_file}"
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
        transformer_data_path="${models_path}/input_output/input"
        echo ${model_name} >> "${run_x86_log_file}"
        if [[ $model_name == "mobilenet.tflite" ]]; then
            echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'_posttraining.ms --inDataFile=/home/workspace/mindspore_dataset/mslite/quantTraining/mnist_calibration_data/00099.bin --benchmarkDataFile='${models_path}'/input_output/output/'${model_name}'_posttraining.ms.out' --accuracyThreshold=${accuracy_limit} >> "${run_x86_log_file}"
            ./benchmark --modelFile=${ms_models_path}/${model_name}_posttraining.ms --inDataFile=/home/workspace/mindspore_dataset/mslite/quantTraining/mnist_calibration_data/00099.bin --benchmarkDataFile=${models_path}/input_output/output/${model_name}_posttraining.ms.out --accuracyThreshold=${accuracy_limit} >> "${run_x86_log_file}"
        fi
        if [[ $model_name == "transformer_20200831_encoder_fp32.tflite" ]]; then
            echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'_posttraining.ms --inDataFile=${transformer_data_path}/encoder_buffer_in_0-35.bin --benchmarkDataFile='${models_path}'/input_output/output/'${model_name}'_posttraining.ms.out' --accuracyThreshold=${accuracy_limit} >> "${run_x86_log_file}"
            ./benchmark --modelFile=${ms_models_path}/${model_name}_posttraining.ms --inDataFile=${transformer_data_path}/encoder_buffer_in_35.bin,${transformer_data_path}/encoder_buffer_in_0.bin,${transformer_data_path}/encoder_buffer_in_1.bin,${transformer_data_path}/encoder_buffer_in_4.bin,${transformer_data_path}/encoder_buffer_in_2.bin,${transformer_data_path}/encoder_buffer_in_3.bin,${transformer_data_path}/encoder_buffer_in_7.bin,${transformer_data_path}/encoder_buffer_in_5.bin,${transformer_data_path}/encoder_buffer_in_6.bin,${transformer_data_path}/encoder_buffer_in_10.bin,${transformer_data_path}/encoder_buffer_in_8.bin,${transformer_data_path}/encoder_buffer_in_9.bin,${transformer_data_path}/encoder_buffer_in_11.bin,${transformer_data_path}/encoder_buffer_in_12.bin,${transformer_data_path}/encoder_buffer_in_15.bin,${transformer_data_path}/encoder_buffer_in_13.bin,${transformer_data_path}/encoder_buffer_in_14.bin,${transformer_data_path}/encoder_buffer_in_18.bin,${transformer_data_path}/encoder_buffer_in_16.bin,${transformer_data_path}/encoder_buffer_in_17.bin,${transformer_data_path}/encoder_buffer_in_21.bin,${transformer_data_path}/encoder_buffer_in_19.bin,${transformer_data_path}/encoder_buffer_in_20.bin,${transformer_data_path}/encoder_buffer_in_22.bin,${transformer_data_path}/encoder_buffer_in_23.bin,${transformer_data_path}/encoder_buffer_in_26.bin,${transformer_data_path}/encoder_buffer_in_24.bin,${transformer_data_path}/encoder_buffer_in_25.bin,${transformer_data_path}/encoder_buffer_in_29.bin,${transformer_data_path}/encoder_buffer_in_27.bin,${transformer_data_path}/encoder_buffer_in_28.bin,${transformer_data_path}/encoder_buffer_in_32.bin,${transformer_data_path}/encoder_buffer_in_30.bin,${transformer_data_path}/encoder_buffer_in_31.bin,${transformer_data_path}/encoder_buffer_in_33.bin,${transformer_data_path}/encoder_buffer_in_34.bin --benchmarkDataFile=${models_path}/input_output/output/${model_name}_posttraining.ms.out --accuracyThreshold=${accuracy_limit} >> "${run_x86_log_file}"
        fi
        if [[ $model_name == "transformer_20200831_decoder_fp32.tflite" ]]; then
            echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'_posttraining.ms --inDataFile=${transformer_data_path}/decoder_buffer_in_0-10.bin --benchmarkDataFile='${models_path}'/input_output/output/'${model_name}'_posttraining.ms.out' --accuracyThreshold=${accuracy_limit} >> "${run_x86_log_file}"
            ./benchmark --modelFile=${ms_models_path}/${model_name}_posttraining.ms --inDataFile=${transformer_data_path}/decoder_buffer_in_9.bin,${transformer_data_path}/decoder_buffer_in_2.bin,${transformer_data_path}/decoder_buffer_in_0.bin,${transformer_data_path}/decoder_buffer_in_1.bin,${transformer_data_path}/decoder_buffer_in_5.bin,${transformer_data_path}/decoder_buffer_in_3.bin,${transformer_data_path}/decoder_buffer_in_4.bin,${transformer_data_path}/decoder_buffer_in_8.bin,${transformer_data_path}/decoder_buffer_in_6.bin,${transformer_data_path}/decoder_buffer_in_7.bin,${transformer_data_path}/decoder_buffer_in_10.bin --benchmarkDataFile=${models_path}/input_output/output/${model_name}_posttraining.ms.out --accuracyThreshold=${accuracy_limit} >> "${run_x86_log_file}"
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
        echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'_posttraining.ms --inDataFile='${models_path}'/input_output/input/'${model_name}'_posttraining.ms.bin --benchmarkDataFile='${models_path}'/input_output/output/'${model_name}'_posttraining.ms.out'  --accuracyThreshold=${accuracy_limit} >> "${run_x86_log_file}"
        ./benchmark --modelFile=${ms_models_path}/${model_name}_posttraining.ms --inDataFile=${models_path}/input_output/input/${model_name}_posttraining.ms.bin --benchmarkDataFile=${models_path}/input_output/output/${model_name}_posttraining.ms.out --accuracyThreshold=${accuracy_limit} >> "${run_x86_log_file}"
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
        echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'.ms --inDataFile='${models_path}'/input_output/input/'${model_name}'.ms.bin --benchmarkDataFile='${models_path}'/input_output/output/'${model_name}'.ms.out' >> "${run_x86_log_file}"
        ./benchmark --modelFile=${ms_models_path}/${model_name}.ms --inDataFile=${models_path}/input_output/input/${model_name}.ms.bin --benchmarkDataFile=${models_path}/input_output/output/${model_name}.ms.out >> "${run_x86_log_file}"
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
        echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'.ms --inDataFile='${models_path}'/input_output/input/'${model_name}'.ms.bin --benchmarkDataFile='${models_path}'/input_output/output/'${model_name}'.ms.out --accuracyThreshold=${accuracy_limit}' >> "${run_x86_log_file}"
        ./benchmark --modelFile=${ms_models_path}/${model_name}_weightquant.ms --inDataFile=${models_path}/input_output/input/${model_name}.ms.bin --benchmarkDataFile=${models_path}/input_output/output/${model_name}.ms.out --accuracyThreshold=${accuracy_limit}>> "${run_x86_log_file}"
        if [ $? = 0 ]; then
            run_result='x86: '${model_name}_weightquant' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='x86: '${model_name}_weightquant' failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
    done < ${models_tflite_weightquant_config}

    # Run caffe weightquant converted models:
    while read line; do
        weight_quant_line_info=${line}
        if [[ ${weight_quant_line_info} == \#* ]]; then
          continue
        fi
        model_name=`echo ${weight_quant_line_info}|awk -F ' ' '{print $1}'`
        accuracy_limit=`echo ${weight_quant_line_info}|awk -F ' ' '{print $2}'`
        echo ${model_name} >> "${run_x86_log_file}"
        echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'.ms --inDataFile='${models_path}'/input_output/input/'${model_name}'.ms.bin --benchmarkDataFile='${models_path}'/input_output/output/'${model_name}'.ms.out --accuracyThreshold=${accuracy_limit}' >> "${run_x86_log_file}"
        ./benchmark --modelFile=${ms_models_path}/${model_name}_weightquant.ms --inDataFile=${models_path}/input_output/input/${model_name}.ms.bin --benchmarkDataFile=${models_path}/input_output/output/${model_name}.ms.out --accuracyThreshold=${accuracy_limit}>> "${run_x86_log_file}"
        if [ $? = 0 ]; then
            run_result='x86: '${model_name}_weightquant' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='x86: '${model_name}_weightquant' failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
    done < ${models_caffe_weightquant_config}

    # Run tf weightquant converted models:
    while read line; do
        weight_quant_line_info=${line}
        if [[ $weight_quant_line_info == \#* ]]; then
          continue
        fi
        model_name=`echo ${weight_quant_line_info}|awk -F ' ' '{print $1}'`
        accuracy_limit=`echo ${weight_quant_line_info}|awk -F ' ' '{print $2}'`
        echo ${model_name} >> "${run_x86_log_file}"
        echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'.ms --inDataFile='${models_path}'/input_output/input/'${model_name}'.ms.bin --inputShapes='${input_shapes}' --benchmarkDataFile='${models_path}'/input_output/output/'${model_name}'.ms.out' >> "${run_x86_log_file}"
        ./benchmark --modelFile=${ms_models_path}/${model_name}_weightquant.ms --inDataFile=${models_path}/input_output/input/${model_name}.ms.bin --benchmarkDataFile=${models_path}/input_output/output/${model_name}.ms.out --accuracyThreshold=${accuracy_limit}>> "${run_x86_log_file}"
        if [ $? = 0 ]; then
            run_result='x86: '${model_name}_weightquant' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='x86: '${model_name}_weightquant' failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
    done < ${models_tf_weightquant_config}

    # Run mindir weight quantization converted models:
    while read line; do
        weight_quant_line_info=${line}
        if [[ $weight_quant_line_info == \#* ]]; then
          continue
        fi
        model_name=`echo ${weight_quant_line_info}|awk -F ' ' '{print $1}'`
        accuracy_limit=`echo ${weight_quant_line_info}|awk -F ' ' '{print $2}'`
        echo ${model_name} >> "${run_x86_log_file}"
        echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'.ms --inDataFile='${models_path}'/input_output/input/'${model_name}'.ms.bin --benchmarkDataFile='${models_path}'/input_output/output/'${model_name}'.ms.out --accuracyThreshold=${accuracy_limit}' >> "${run_x86_log_file}"
        ./benchmark --modelFile=${ms_models_path}/${model_name}_weightquant.ms --inDataFile=${models_path}/input_output/input/${model_name}.ms.bin --benchmarkDataFile=${models_path}/input_output/output/${model_name}.ms.out --accuracyThreshold=${accuracy_limit} >> "${run_x86_log_file}"
        if [ $? = 0 ]; then
            run_result='x86: '${model_name}'_weightquant pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='x86: '${model_name}'_weightquant failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
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
        echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'_7bit.ms --inDataFile='${models_path}'/input_output/input/'${model_name}'.ms.bin --benchmarkDataFile='${models_path}'/input_output/output/'${model_name}'.ms.out --accuracyThreshold=${accuracy_limit_7bit}' >> "${run_x86_log_file}"
        ./benchmark --modelFile=${ms_models_path}/${model_name}_7bit.ms --inDataFile=${models_path}/input_output/input/${model_name}.ms.bin --benchmarkDataFile=${models_path}/input_output/output/${model_name}.ms.out --accuracyThreshold=${accuracy_limit_7bit} >> "${run_x86_log_file}"
        if [ $? = 0 ]; then
            run_result='x86: '${model_name}'_7bit pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='x86: '${model_name}'_7bit failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
        echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'_9bit.ms --inDataFile='${models_path}'/input_output/input/'${model_name}'.ms.bin --benchmarkDataFile='${models_path}'/input_output/output/'${model_name}'.ms.out --accuracyThreshold=${accuracy_limit_9bit}' >> "${run_x86_log_file}"
        ./benchmark --modelFile=${ms_models_path}/${model_name}_9bit.ms --inDataFile=${models_path}/input_output/input/${model_name}.ms.bin --benchmarkDataFile=${models_path}/input_output/output/${model_name}.ms.out --accuracyThreshold=${accuracy_limit_9bit} >> "${run_x86_log_file}"
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
        data_path="${models_path}/input_output/"
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
        data_path="${models_path}/input_output/"
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

    # Run mindspore converted train models:
    fail=0
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
        export_file="${ms_train_models_path}/${model_name}_tod"
        inference_file="${ms_train_models_path}/${model_name}_infer"
        rm -f ${inference_file}"*"
        rm -f ${export_file}"*"
        echo ${model_name} >> "${run_x86_log_file}"
        ${run_valgrind}./tools/benchmark_train/benchmark_train \
        --modelFile=${ms_train_models_path}/${model_name}.ms \
        --inDataFile=${train_io_path}/${model_prefix}_input \
        --expectedDataFile=${train_io_path}/${model_prefix}_output --epochs=${epoch_num} --numThreads=${threads} \
        --accuracyThreshold=${accuracy_limit} --inferenceFile=${inference_file} \
        --exportFile=${export_file} >> "${run_x86_log_file}"
        if [ $? = 0 ]; then
            run_result='x86_train: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_train_result_file}
        else
            run_result='x86_train: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_train_result_file}
            fail=1
        fi
    done < ${models_ms_train_config}
    return ${fail}
}

# Run on x86 sse platform:
function Run_x86_sse() {
    cd ${x86_path}/sse || exit 1
    tar -zxf mindspore-lite-${version}-linux-x64.tar.gz || exit 1
    cd ${x86_path}/sse/mindspore-lite-${version}-linux-x64 || return 1
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./runtime/lib
    cp tools/benchmark/benchmark ./ || exit 1

    # Run tflite converted models:
    while read line; do
        model_name=${line}
        if [[ $model_name == \#* ]]; then
          continue
        fi
        echo ${model_name} >> "${run_x86_sse_log_file}"
        echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'.ms --inDataFile='${models_path}'/input_output/input/'${model_name}'.ms.bin --benchmarkDataFile='${models_path}'/input_output/output/'${model_name}'.ms.out' >> "${run_x86_sse_log_file}"
        ./benchmark --modelFile=${ms_models_path}/${model_name}.ms --inDataFile=${models_path}/input_output/input/${model_name}.ms.bin --benchmarkDataFile=${models_path}/input_output/output/${model_name}.ms.out >> "${run_x86_sse_log_file}"
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
        echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'.ms --inDataFile='${models_path}'/input_output/input/'${model_name}'.ms.bin --benchmarkDataFile='${models_path}'/input_output/output/'${model_name}'.ms.out' >> "${run_x86_sse_log_file}"
        ./benchmark --modelFile=${ms_models_path}/${model_name}.ms --inDataFile=${models_path}/input_output/input/${model_name}.ms.bin --benchmarkDataFile=${models_path}/input_output/output/${model_name}.ms.out >> "${run_x86_sse_log_file}"
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
        echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'.ms --inDataFile='${models_path}'/input_output/input/'${model_name}'.ms.bin --inputShapes='${input_shapes}' --benchmarkDataFile='${models_path}'/input_output/output/'${model_name}'.ms.out' >> "${run_x86_sse_log_file}"
        ./benchmark --modelFile=${ms_models_path}/${model_name}.ms --inDataFile=${models_path}/input_output/input/${model_name}.ms.bin --inputShapes=${input_shapes} --benchmarkDataFile=${models_path}/input_output/output/${model_name}.ms.out >> "${run_x86_sse_log_file}"
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
        echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'.ms --inDataFile='${models_path}'/input_output/input/'${model_name}'.ms.bin --benchmarkDataFile='${models_path}'/input_output/output/'${model_name}'.ms.out' >> "${run_x86_sse_log_file}"
        ./benchmark --modelFile=${ms_models_path}/${model_name}.ms --inDataFile=${models_path}/input_output/input/${model_name}.ms.bin --benchmarkDataFile=${models_path}/input_output/output/${model_name}.ms.out --accuracyThreshold=${accuracy_limit} >> "${run_x86_sse_log_file}"
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
        echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'_train.ms --inDataFile='${models_path}'/input_output/input/'${model_name}'.ms.bin --benchmarkDataFile=${models_path}/input_output/output/'${model_name}'.train.ms.out' >> "${run_x86_sse_log_file}"
        ./benchmark --modelFile=${ms_models_path}/${model_name}'_train'.ms --inDataFile=${models_path}/input_output/input/${model_name}.ms.bin --benchmarkDataFile=${models_path}/input_output/output/${model_name}.train.ms.out --accuracyThreshold=1.5 >> "${run_x86_sse_log_file}"
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
        transformer_data_path="${models_path}/input_output/input"
        echo ${model_name} >> "${run_x86_sse_log_file}"
        if [[ $model_name == "mobilenet.tflite" ]]; then
            echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'_posttraining.ms --inDataFile=/home/workspace/mindspore_dataset/mslite/quantTraining/mnist_calibration_data/00099.bin --benchmarkDataFile='${models_path}'/input_output/output/'${model_name}'_posttraining.ms.out' --accuracyThreshold=${accuracy_limit} >> "${run_x86_sse_log_file}"
            ./benchmark --modelFile=${ms_models_path}/${model_name}_posttraining.ms --inDataFile=/home/workspace/mindspore_dataset/mslite/quantTraining/mnist_calibration_data/00099.bin --benchmarkDataFile=${models_path}/input_output/output/${model_name}_posttraining.ms.out --accuracyThreshold=${accuracy_limit} >> "${run_x86_sse_log_file}"
        fi
        if [[ $model_name == "transformer_20200831_encoder_fp32.tflite" ]]; then
            echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'_posttraining.ms --inDataFile=${transformer_data_path}/encoder_buffer_in_0-35.bin --benchmarkDataFile='${models_path}'/input_output/output/'${model_name}'_posttraining.ms.out' --accuracyThreshold=${accuracy_limit} >> "${run_x86_sse_log_file}"
            ./benchmark --modelFile=${ms_models_path}/${model_name}_posttraining.ms --inDataFile=${transformer_data_path}/encoder_buffer_in_35.bin,${transformer_data_path}/encoder_buffer_in_0.bin,${transformer_data_path}/encoder_buffer_in_1.bin,${transformer_data_path}/encoder_buffer_in_4.bin,${transformer_data_path}/encoder_buffer_in_2.bin,${transformer_data_path}/encoder_buffer_in_3.bin,${transformer_data_path}/encoder_buffer_in_7.bin,${transformer_data_path}/encoder_buffer_in_5.bin,${transformer_data_path}/encoder_buffer_in_6.bin,${transformer_data_path}/encoder_buffer_in_10.bin,${transformer_data_path}/encoder_buffer_in_8.bin,${transformer_data_path}/encoder_buffer_in_9.bin,${transformer_data_path}/encoder_buffer_in_11.bin,${transformer_data_path}/encoder_buffer_in_12.bin,${transformer_data_path}/encoder_buffer_in_15.bin,${transformer_data_path}/encoder_buffer_in_13.bin,${transformer_data_path}/encoder_buffer_in_14.bin,${transformer_data_path}/encoder_buffer_in_18.bin,${transformer_data_path}/encoder_buffer_in_16.bin,${transformer_data_path}/encoder_buffer_in_17.bin,${transformer_data_path}/encoder_buffer_in_21.bin,${transformer_data_path}/encoder_buffer_in_19.bin,${transformer_data_path}/encoder_buffer_in_20.bin,${transformer_data_path}/encoder_buffer_in_22.bin,${transformer_data_path}/encoder_buffer_in_23.bin,${transformer_data_path}/encoder_buffer_in_26.bin,${transformer_data_path}/encoder_buffer_in_24.bin,${transformer_data_path}/encoder_buffer_in_25.bin,${transformer_data_path}/encoder_buffer_in_29.bin,${transformer_data_path}/encoder_buffer_in_27.bin,${transformer_data_path}/encoder_buffer_in_28.bin,${transformer_data_path}/encoder_buffer_in_32.bin,${transformer_data_path}/encoder_buffer_in_30.bin,${transformer_data_path}/encoder_buffer_in_31.bin,${transformer_data_path}/encoder_buffer_in_33.bin,${transformer_data_path}/encoder_buffer_in_34.bin --benchmarkDataFile=${models_path}/input_output/output/${model_name}_posttraining.ms.out --accuracyThreshold=${accuracy_limit} >> "${run_x86_sse_log_file}"
        fi
        if [[ $model_name == "transformer_20200831_decoder_fp32.tflite" ]]; then
            echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'_posttraining.ms --inDataFile=${transformer_data_path}/encoder_buffer_in_0-10.bin --benchmarkDataFile='${models_path}'/input_output/output/'${model_name}'_posttraining.ms.out' --accuracyThreshold=${accuracy_limit} >> "${run_x86_sse_log_file}"
            ./benchmark --modelFile=${ms_models_path}/${model_name}_posttraining.ms --inDataFile=${transformer_data_path}/decoder_buffer_in_9.bin,${transformer_data_path}/decoder_buffer_in_2.bin,${transformer_data_path}/decoder_buffer_in_0.bin,${transformer_data_path}/decoder_buffer_in_1.bin,${transformer_data_path}/decoder_buffer_in_5.bin,${transformer_data_path}/decoder_buffer_in_3.bin,${transformer_data_path}/decoder_buffer_in_4.bin,${transformer_data_path}/decoder_buffer_in_8.bin,${transformer_data_path}/decoder_buffer_in_6.bin,${transformer_data_path}/decoder_buffer_in_7.bin,${transformer_data_path}/decoder_buffer_in_10.bin --benchmarkDataFile=${models_path}/input_output/output/${model_name}_posttraining.ms.out --accuracyThreshold=${accuracy_limit} >> "${run_x86_sse_log_file}"
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
        echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'_posttraining.ms --inDataFile='${models_path}'/input_output/input/'${model_name}'_posttraining.ms.bin --benchmarkDataFile='${models_path}'/input_output/output/'${model_name}'_posttraining.ms.out'  --accuracyThreshold=${accuracy_limit} >> "${run_x86_sse_log_file}"
        ./benchmark --modelFile=${ms_models_path}/${model_name}_posttraining.ms --inDataFile=${models_path}/input_output/input/${model_name}_posttraining.ms.bin --benchmarkDataFile=${models_path}/input_output/output/${model_name}_posttraining.ms.out --accuracyThreshold=${accuracy_limit} >> "${run_x86_sse_log_file}"
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
        echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'.ms --inDataFile='${models_path}'/input_output/input/'${model_name}'.ms.bin --benchmarkDataFile='${models_path}'/input_output/output/'${model_name}'.ms.out' >> "${run_x86_sse_log_file}"
        ./benchmark --modelFile=${ms_models_path}/${model_name}.ms --inDataFile=${models_path}/input_output/input/${model_name}.ms.bin --benchmarkDataFile=${models_path}/input_output/output/${model_name}.ms.out >> "${run_x86_sse_log_file}"
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
        echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'.ms --inDataFile='${models_path}'/input_output/input/'${model_name}'.ms.bin --benchmarkDataFile='${models_path}'/input_output/output/'${model_name}'.ms.out --accuracyThreshold=${accuracy_limit}' >> "${run_x86_sse_log_file}"
        ./benchmark --modelFile=${ms_models_path}/${model_name}_weightquant.ms --inDataFile=${models_path}/input_output/input/${model_name}.ms.bin --benchmarkDataFile=${models_path}/input_output/output/${model_name}.ms.out --accuracyThreshold=${accuracy_limit} >> "${run_x86_sse_log_file}"
        if [ $? = 0 ]; then
            run_result='x86_sse: '${model_name}_weightquant' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='x86_sse: '${model_name}_weightquant' failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
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
        echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'.ms --inDataFile='${models_path}'/input_output/input/'${model_name}'.ms.bin --benchmarkDataFile='${models_path}'/input_output/output/'${model_name}'.ms.out --accuracyThreshold=${accuracy_limit}' >> "${run_x86_sse_log_file}"
        ./benchmark --modelFile=${ms_models_path}/${model_name}_weightquant.ms --inDataFile=${models_path}/input_output/input/${model_name}.ms.bin --benchmarkDataFile=${models_path}/input_output/output/${model_name}.ms.out --accuracyThreshold=${accuracy_limit} >> "${run_x86_sse_log_file}"
        if [ $? = 0 ]; then
            run_result='x86_sse: '${model_name}'_weightquant pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='x86_sse: '${model_name}'_weightquant failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
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
        echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'_7bit.ms --inDataFile='${models_path}'/input_output/input/'${model_name}'.ms.bin --benchmarkDataFile='${models_path}'/input_output/output/'${model_name}'.ms.out --accuracyThreshold=${accuracy_limit_7bit}' >> "${run_x86_sse_log_file}"
        ./benchmark --modelFile=${ms_models_path}/${model_name}_7bit.ms --inDataFile=${models_path}/input_output/input/${model_name}.ms.bin --benchmarkDataFile=${models_path}/input_output/output/${model_name}.ms.out --accuracyThreshold=${accuracy_limit_7bit} >> "${run_x86_sse_log_file}"
        if [ $? = 0 ]; then
            run_result='x86_sse: '${model_name}'_7bit pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='x86_sse: '${model_name}'_7bit failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
        echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'_9bit.ms --inDataFile='${models_path}'/input_output/input/'${model_name}'.ms.bin --benchmarkDataFile='${models_path}'/input_output/output/'${model_name}'.ms.out --accuracyThreshold=${accuracy_limit_9bit}' >> "${run_x86_sse_log_file}"
        ./benchmark --modelFile=${ms_models_path}/${model_name}_9bit.ms --inDataFile=${models_path}/input_output/input/${model_name}.ms.bin --benchmarkDataFile=${models_path}/input_output/output/${model_name}.ms.out --accuracyThreshold=${accuracy_limit_9bit} >> "${run_x86_sse_log_file}"
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
        data_path="${models_path}/input_output/"
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
    tar -zxf mindspore-lite-${version}-linux-x64.tar.gz || exit 1
    cd ${x86_path}/avx/mindspore-lite-${version}-linux-x64 || return 1
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./runtime/lib
    cp tools/benchmark/benchmark ./ || exit 1

    # Run tflite converted models:
    while read line; do
        model_name=${line}
        if [[ $model_name == \#* ]]; then
          continue
        fi
        echo ${model_name} >> "${run_x86_avx_log_file}"
        echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'.ms --inDataFile='${models_path}'/input_output/input/'${model_name}'.ms.bin --benchmarkDataFile='${models_path}'/input_output/output/'${model_name}'.ms.out' >> "${run_x86_avx_log_file}"
        ./benchmark --modelFile=${ms_models_path}/${model_name}.ms --inDataFile=${models_path}/input_output/input/${model_name}.ms.bin --benchmarkDataFile=${models_path}/input_output/output/${model_name}.ms.out >> "${run_x86_avx_log_file}"
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
        echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'.ms --inDataFile='${models_path}'/input_output/input/'${model_name}'.ms.bin --benchmarkDataFile='${models_path}'/input_output/output/'${model_name}'.ms.out' >> "${run_x86_avx_log_file}"
        ./benchmark --modelFile=${ms_models_path}/${model_name}.ms --inDataFile=${models_path}/input_output/input/${model_name}.ms.bin --benchmarkDataFile=${models_path}/input_output/output/${model_name}.ms.out >> "${run_x86_avx_log_file}"
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
        echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'.ms --inDataFile='${models_path}'/input_output/input/'${model_name}'.ms.bin --inputShapes='${input_shapes}' --benchmarkDataFile='${models_path}'/input_output/output/'${model_name}'.ms.out' >> "${run_x86_avx_log_file}"
        ./benchmark --modelFile=${ms_models_path}/${model_name}.ms --inDataFile=${models_path}/input_output/input/${model_name}.ms.bin --inputShapes=${input_shapes} --benchmarkDataFile=${models_path}/input_output/output/${model_name}.ms.out >> "${run_x86_avx_log_file}"
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
        echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'.ms --inDataFile='${models_path}'/input_output/input/'${model_name}'.ms.bin --benchmarkDataFile='${models_path}'/input_output/output/'${model_name}'.ms.out' >> "${run_x86_avx_log_file}"
        ./benchmark --modelFile=${ms_models_path}/${model_name}.ms --inDataFile=${models_path}/input_output/input/${model_name}.ms.bin --benchmarkDataFile=${models_path}/input_output/output/${model_name}.ms.out --accuracyThreshold=${accuracy_limit} >> "${run_x86_avx_log_file}"
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
        echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'_train.ms --inDataFile='${models_path}'/input_output/input/'${model_name}'.ms.bin --benchmarkDataFile=${models_path}/input_output/output/'${model_name}'.train.ms.out' >> "${run_x86_avx_log_file}"
        ./benchmark --modelFile=${ms_models_path}/${model_name}'_train'.ms --inDataFile=${models_path}/input_output/input/${model_name}.ms.bin --benchmarkDataFile=${models_path}/input_output/output/${model_name}.train.ms.out --accuracyThreshold=1.5 >> "${run_x86_avx_log_file}"
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
        transformer_data_path="${models_path}/input_output/input"
        echo ${model_name} >> "${run_x86_avx_log_file}"
        if [[ $model_name == "mobilenet.tflite" ]]; then
            echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'_posttraining.ms --inDataFile=/home/workspace/mindspore_dataset/mslite/quantTraining/mnist_calibration_data/00099.bin --benchmarkDataFile='${models_path}'/input_output/output/'${model_name}'_posttraining.ms.out' --accuracyThreshold=${accuracy_limit} >> "${run_x86_avx_log_file}"
            ./benchmark --modelFile=${ms_models_path}/${model_name}_posttraining.ms --inDataFile=/home/workspace/mindspore_dataset/mslite/quantTraining/mnist_calibration_data/00099.bin --benchmarkDataFile=${models_path}/input_output/output/${model_name}_posttraining.ms.out --accuracyThreshold=${accuracy_limit} >> "${run_x86_avx_log_file}"
        fi
        if [[ $model_name == "transformer_20200831_encoder_fp32.tflite" ]]; then
            echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'_posttraining.ms --inDataFile=${transformer_data_path}/encoder_buffer_in_0-35.bin --benchmarkDataFile='${models_path}'/input_output/output/'${model_name}'_posttraining.ms.out' --accuracyThreshold=${accuracy_limit} >> "${run_x86_avx_log_file}"
            ./benchmark --modelFile=${ms_models_path}/${model_name}_posttraining.ms --inDataFile=${transformer_data_path}/encoder_buffer_in_35.bin,${transformer_data_path}/encoder_buffer_in_0.bin,${transformer_data_path}/encoder_buffer_in_1.bin,${transformer_data_path}/encoder_buffer_in_4.bin,${transformer_data_path}/encoder_buffer_in_2.bin,${transformer_data_path}/encoder_buffer_in_3.bin,${transformer_data_path}/encoder_buffer_in_7.bin,${transformer_data_path}/encoder_buffer_in_5.bin,${transformer_data_path}/encoder_buffer_in_6.bin,${transformer_data_path}/encoder_buffer_in_10.bin,${transformer_data_path}/encoder_buffer_in_8.bin,${transformer_data_path}/encoder_buffer_in_9.bin,${transformer_data_path}/encoder_buffer_in_11.bin,${transformer_data_path}/encoder_buffer_in_12.bin,${transformer_data_path}/encoder_buffer_in_15.bin,${transformer_data_path}/encoder_buffer_in_13.bin,${transformer_data_path}/encoder_buffer_in_14.bin,${transformer_data_path}/encoder_buffer_in_18.bin,${transformer_data_path}/encoder_buffer_in_16.bin,${transformer_data_path}/encoder_buffer_in_17.bin,${transformer_data_path}/encoder_buffer_in_21.bin,${transformer_data_path}/encoder_buffer_in_19.bin,${transformer_data_path}/encoder_buffer_in_20.bin,${transformer_data_path}/encoder_buffer_in_22.bin,${transformer_data_path}/encoder_buffer_in_23.bin,${transformer_data_path}/encoder_buffer_in_26.bin,${transformer_data_path}/encoder_buffer_in_24.bin,${transformer_data_path}/encoder_buffer_in_25.bin,${transformer_data_path}/encoder_buffer_in_29.bin,${transformer_data_path}/encoder_buffer_in_27.bin,${transformer_data_path}/encoder_buffer_in_28.bin,${transformer_data_path}/encoder_buffer_in_32.bin,${transformer_data_path}/encoder_buffer_in_30.bin,${transformer_data_path}/encoder_buffer_in_31.bin,${transformer_data_path}/encoder_buffer_in_33.bin,${transformer_data_path}/encoder_buffer_in_34.bin --benchmarkDataFile=${models_path}/input_output/output/${model_name}_posttraining.ms.out --accuracyThreshold=${accuracy_limit} >> "${run_x86_avx_log_file}"
        fi
        if [[ $model_name == "transformer_20200831_decoder_fp32.tflite" ]]; then
            echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'_posttraining.ms --inDataFile=${transformer_data_path}/encoder_buffer_in_0-10.bin --benchmarkDataFile='${models_path}'/input_output/output/'${model_name}'_posttraining.ms.out' --accuracyThreshold=${accuracy_limit} >> "${run_x86_avx_log_file}"
            ./benchmark --modelFile=${ms_models_path}/${model_name}_posttraining.ms --inDataFile=${transformer_data_path}/decoder_buffer_in_9.bin,${transformer_data_path}/decoder_buffer_in_2.bin,${transformer_data_path}/decoder_buffer_in_0.bin,${transformer_data_path}/decoder_buffer_in_1.bin,${transformer_data_path}/decoder_buffer_in_5.bin,${transformer_data_path}/decoder_buffer_in_3.bin,${transformer_data_path}/decoder_buffer_in_4.bin,${transformer_data_path}/decoder_buffer_in_8.bin,${transformer_data_path}/decoder_buffer_in_6.bin,${transformer_data_path}/decoder_buffer_in_7.bin,${transformer_data_path}/decoder_buffer_in_10.bin --benchmarkDataFile=${models_path}/input_output/output/${model_name}_posttraining.ms.out --accuracyThreshold=${accuracy_limit} >> "${run_x86_avx_log_file}"
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
        echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'_posttraining.ms --inDataFile='${models_path}'/input_output/input/'${model_name}'_posttraining.ms.bin --benchmarkDataFile='${models_path}'/input_output/output/'${model_name}'_posttraining.ms.out'  --accuracyThreshold=${accuracy_limit} >> "${run_x86_avx_log_file}"
        ./benchmark --modelFile=${ms_models_path}/${model_name}_posttraining.ms --inDataFile=${models_path}/input_output/input/${model_name}_posttraining.ms.bin --benchmarkDataFile=${models_path}/input_output/output/${model_name}_posttraining.ms.out --accuracyThreshold=${accuracy_limit} >> "${run_x86_avx_log_file}"
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
        echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'.ms --inDataFile='${models_path}'/input_output/input/'${model_name}'.ms.bin --benchmarkDataFile='${models_path}'/input_output/output/'${model_name}'.ms.out' >> "${run_x86_avx_log_file}"
        ./benchmark --modelFile=${ms_models_path}/${model_name}.ms --inDataFile=${models_path}/input_output/input/${model_name}.ms.bin --benchmarkDataFile=${models_path}/input_output/output/${model_name}.ms.out >> "${run_x86_avx_log_file}"
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
        echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'.ms --inDataFile='${models_path}'/input_output/input/'${model_name}'.ms.bin --benchmarkDataFile='${models_path}'/input_output/output/'${model_name}'.ms.out --accuracyThreshold=${accuracy_limit}' >> "${run_x86_avx_log_file}"
        ./benchmark --modelFile=${ms_models_path}/${model_name}_weightquant.ms --inDataFile=${models_path}/input_output/input/${model_name}.ms.bin --benchmarkDataFile=${models_path}/input_output/output/${model_name}.ms.out --accuracyThreshold=${accuracy_limit} >> "${run_x86_avx_log_file}"
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
        echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'.ms --inDataFile='${models_path}'/input_output/input/'${model_name}'.ms.bin --benchmarkDataFile='${models_path}'/input_output/output/'${model_name}'.ms.out --accuracyThreshold=${accuracy_limit}' >> "${run_x86_avx_log_file}"
        ./benchmark --modelFile=${ms_models_path}/${model_name}_weightquant.ms --inDataFile=${models_path}/input_output/input/${model_name}.ms.bin --benchmarkDataFile=${models_path}/input_output/output/${model_name}.ms.out --accuracyThreshold=${accuracy_limit} >> "${run_x86_avx_log_file}"
        if [ $? = 0 ]; then
            run_result='x86_avx: '${model_name}'_weightquant pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='x86_avx: '${model_name}'_weightquant failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
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
        echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'_7bit.ms --inDataFile='${models_path}'/input_output/input/'${model_name}'.ms.bin --benchmarkDataFile='${models_path}'/input_output/output/'${model_name}'.ms.out --accuracyThreshold=${accuracy_limit_7bit}' >> "${run_x86_avx_log_file}"
        ./benchmark --modelFile=${ms_models_path}/${model_name}_7bit.ms --inDataFile=${models_path}/input_output/input/${model_name}.ms.bin --benchmarkDataFile=${models_path}/input_output/output/${model_name}.ms.out --accuracyThreshold=${accuracy_limit_7bit} >> "${run_x86_avx_log_file}"
        if [ $? = 0 ]; then
            run_result='x86_avx: '${model_name}'_7bit pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='x86_avx: '${model_name}'_7bit failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
        echo './benchmark --modelFile='${ms_models_path}'/'${model_name}'_9bit.ms --inDataFile='${models_path}'/input_output/input/'${model_name}'.ms.bin --benchmarkDataFile='${models_path}'/input_output/output/'${model_name}'.ms.out --accuracyThreshold=${accuracy_limit_9bit}' >> "${run_x86_avx_log_file}"
        ./benchmark --modelFile=${ms_models_path}/${model_name}_9bit.ms --inDataFile=${models_path}/input_output/input/${model_name}.ms.bin --benchmarkDataFile=${models_path}/input_output/output/${model_name}.ms.out --accuracyThreshold=${accuracy_limit_9bit} >> "${run_x86_avx_log_file}"
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
        data_path="${models_path}/input_output/"
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

# Run on x86 java platform:
function Run_x86_java() {
    cd ${x86_java_path} || exit 1
    tar -zxf mindspore-lite-${version}-linux-x64.tar.gz || exit 1
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${x86_java_path}/mindspore-lite-${version}-linux-x64/runtime/lib
    # compile benchmark
    echo "javac -cp ${x86_java_path}/mindspore-lite-${version}-linux-x64/runtime/lib/mindspore-lite-java.jar ${basepath}/java/src/main/java/Benchmark.java -d ."
    javac -cp ${x86_java_path}/mindspore-lite-${version}-linux-x64/runtime/lib/mindspore-lite-java.jar ${basepath}/java/src/main/java/Benchmark.java -d .

    count=0
    # Run tflite converted models:
    while read line; do
        # only run top5.
        count=`expr ${count}+1`
        if [[ ${count} -gt 5 ]]; then
            break
        fi
        model_name=${line}
        if [[ $model_name == \#* ]]; then
          continue
        fi
        echo ${model_name} >> "${run_x86_java_log_file}"
        echo "java -classpath .:${x86_java_path}/mindspore-lite-${version}-linux-x64/runtime/lib/mindspore-lite-java.jar Benchmark ${ms_models_path}/${model_name}.ms '${models_path}'/input_output/input/${model_name}.ms.bin '${models_path}'/input_output/output/${model_name}.ms.out 1" >> "${run_x86_java_log_file}"
        java -classpath .:${x86_java_path}/mindspore-lite-${version}-linux-x64/runtime/lib/mindspore-lite-java.jar Benchmark ${ms_models_path}/${model_name}.ms ${models_path}/input_output/input/${model_name}.ms.bin ${models_path}/input_output/output/${model_name}.ms.out 1
        if [ $? = 0 ]; then
            run_result='x86_java: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='x86_java: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
    done < ${models_tflite_config}
}

# Run on x86 codegen benchmark
function Run_x86_codegen() {
    local CODEGEN_PATH=${x86_path}/mindspore-lite-${version}-linux-x64/tools/codegen

    rm -rf ${build_path}
    mkdir -p ${build_path}

    while read line; do
        model_name=${line}
        if [[ $model_name == \#* ]]; then
          continue
        fi
        echo ${model_name} >> "${run_x86_codegen_log_file}"
        ${CODEGEN_PATH}/codegen --codePath=${build_path} --modelPath=${ms_models_path}/${model_name}.ms >> ${run_x86_codegen_log_file}
        # 1. build benchmark
        mkdir -p ${build_path}/${model_name}/build && cd ${build_path}/${model_name}/build || exit 1
        cmake -DPKG_PATH=${x86_path}/mindspore-lite-${version}-linux-x64 ${build_path}/${model_name} >> ${run_x86_codegen_log_file}
        make >> ${run_x86_codegen_log_file}
        # 2. run benchmark
        echo "net file: ${build_path}/${model_name}/src/net.bin" >> ${run_x86_codegen_log_file}
        echo "./benchmark ${models_path}/input_output/input/${model_name}.ms.bin ${build_path}/${model_name}/src/net.bin 1 ${models_path}/input_output/output/${model_name}.ms.out" >> ${run_x86_codegen_log_file}
        ./benchmark ${models_path}/input_output/input/${model_name}.ms.bin ${build_path}/${model_name}/src/net.bin 1 ${models_path}/input_output/output/${model_name}.ms.out >> ${run_x86_codegen_log_file}
        if [ $? = 0 ]; then
            run_result='x86_codegen: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='x86_codegen: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
    done < ${models_codegen_config}

    rm -rf ${build_path}
}

# Run on x86 codegen benchmark parallel
function Run_x86_codegen_parallel() {
    local CODEGEN_PATH=${x86_path}/mindspore-lite-${version}-linux-x64/tools/codegen

    rm -rf ${build_parallal_path}
    mkdir -p ${build_parallal_path}

    while read line; do
        model_name=${line}
        if [[ $model_name == \#* ]]; then
          continue
        fi
        echo ${model_name} >> "${run_x86_codegen_parallel_log_file}"
        ${CODEGEN_PATH}/codegen --codePath=${build_parallal_path} --modelPath=${ms_models_path}/${model_name}.ms --supportParallel=true >> ${run_x86_codegen_parallel_log_file}
        # 1. build benchmark
        mkdir -p ${build_parallal_path}/${model_name}/build && cd ${build_parallal_path}/${model_name}/build || exit 1
        cmake -DPKG_PATH=${x86_path}/mindspore-lite-${version}-linux-x64 ${build_parallal_path}/${model_name} >> ${run_x86_codegen_parallel_log_file}
        make >> ${run_x86_codegen_parallel_log_file}
        # 2. run benchmark
        echo "net file: ${build_parallal_path}/${model_name}/src/net.bin" >> ${run_x86_codegen_parallel_log_file}
        echo "./benchmark ${models_path}/input_output/input/${model_name}.ms.bin ${build_parallal_path}/${model_name}/src/net.bin 1 ${models_path}/input_output/output/${model_name}.ms.out thread_num=4 0" >> ${run_x86_codegen_parallel_log_file}
        ./benchmark ${models_path}/input_output/input/${model_name}.ms.bin ${build_parallal_path}/${model_name}/src/net.bin 1 ${models_path}/input_output/output/${model_name}.ms.out 4 0 >> ${run_x86_codegen_parallel_log_file}
        if [ $? = 0 ]; then
            run_result='x86_codegen_parallel: '${model_name}' pass'; echo ${run_result} >> ${run_benchmark_result_file}
        else
            run_result='x86_codegen_parallel: '${model_name}' failed'; echo ${run_result} >> ${run_benchmark_result_file}; return 1
        fi
    done < ${models_codegen_parallel_config}

    rm -rf ${build_parallal_path}
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

# Write benchmark_train result to temp file
run_benchmark_train_result_file=${logs_path}/run_benchmark_train_result.txt
echo ' ' > ${run_benchmark_train_result_file}

epoch_num=1
threads=2
train_models_path=""
train_io_path=""
# Example:sh run_benchmark_x86.sh -r /home/temp_test -m /home/temp_test/models -e arm_cpu
while getopts "r:m:M:e:i:v:p:t:" opt; do
    case ${opt} in
        r)
            release_path=${OPTARG}
            echo "release_path is ${OPTARG}"
            ;;
        m)
            models_path=${OPTARG}
            echo "models_path is ${OPTARG}"
            ;;
        M)
            train_models_path=${OPTARG}
            echo "train_models_path is ${models_path}"
            ;;
        e)
            backend=${OPTARG}
            echo "backend is ${OPTARG}"
            ;;
        i)
            train_io_path=${OPTARG}
            echo "train_io_path is ${OPTARG}"
            ;;
        v)
            run_valgrind="valgrind --log-file=valgrind.log "
            echo "Run x86 with valgrind"
            ;;
        t)
            epoch_num=${OPTARG}
            echo "train epoch num is ${epoch_num}"
            ;;
        p)
            enable_export=1
            docker_image=${OPTARG}
            echo "enable_export = 1, docker_image = ${OPTARG}"
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
models_mindspore_config=${basepath}/../config/models_mindspore.cfg
models_mindspore_train_config=${basepath}/../config/models_mindspore_train.cfg
models_mindspore_mixbit_config=${basepath}/../config/models_mindspore_mixbit.cfg
models_mindspore_weightquant_config=${basepath}/../config/models_mindspore_weightquant.cfg
models_with_multiple_inputs_config=${basepath}/../config/models_with_multiple_inputs.cfg
models_for_process_only_config=${basepath}/../config/models_for_process_only.cfg
models_tf_weightquant_config=${basepath}/../config/models_tf_weightquant.cfg
models_codegen_config=${basepath}/../codegen/models_codegen.cfg
models_codegen_parallel_config=${basepath}/../codegen/models_codegen_parallel.cfg
models_ms_train_config=${basepath}/../config/models_ms_train.cfg

ms_models_path=${basepath}/ms_models
ms_train_models_path=${basepath}/ms_train_models
rm -rf ${ms_train_models_path}
mkdir -p ${ms_train_models_path}
build_path=${basepath}/codegen_build
build_parallal_path=${basepath}/codegen_parallel_build
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

run_x86_log_file=${basepath}/run_x86_log.txt
echo 'run x86 logs: ' > ${run_x86_log_file}

run_x86_sse_log_file=${basepath}/run_x86_sse_log.txt
echo 'run x86 sse logs: ' > ${run_x86_sse_log_file}

run_x86_avx_log_file=${basepath}/run_x86_avx_log.txt
echo 'run x86 avx logs: ' > ${run_x86_avx_log_file}

run_x86_java_log_file=${basepath}/run_x86_java_log.txt
echo 'run x86 java logs: ' > ${run_x86_java_log_file}

run_x86_codegen_log_file=${basepath}/run_x86_codegen_log.txt
echo 'run x86 codegen logs: ' > ${run_x86_codegen_log_file}

run_x86_codegen_parallel_log_file=${basepath}/run_x86_codegen_parallel_log.txt
echo 'run x86 codegen parallel logs: ' > ${run_x86_codegen_parallel_log_file}

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

if [[ $backend == "all" || $backend == "x86-all" || $backend == "x86-java" ]]; then
    # Run on x86-java
    echo "start Run x86 java ..."
    x86_java_path=${release_path}/ubuntu_x86/avx
    Run_x86_java &
    Run_x86_java_PID=$!
    sleep 1
fi

if [[ $backend == "all" || $backend == "x86-all" || $backend == "x86-codegen" ]]; then
    # Run on x86-codegen
    echo "start Run x86 codegen ..."
    Run_x86_codegen &
    Run_x86_codegen_PID=$!
    sleep 1
fi

if [[ $backend == "all" || $backend == "x86-all" || $backend == "x86-codegen-parallel" ]]; then
    # Run on x86-codegen-parallel
    echo "start Run x86 codegen parallel ..."
    Run_x86_codegen_parallel &
    Run_x86_codegen_parallel_PID=$!
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
if [[ $backend == "all" || $backend == "x86-all" || $backend == "x86-java" ]]; then
    wait ${Run_x86_java_PID}
    Run_x86_java_status=$?

    if [[ ${Run_x86_java_status} != 0 ]];then
        echo "Run_x86 java failed"
        cat ${run_x86_java_log_file}
        isFailed=1
    fi
fi

if [[ $backend == "all" || $backend == "x86-all" || $backend == "x86-codegen" ]]; then
    wait ${Run_x86_codegen_PID}
    Run_x86_codegen_status=$?

    if [[ ${Run_x86_codegen_status} != 0 ]];then
        echo "Run_x86 codegen failed"
        cat ${run_x86_codegen_log_file}
        isFailed=1
    fi
fi

if [[ $backend == "all" || $backend == "x86-all" || $backend == "x86-codegen-parallel" ]]; then
    wait ${Run_x86_codegen_parallel_PID}
    Run_x86_codegen_parallel_status=$?

    if [[ ${Run_x86_codegen_parallel_status} != 0 ]];then
        echo "Run_x86 codegen parallel failed"
        cat ${run_x86_codegen_log_file}
        isFailed=1
    fi
fi

echo "Run_x86 and Run_x86_sse and Run_x86_avx and is ended"
Print_Benchmark_Result
if [[ $isFailed == 1 ]]; then
    exit 1
fi
echo "Run x86 train end"
Print_Benchmark_Train_Result
if [[ $isFailed == 1 ]]; then
    exit 1
fi
exit 0
