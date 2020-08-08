#!/bin/bash
basepath=$(pwd)
echo $basepath
set -e
#example：sh run_benchmark_nets.sh -a /home/temp_test -c /home/temp_test -m /home/temp_test/models -d "8KE5T19620002408"
while getopts "a:c:m:d:" opt
do
    case $opt in
        a)
		arm_path=$OPTARG
        echo "arm_path is $OPTARG"
        ;;
        c)
		convertor_path=$OPTARG
        echo "convertor_path is $OPTARG"
        ;;
        m)
		models_path=$OPTARG
        echo "models_path is $OPTARG"
        ;;		
        d)
		device_id=$OPTARG
        echo "device_id is $OPTARG"
        ;;
        ?)
        echo "unknown para"
        exit 1;;
    esac
done



#unzip arm 
cd $arm_path
tar -zxf MSLite-*-linux_arm64.tar.gz


#unzip convertor 
cd $convertor_path
tar -zxf MSLite-*-linux_x86_64.tar.gz
cd $convertor_path/MSLite-*-linux_x86_64
cp converter/converter_lite ./
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./lib/:./third_party/protobuf/lib

#the original model's path: $models_path/


#convert the models
cd $convertor_path/MSLite-*-linux_x86_64

#models_config_filename=/home/workspace/mindspore_dataset/mslite/models/models_config.txt
models_tflite_config=${basepath}/models_tflite.cfg
models_caffe_config=${basepath}/models_caffe.cfg

rm -rf ${basepath}/ms_models
mkdir -p ${basepath}/ms_models
ms_models_path=${basepath}/ms_models

#convert tflite models:
while read line;do
	model_name=$line
	echo $model_name
	echo './converter_lite  --fmk=TFLITE --modelFile='${models_path}'/'${model_name}' --outputFile='${ms_models_path}'/'${model_name}''
	./converter_lite  --fmk=TFLITE --modelFile=$models_path/${model_name} --outputFile=${ms_models_path}/${model_name}
done < ${models_tflite_config}

#convert caffe models:
while read line;do
        model_name=$line
        echo $model_name
	pwd
        echo './converter_lite  --fmk=CAFFE --modelFile='${models_path}'/'${model_name}'.prototxt --weightFile='${models_path}'/'${model_name}'.caffemodel --outputFile='${ms_models_path}'/'${model_name}''
	./converter_lite  --fmk=CAFFE --modelFile=${models_path}/${model_name}.prototxt --weightFile=${models_path}/${model_name}.caffemodel --outputFile=${ms_models_path}/${model_name}
done < ${models_caffe_config}


#push to the arm and run benchmark：

#=====first：copy benchmark exe and so files to the server which connected to the phone
rm -rf ${basepath}/benchmark_test
mkdir -p ${basepath}/benchmark_test
benchmark_test_path=${basepath}/benchmark_test
cd ${benchmark_test_path}
cp  $arm_path/MSLite-*-linux_arm64/lib/libmindspore-lite.so ${benchmark_test_path}/libmindspore-lite.so
cp  $arm_path/MSLite-*-linux_arm64/benchmark/benchmark ${benchmark_test_path}/benchmark

#copy the MindSpore models：
cp  ${ms_models_path}/*.ms ${benchmark_test_path}

#=====second：adb push all needed files to the phone
adb -s $device_id push ${benchmark_test_path} /data/local/tmp/

#=====third：run adb ,run session ,check the result:
echo 'cd  /data/local/tmp/benchmark_test' > adb_cmd.txt
echo 'cp  /data/local/tmp/libc++_shared.so ./' >> adb_cmd.txt
echo 'chmod 777 benchmark' >> adb_cmd.txt

adb -s $device_id shell < adb_cmd.txt

#run tflite converted models：
while read line;do
	model_name=$line
	echo $model_name
	echo 'cd  /data/local/tmp/benchmark_test' > adb_run_cmd.txt
	echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --modelPath='${model_name}'.ms --inDataPath=/data/local/tmp/input_output/input/'${model_name}'.ms.bin --calibDataPath=/data/local/tmp/input_output/output/'${model_name}'.ms.out --warmUpLoopCount=1 --loopCount=1'
	echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --modelPath='${model_name}'.ms --inDataPath=/data/local/tmp/input_output/input/'${model_name}'.ms.bin --calibDataPath=/data/local/tmp/input_output/output/'${model_name}'.ms.out --warmUpLoopCount=1 --loopCount=1' >> adb_run_cmd.txt
        adb -s $device_id shell < adb_run_cmd.txt
done < ${models_tflite_config}

#run caffe converted models:
while read line;do
        model_name=$line
        echo $model_name
        echo 'cd  /data/local/tmp/benchmark_test' > adb_run_cmd.txt
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --modelPath='${model_name}'.ms --inDataPath=/data/local/tmp/input_output/input/'${model_name}'.ms.bin --calibDataPath=/data/local/tmp/input_output/output/'${model_name}'.ms.out --warmUpLoopCount=1 --loopCount=1'
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --modelPath='${model_name}'.ms --inDataPath=/data/local/tmp/input_output/input/'${model_name}'.ms.bin --calibDataPath=/data/local/tmp/input_output/output/'${model_name}'.ms.out --warmUpLoopCount=1 --loopCount=1' >> adb_run_cmd.txt
        adb -s $device_id shell < adb_run_cmd.txt
done < ${models_caffe_config}
