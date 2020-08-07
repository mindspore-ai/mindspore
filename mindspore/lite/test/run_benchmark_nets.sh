#!/bin/bash
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

models_config_filename=/home/workspace/mindspore_dataset/mslite/models/models_config.txt

while read line;do
	model_name=$line
	./converter_lite  --fmk=TFLITE --modelFile=$models_path/${model_name} --outputFile=$models_path/${model_name}
done < ${models_config_filename}

#push to the arm and run benchmark：

#first：copy to the server which connected to the phone
mkdir -p ./benchmark_test
cp  $arm_path/MSLite-0.6.0-linux_arm64/lib/libmindspore-lite.so ./benchmark_test/libmindspore-lite.so
cp  $arm_path/MSLite-0.6.0-linux_arm64/benchmark/benchmark ./benchmark_test/benchmark

#copy the models：
cp  $models_path/*.ms ./benchmark_test/

#second：adb push to the phone
adb -s $device_id push ./benchmark_test /data/local/tmp/

#third：run adb ,run session ,check the result:
echo 'cd  /data/local/tmp/benchmark_test' > adb_cmd.txt
echo 'cp  /data/local/tmp/libc++_shared.so ./' >> adb_cmd.txt
echo 'chmod 777 benchmark' >> adb_cmd.txt
#run models：
while read line;do
	model_name=$line
	echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --modelPath='${model_name}'.ms --inDataPath=/data/local/tmp/input_output/input/'${model_name}'.ms.bin --calibDataPath=/data/local/tmp/input_output/output/'${model_name}'.ms.out --warmUpLoopCount=1 --loopCount=1' >> adb_cmd.txt
done < ${models_config_filename}

adb -s $device_id shell < adb_cmd.txt

