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
#model1：
./converter_lite  --fmk=CAFFE --modelFile=$models_path/test.prototxt --outputFile=$models_path/test --weightFile=$models_path/test.caffemodel 
#model2：
./converter_lite  --fmk=TFLITE --modelFile=$models_path/hiai_bigmodel_ghost_2_1_no_normalized_no_trans_tflite.tflite --outputFile=$models_path/hiai_bigmodel_ghost_2_1_no_normalized_no_trans_tflite 

./converter_lite  --fmk=TFLITE --modelFile=$models_path/hiai_cn_recognize_modify_padv2.tflite --outputFile=$models_path/hiai_cn_recognize_modify_padv2

./converter_lite  --fmk=TFLITE --modelFile=$models_path/hiai_detect_curve_model_float32.tflite --outputFile=$models_path/hiai_detect_curve_model_float32

./converter_lite  --fmk=TFLITE --modelFile=$models_path/hiai_detectmodel_desnet_256_128_64_32.tflite --outputFile=$models_path/hiai_detectmodel_desnet_256_128_64_32

./converter_lite  --fmk=TFLITE --modelFile=$models_path/mobilenet_v2_1_0_224.tflite --outputFile=$models_path/mobilenet_v2_1_0_224

#push to the arm and run benchmark：

#first：copy to the server which connected to the phone
mkdir -p ./benchmark_test
cp  $arm_path/MSLite-0.6.0-linux_arm64/lib/libmindspore-lite.so ./benchmark_test/libmindspore-lite.so
cp  $arm_path/MSLite-0.6.0-linux_arm64/benchmark/benchmark ./benchmark_test/benchmark

#copy the models：
cp  $models_path/*.ms ./benchmark_test/
#model1：
#cp  $models_path/test.ms ./benchmark_test/
#model2：
#cp  $models_path/hiai_bigmodel_ghost_2_1_no_normalized_no_trans_tflite.ms ./benchmark_test/
#cp  $models_path/mobilenet_v2_1.0_224.tflite.ms ./benchmark_test/

#second：adb push to the phone
adb -s $device_id push ./benchmark_test /data/local/tmp/

#third：run adb ,run session ,check the result:
echo 'cd  /data/local/tmp/' > adb_cmd.txt
echo 'chmod 777 benchmark' >> adb_cmd.txt
#model1：
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/;./benchmark --modelPath=test.ms' >> adb_cmd.txt
#model2：
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --modelPath=hiai_bigmodel_ghost_2_1_no_normalized_no_trans_tflite.ms' >> adb_cmd.txt

echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --modelPath=hiai_cn_recognize_modify_padv2.ms' >> adb_cmd.txt

echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --modelPath=hiai_detect_curve_model_float32.ms' >> adb_cmd.txt

echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --modelPath=hiai_detectmodel_desnet_256_128_64_32.ms' >> adb_cmd.txt


echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --modelPath=mobilenet_v2_1_0_224.ms' >> adb_cmd.txt

adb -s $device_id shell < adb_cmd.txt

