#!/bin/bash

#获取相关输入参数
#举例：sh run_benchmark_nets.sh -a /home/temp_test -c /home/temp_test -m /home/temp_test/models -d "8KE5T19620002408"
while getopts "a:c:m:d:" opt
do
    case $opt in
        a)
                arm_path=$OPTARG
        echo "参数arm_path的值$OPTARG"
        ;;
        c)
                convertor_path=$OPTARG
        echo "参数convertor_path的值$OPTARG"
        ;;
        m)
                models_path=$OPTARG
        echo "参数models_path的值$OPTARG"
        ;;
        d)
                device_id=$OPTARG
        echo "参数device_id的值$OPTARG"
        ;;
        ?)
        echo "未知参数"
        exit 1;;
    esac
done

#将编译好的arm包先放在如下目录进行调试
cd $arm_path
tar -zxf MSLite-*-linux_arm64.tar.gz


#部署模型转换工具
cd $convertor_path
tar -zxf MSLite-*-linux_x86_64.tar.gz
cd $convertor_path/MSLite-*-linux_x86_64
cp converter/converter_lite ./
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./lib/:./third_party/protobuf/lib

#获取原始模型提前放置在$models_path/


#进行模型转换
cd $convertor_path/MSLite-*-linux_x86_64
#模型1：
./converter_lite  --fmk=CAFFE --modelFile=$models_path/test.prototxt --outputFile=$models_path/test --weightFile=$models_path/test.caffemodel
#模型2：
./converter_lite  --fmk=TFLITE --modelFile=$models_path/hiai_bigmodel_ghost_2_1_no_normalized_no_trans_tflite.tflite --outputFile=$models_path/hiai_bigmodel_ghost_2_1_no_normalized_no_trans_tflite


#推送到手机上执行benchmark：

#一：复制到手机所在的机器上
mkdir -p ./benchmark_test
cp  $arm_path/MSLite-0.6.0-linux_arm64/lib/libmindspore-lite.so ./benchmark_test/libmindspore-lite.so
cp  $arm_path/MSLite-0.6.0-linux_arm64/benchmark/benchmark ./benchmark_test/benchmark

#复制模型到连接手机服务器所在目录：
#模型1：
cp  $models_path/test.ms ./benchmark_test/
#模型2：
cp  $models_path/hiai_bigmodel_ghost_2_1_no_normalized_no_trans_tflite.ms ./benchmark_test/

#二：adb 推送到手机上
adb -s $device_id push ./benchmark_test /data/local/tmp/

#三：执行adb命令，运行推理，获取返回值判断结果；
echo 'cd  /data/local/tmp/' > adb_cmd.txt
echo 'chmod 777 benchmark' >> adb_cmd.txt
#模型1：
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/;./benchmark --modelPath=test.ms' >> adb_cmd.txt
#模型2：
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/benchmark_test;./benchmark --modelPath=hiai_bigmodel_ghost_2_1_no_normalized_no_trans_tflite.ms' >> adb_cmd.txt

adb -s $device_id shell < adb_cmd.txt