#!/bin/bash

echo "$tag =========================== starting the ci for fl-lenet-train-eval-infer: $0 ===========================";

ci_start_time=`date +%s`
tag="[FL_CI]"
server_start_time_window=30
client_train_time_windom=50
client_inference_time_windom=20
server_success_tag="Server started successfully"
train_finish_tag="\[onFlJobFinished\]"
inference_finish_tag="inference finish"

resource_path=$1
packages_path=$2
jdk_path=$3


scrip_path=$(dirname "$(dirname "$(readlink -f "$0")")")
cloud_tarin=$scrip_path/test_mobile_lenet.py

echo "$tag the resource_path: $resource_path"
echo "$tag the packages_path: $packages_path"
echo "$tag the scrip_path: $scrip_path"
echo "$tag the cloud_tarin file: $cloud_tarin"

exit_opt()
{
  echo "$tag Clear temporary files."
  cd $scrip_path
  rm -rf temp

  echo "$tag finish server"
  python finish_mobile.py --scheduler_port=6001

  echo "$tag finish client"
  cd ./ci_script
  python fl_client_finish.py --kill_tag=mindspore-lite-java-flclient

  echo "$tag del code for server"
  cd $resource_path/server/script
  sh del_code.sh $cloud_tarin

  ci_inter=`date +%s`
  ci_inter_time=`echo $ci_start_time $ci_inter | awk '{print $2-$1}'`
  echo "$tag the total cost time is: $ci_inter_time s"
}

check_exe_result()
{
  if [ "$?" != "0" ]; then
    echo "$tag catch error when $1, will return 1, please check"
    exit_opt
    exit 1
  fi
}

check_document() {
  if [ ! -d "$1" ]; then
    echo "$tag the $2: $1 is not exist, will return 1, please check"
    exit_opt
    exit 1
  fi
}

check_file() {
  if [ ! -f "$1" ]; then
    echo "$tag the $2: $1 is not exist, will return 1, please check"
    exit_opt
    exit 1
  fi
}

echo "$tag **************** <1> check resource_path, packages_path, model train script ****************"
check_document $resource_path "resource_path"
check_document $packages_path "packages_path"
check_file $cloud_tarin "cloud_tarin"

client_package=$(ls $packages_path/mindspore-lite-*-linux-x64.tar.gz)
server_package=$(ls $packages_path/mindspore*-linux_x86_64.whl)
echo "$tag the client_package: $client_package"
echo "$tag the server_package: $server_package"

if [ ! -f "$client_package" ] || [ ! -f "$server_package" ]; then
  echo "$tag the client_package or the server_package is not exist, do not start the FL ci, will return 0."
  exit_opt
  exit 0
fi


echo "$tag **************** <2> get ip ****************"
ip_info=$(hostname -I)
echo "ip information: $ip_info"
#ip_array=(${ip_info/// })
IFS=" " read -r -a ip_array <<< "$ip_info"
ip=${ip_array[0]}
echo "the main ip: $ip"

echo "$tag **************** <3> Creating a Temporary Directory ****************"
cd $scrip_path
check_exe_result "cd $scrip_path"
rm -rf temp
mkdir temp
cd temp
mkdir server
mkdir client
mkdir packages
cd server
mkdir init
mkdir train_log
cd ../packages
mkdir libs
cd ../client
mkdir ms
temp_path=$scrip_path/temp

echo "$tag ****add code for server****"
cd $resource_path/server/script/
sh add_code.sh $cloud_tarin
check_exe_result "add code for server"

echo "$tag **************** <4> prepare parameters for server ****************"
scheduler_ip=$ip
scheduler_port=6001
scheduler_manage_port=6000
fl_server_port=6003
server_num=1
worker_num=0
enable_ssl="False"
config_file_path=$scrip_path/config.json
start_fl_job_threshold=1
client_batch_size=32
client_epoch_num=1
fl_iteration_num=1
start_fl_job_time_window=30000
update_model_time_window=30000
encrypt_type="NOT_ENCRYPT"

echo "$tag ****check the parameters of server****"
check_file $config_file_path "config_file_path"
echo "$tag ****the parameters of server are ok****"

echo "$tag **************** <5> prepare libs,jar,initial model for client****************"
cd $resource_path/client/
cp -rf $client_package $temp_path/packages/
cd $temp_path/packages
tar -zxvf mindspore-*.tar.gz
libminddata_lite=$(ls $temp_path/packages/mindspore-*/runtime/lib/libminddata-lite.so)
libmindspore_lite_jni=$(ls $temp_path/packages/mindspore-*/runtime/lib/libmindspore-lite-jni.so)
libmindspore_lite_train=$(ls $temp_path/packages/mindspore-*/runtime/lib/libmindspore-lite-train.so)
libmindspore_lite_train_jni=$(ls $temp_path/packages/mindspore-*/runtime/lib/libmindspore-lite-train-jni.so)
libmindspore_lite=$(ls $temp_path/packages/mindspore-*/runtime/lib/libmindspore-lite.so)
libjpeg=$(ls $temp_path/packages/mindspore-*/runtime/third_party/libjpeg-turbo/lib/libjpeg.so.62)
libturbojpeg=$(ls $temp_path/packages/mindspore-*/runtime/third_party/libjpeg-turbo/lib/libturbojpeg.so.0)

echo "$tag **** 5-1: check the .so files for fl****"
check_file $libminddata_lite "libminddata_lite"
check_file $libmindspore_lite_jni "libmindspore_lite_jni"
check_file $libmindspore_lite_train "libmindspore-lite-train"
check_file $libmindspore_lite_train_jni "libmindspore-lite-train-jni"
check_file $libmindspore_lite "libmindspore-lite"
check_file $libjpeg "libjpeg"
check_file $libturbojpeg "libturbojpeg"
echo "$tag the .so files for fl are exist"

cp -rf $libminddata_lite ./libs/      # all so ?
cp -rf $libmindspore_lite_jni ./libs/
cp -rf $libmindspore_lite_train ./libs/
cp -rf $libmindspore_lite_train_jni ./libs/
cp -rf $libmindspore_lite ./libs/
cp -rf $libjpeg ./libs/
cp -rf $libturbojpeg ./libs/
libs_path=$temp_path/packages/libs

echo "$tag ****5-2: prepare case jar for client****"
raw_jar_path=$(ls $temp_path/packages/mindspore-*/runtime/lib/mindspore-lite-java-flclient.jar)
mkdir frame_jar
cp -rf $raw_jar_path ./frame_jar
frame_jar_path=$(ls $temp_path/packages/frame_jar/mindspore-lite-java-flclient.jar)
echo "$tag check the frame jar file: $frame_jar_path for fl"
check_file $frame_jar_path "frame_jar_path"
echo "$tag the frame jar file: $frame_jar_path for fl is exist"

echo "$tag ****5-3: prepare case jar for client****"
mkdir case_jar
root_path=$(dirname "$(dirname "$(dirname "$(dirname $scrip_path)")")")
case_code_path=$root_path/mindspore/lite/examples/quick_start_flclient
client_tar_path=$(ls $temp_path/packages/mindspore-*.tar.gz)
cd $case_code_path
sh ./build.sh -r $client_tar_path
check_exe_result "sh ./build.sh -r $client_tar_path"
build_case_jar_path=$case_code_path/target/quick_start_flclient.jar
echo "$tag check the case jar file <$build_case_jar_path> after run sh build.sh in document <$case_code_path> for fl"
check_file $build_case_jar_path "build_case_jar_path"
echo "$tag the case jar file: $build_case_jar_path for fl is exist"
cp -rf $build_case_jar_path $temp_path/packages/case_jar/
case_jar_path=$temp_path/packages/case_jar/quick_start_flclient.jar
echo "$tag check the case jar file: $case_jar_path for fl"
check_file $case_jar_path "case_jar_path"
echo "$tag the case jar file: $case_jar_path for fl is exist"

echo "$tag ****5-4: prepare initial model for client****"
cd $resource_path/client/
cp -rf ./ms/lenet_train.mindir0.ms $temp_path/client/ms/
check_exe_result "cp -rf ./ms/lenet_train.mindir0.ms $temp_path/client/ms/"

echo "$tag **************** <6> prepare parameters for client ****************"
train_dataset=$resource_path/client/data/f0049_32
flName="com.mindspore.flclient.demo.lenet.LenetClient"
train_model_path=$temp_path/client/ms/lenet_train.mindir0.ms
infer_model_path=$temp_path/client/ms/lenet_train.mindir0.ms
ssl_protocol="TLSv1.2"
deploy_env="x86"
domain_name=http://$scheduler_ip:$fl_server_port
cert_path=$resource_path/client/cert/CARoot.pem
server_num=1
client_num=1
use_elb="false"
thread_num=4
server_mode="FEDERATED_LEARNING"
batch_size=$client_batch_size
task1="train"
task2="inference"

echo "$tag ****check the parameters of client****"
check_document $train_dataset "train_dataset"
check_file $train_model_path "train_model_path"
check_file $infer_model_path "infer_model_path"
check_file $cert_path "cert_path"
echo "$tag ****the parameters of client are ok****"


echo "$tag **************** <7> get the log files path ****************"
train_log_path=$scrip_path/ci_script/client_train0/client-train.log
inference_log_path=$scrip_path/ci_script/client_inference0/client-inference.log
server_log=$scrip_path/server_0/server.log
echo "$tag train_log_path: $train_log_path"
echo "$tag inference_log_path: $inference_log_path"
echo "$tag server_log: $server_log"

echo "$tag **************** <8> tart server ****************"
cd $scrip_path
rm -rf server_*
rm -rf scheduler
rm -rf worker_*
cmd_server="python run_mobile_sched.py --scheduler_ip=$scheduler_ip --scheduler_port=$scheduler_port --server_num=$server_num --worker_num=$worker_num --scheduler_manage_port=$scheduler_manage_port --enable_ssl=$enable_ssl --config_file_path=$config_file_path && python run_mobile_server.py --scheduler_ip=$scheduler_ip --scheduler_port=$scheduler_port --fl_server_port=$fl_server_port --server_num=$server_num --worker_num=$worker_num --start_fl_job_threshold=$start_fl_job_threshold --client_batch_size=$client_batch_size --client_epoch_num=$client_epoch_num --fl_iteration_num=$fl_iteration_num --start_fl_job_time_window=$start_fl_job_time_window --update_model_time_window=$update_model_time_window --encrypt_type=$encrypt_type --enable_ssl=$enable_ssl --config_file_path=$config_file_path"
echo "$tag $cmd_server"

server_start=`date +%s`
server_tag=1

python run_mobile_sched.py --scheduler_ip=$scheduler_ip --scheduler_port=$scheduler_port --server_num=$server_num --worker_num=$worker_num --scheduler_manage_port=$scheduler_manage_port --enable_ssl=$enable_ssl --config_file_path=$config_file_path \
&& python run_mobile_server.py --scheduler_ip=$scheduler_ip --scheduler_port=$scheduler_port --fl_server_port=$fl_server_port --server_num=$server_num \
--worker_num=$worker_num --start_fl_job_threshold=$start_fl_job_threshold --client_batch_size=$client_batch_size --client_epoch_num=$client_epoch_num --fl_iteration_num=$fl_iteration_num \
--start_fl_job_time_window=$start_fl_job_time_window --update_model_time_window=$update_model_time_window --encrypt_type=$encrypt_type --enable_ssl=$enable_ssl --config_file_path=$config_file_path

#echo "$tag ****check servre log file****"
#check_document $server_log "server_log"
logcat1=""
until [ "$server_tag" = 0 ];
do
  inter=`date +%s`
  inter_time=`echo $server_start $inter | awk '{print $2-$1}'`
  if [ $inter_time -ge $server_start_time_window ]; then
    echo "$tag server start out of time"
    break
#    exit_opt
#    exit 1
  fi
  logcat1=$(grep -r "$server_success_tag" $server_log)
  server_tag=$?
done
if [ "$server_tag" = 0 ]; then
  echo "$tag server started successfully"
fi
server_end=`date +%s`
server_time=`echo $server_start $server_end | awk '{print $2-$1}'`
echo "$tag server logcat1: $logcat1"
echo "$tag the cost time of starting server: $server_time s"


echo "$tag **************** <9> tart client training ****************"
echo "$tag set LD_LIBRARY_PATH for client"
export LD_LIBRARY_PATH=$libs_path:$LD_LIBRARY_PATH

echo "$tag ****check jdk ptah: $jdk_path for client****"
check_document $jdk_path "jdk_path"
echo "$tag set jdk ptah for client"
export PATH=$jdk_path:$PATH
check_exe_result "export PATH=$jdk_path:$PATH"

train_start=`date +%s`
train_tag=1
cd ./ci_script
python fl_client_run_lenet.py --jarPath=$frame_jar_path  --case_jarPath=$case_jar_path --train_dataset=$train_dataset \
--test_dataset="null" --vocal_file="null" --ids_file="null" --flName=$flName --train_model_path=$train_model_path \
--infer_model_path=$infer_model_path --ssl_protocol=$ssl_protocol  --deploy_env=$deploy_env --domain_name=$domain_name \
 --cert_path=$cert_path --server_num=$server_num --client_num=$client_num --use_elb=$use_elb --thread_num=$thread_num \
--server_mode=$server_mode --batch_size=$batch_size --task=$task1

#sleep 5
#echo "$tag ****check train log file****"
#check_document $train_log_path "train_log_path"

until [ "$train_tag" = 0 ];
do
  train_inter=`date +%s`
  train_inter_time=`echo $train_start $train_inter | awk '{print $2-$1}'`
  if [ $train_inter_time -ge $client_train_time_windom ]; then
    echo "$tag client train out of time"
    break
#    exit_opt
#    exit 1
  fi
  logcat2=$(grep -r "$train_finish_tag" $train_log_path)
  train_tag=$?
done

if [ "$train_tag" = 0 ]; then
  echo "$tag client train finished"
fi

train_end=`date +%s`
train_time=`echo $train_start $train_end | awk '{print $2-$1}'`
echo "$tag the cost time of client training: $train_time s"


echo "$tag **************** <10> start client inference ****************"
inference_start=`date +%s`
inference_tag=1
python fl_client_run_lenet.py --jarPath=$frame_jar_path  --case_jarPath=$case_jar_path --train_dataset=$train_dataset \
--test_dataset="null" --vocal_file="null" --ids_file="null" --flName=$flName --train_model_path=$train_model_path \
--infer_model_path=$infer_model_path --ssl_protocol=$ssl_protocol  --deploy_env=$deploy_env --domain_name=$domain_name \
 --cert_path=$cert_path --server_num=$server_num --client_num=$client_num --use_elb=$use_elb --thread_num=$thread_num \
--server_mode=$server_mode --batch_size=$batch_size --task=$task2

#echo "$tag ****check inference log file****"
#check_document $inference_log_path "inference_log_path"
logcat2=""
until [ "$inference_tag" = 0 ];
do
  inference_inter=`date +%s`
  inference_inter_time=`echo $inference_start $inference_inter | awk '{print $2-$1}'`
  if [ $inference_inter_time -ge $client_inference_time_windom ]; then
    echo "$tag client inference out of time"
    break
#    exit_opt
#    exit 1
  fi
  logcat2=$(grep -r "$inference_finish_tag" $inference_log_path)
  inference_tag=$?
done

if [ "$inference_tag" = 0 ]; then
  echo "$tag client inference finished"
fi

inference_end=`date +%s`
inference_time=`echo $inference_start $inference_end | awk '{print $2-$1}'`
echo "$tag inference logcat: $logcat2"
echo "$tag the cost time of client inference: $inference_time s"


clear_log()
{
  echo "$tag success, please clear the client and server logs"
}


train_result="success"
inference_result="success"
train_keywords1="the total response of 1: SUCCESS"
train_keywords2="\[onFlJobFinished\] modelName: $flName iterationCount: 1 resultCode: 200"
inference_keywords1="the predicted outputs"
inference_keywords2="inference finish"
scheduler_log=$scrip_path/scheduler/scheduler.log
worker_log=$scrip_path/worker_0/worker.log

echo "$tag ********check train and inference log files********"
check_file $train_log_path "train_log_path"
check_file $inference_log_path "inference_log_path"

echo "$tag ********check the training results********"
train_logcat1=$(grep -r "$train_keywords1" $train_log_path)
r1=$?
train_logcat2=$(grep -r "$train_keywords2" $train_log_path)
r2=$?
echo "$tag train_logcat1: $train_logcat1"
echo "$tag train_logcat2: $train_logcat2"

if [ "$r1" != "0" ] || [ "$r2" != "0" ]; then
  echo "$tag train failed: "
  if [ "$r1" != "0" ]; then
    echo "$tag the Keyword < $train_keywords1 > does not appear in the log"
  fi

  if [ "$r2" != "0" ]; then
    echo "$tag the Keyword < $train_keywords2 > does not appear in the log"
  fi

  echo "$tag please check: "
  echo "$tag the client train log< $train_log_path >"
  echo "$tag the server log< $server_log >"
  echo "$tag the scheduler log< $scheduler_log >"
  echo "$tag the worker log< $worker_log >"
  train_result="failed"
fi

echo "$tag ********check the inference results********"
inference_logcat1=$(grep -r "$inference_keywords1" $inference_log_path)
r3=$?
inference_logcat2=$(grep -r "$inference_keywords2" $inference_log_path)
r4=$?

echo "$tag inference_logcat1: $inference_logcat1"
echo "$tag inference_logcat2: $inference_logcat2"

labels=${inference_logcat1##*:}
#array=(${labels//,/ })
IFS="," read -r -a array <<< "$labels"
labels_num=${#array[@]}
echo "$tag predicted labels: $labels"
echo "$tag the number of predicted labels is: $labels_num"

if [ "$r3" != "0" ] || [ "$r4" != "0" ] || [ "$labels_num" != "$batch_size" ]; then
  echo "$tag inference failed: "
  if [ "$r3" != "0" ]; then
    echo "$tag the Keyword < $inference_keywords1 > does not appear in the log"
  fi

  if [ "$r4" != "0" ]; then
    echo "$tag the Keyword < $inference_keywords2 > does not appear in the log"
  fi

  if [ "$labels_num" != "$batch_size" ]; then
    echo "$tag the number of predicted labels is not right, must be $batch_size"
  fi

  echo "$tag please check: "
  echo "$tag the client inference log< $inference_log_path >"
  echo "$tag the server log< $server_log >"
  echo "$tag the scheduler log< $scheduler_log >"
  echo "$tag the worker log< $worker_log >"
  inference_result="failed"
fi

if [ "$train_result" = failed ] || [ "$inference_result" = failed ]; then
  echo "$tag the total results are, train: $train_result, inference: $inference_result"
  exit_opt
  exit 1
fi

echo "$tag the total results are, train: $train_result, inference: $inference_result"
exit_opt
clear_log
exit 0


