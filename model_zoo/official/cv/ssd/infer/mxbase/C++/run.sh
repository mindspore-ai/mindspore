export LD_LIBRARY_PATH=${MX_SDK_HOME}/lib:${MX_SDK_HOME}/lib/modelpostprocessors:${MX_SDK_HOME}/opensource/lib:${MX_SDK_HOME}/opensource/lib64:/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64:${LD_LIBRARY_PATH} 
OM_FILE=/data/pretrained_models/ms/mobilenet_v1/ckpt_0/mobilenetv1-90_625_2.om

cp ./build/ssd_mobilenet_v1_fpn .
./ssd_mobilenet_v1_fpn ${OM_FILE} ./test.jpg
