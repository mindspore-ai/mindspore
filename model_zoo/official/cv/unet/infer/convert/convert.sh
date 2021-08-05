model_path=$1
aipp_cfg_path=$2
output_model_name=$3

/usr/local/Ascend/atc/bin/atc \
--model=$model_path \
--input_format=NCHW \
--framework=1 \
--output=$output_model_name \
--log=error \
--soc_version=Ascend310 \
--insert_op_conf=$aipp_cfg_path