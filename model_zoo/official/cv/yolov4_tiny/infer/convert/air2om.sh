model_path=$1
output_model_name=$2

/usr/local/Ascend/atc/bin/atc \
--model=$model_path \
--framework=1 \
--output=$output_model_name \
--input_format=NCHW --input_shape="actual_input_1:1,3,608,608" \
--enable_small_channel=1 \
--log=error \
--soc_version=Ascend310 \
--insert_op_conf=./aipp.config