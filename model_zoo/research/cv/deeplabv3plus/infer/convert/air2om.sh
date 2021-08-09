model_path=$1
framework=1
output_model_name=$2

atc \
    --model=$model_path \
    --framework=$framework \
    --output=$output_model_name \
    --input_format=NHWC \
    --log=error \
    --soc_version=Ascend310
