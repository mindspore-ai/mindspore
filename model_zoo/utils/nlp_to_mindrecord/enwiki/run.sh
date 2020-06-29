#!/bin/bash
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

if [ $# -ne 3 ]; then
    echo "Usage: $0 input_dir output_dir vocab_file"
    exit 1
fi

if [ ! -d $1 ]; then
    echo "The input dir: $1 is not exist."
    exit 1
fi

if [ ! -d $2 ]; then
    echo "The output dir: $2 is not exist."
    exit 1
fi
rm -fr $2/*.mindrecord*

if [ ! -f $3 ]; then
    echo "The vocab file: $3 is not exist."
    exit 1
fi

data_dir=$1
output_dir=$2
vocab_file=$3
file_list=()
output_filename=()
file_index=0

function getdir() {
    elements=`ls $1`
    for element in ${elements[*]};
    do
        dir_or_file=$1"/"$element
        if [ -d $dir_or_file ];
        then
            getdir $dir_or_file
        else
            file_list[$file_index]=$dir_or_file
            echo "${dir_or_file}" | tr '/' '\n' > dir_file_list.txt   # dir dir file to mapfile
            mapfile parent_dir < dir_file_list.txt
            rm dir_file_list.txt >/dev/null 2>&1
            tmp_output_filename=${parent_dir[${#parent_dir[@]}-1]}".mindrecord"
            output_filename[$file_index]=`echo ${tmp_output_filename} | sed 's/ //g'`
            file_index=`expr $file_index + 1`
        fi
    done
}

getdir "${data_dir}"
# echo "The input files: "${file_list[@]}
# echo "The output files: "${output_filename[@]}

if [ ! -d "../../../../third_party/to_mindrecord/zhwiki" ]; then
    echo "The patch base dir ../../../../third_party/to_mindrecord/zhwiki is not exist."
    exit 1
fi

if [ ! -f "../../../../third_party/patch/to_mindrecord/zhwiki/create_pretraining_data.patch" ]; then
    echo "The patch file ../../../../third_party/patch/to_mindrecord/zhwiki/create_pretraining_data.patch is not exist."
    exit 1
fi

# patch for create_pretraining_data.py
patch -p0 -d ../../../../third_party/to_mindrecord/zhwiki/ -o create_pretraining_data_patched.py < ../../../../third_party/patch/to_mindrecord/zhwiki/create_pretraining_data.patch
if [ $? -ne 0 ]; then
    echo "Patch ../../../../third_party/to_mindrecord/zhwiki/create_pretraining_data.py failed"
    exit 1
fi

# get the cpu core count
num_cpu_core=`cat /proc/cpuinfo | grep "processor" | wc -l`
avaiable_core_size=`expr $num_cpu_core / 3 \* 2`

echo "Begin preprocess `date`"

# using patched script to generate mindrecord
file_list_len=`expr ${#file_list[*]} - 1`
for index in $(seq 0 $file_list_len); do
    echo "Begin preprocess input file: ${file_list[$index]}"
    echo "Begin output file: ${output_filename[$index]}"
    python ../../../../third_party/to_mindrecord/zhwiki/create_pretraining_data_patched.py \
        --input_file=${file_list[$index]} \
        --output_file=${output_dir}/${output_filename[$index]} \
        --partition_number=1 \
        --vocab_file=${vocab_file} \
        --do_lower_case=True \
        --max_seq_length=512 \
        --max_predictions_per_seq=76 \
        --masked_lm_prob=0.15 \
        --random_seed=12345 \
        --dupe_factor=10 >/tmp/${output_filename[$index]}.log 2>&1 &
    process_count=`ps -ef | grep create_pretraining_data_patched | grep -v grep | wc -l`
    echo "Total task: ${#file_list[*]}, processing: ${process_count}"
    if [ $process_count -ge $avaiable_core_size ]; then
        while [ 1 ]; do
            process_num=`ps -ef | grep create_pretraining_data_patched | grep -v grep | wc -l`
            if [ $process_count -gt $process_num ]; then
                process_count=$process_num
                break;
            fi
            sleep 2
        done
    fi
done

process_num=`ps -ef | grep create_pretraining_data_patched | grep -v grep | wc -l`
while [ 1 ]; do
    if [ $process_num -eq 0 ]; then
        break;
    fi
    echo "There are still ${process_num} preprocess running ..."
    sleep 2
    process_num=`ps -ef | grep create_pretraining_data_patched | grep -v grep | wc -l`
done

echo "Preprocess all the data success."
echo "End preprocess `date`"
