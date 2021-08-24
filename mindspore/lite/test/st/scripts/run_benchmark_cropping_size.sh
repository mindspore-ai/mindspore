#!/bin/bash
source ./scripts/base_functions.sh

# Run converter on x86 platform:
function Run_Converter() {
    # Unzip x86 runtime and converter
    cd ${x86_path} || exit 1
    tar -zxf mindspore-lite-${version}-linux-x64.tar.gz || exit 1
    cd ${x86_path}/mindspore-lite-${version}-linux-x64/ || exit 1

    cp tools/converter/converter/converter_lite ./ || exit 1
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:./tools/converter/lib/:./tools/converter/third_party/glog/lib

    rm -rf ${ms_models_path}
    mkdir -p ${ms_models_path}

    # Prepare the config file list
    local cfg_file_list=("$models_cropped_config")
    # Convert models:
    # $1:cfgFileList; $2:inModelPath; $3:outModelPath; $4:logFile; $5:resultFile;
    Convert "${cfg_file_list[*]}" $models_path $ms_models_path $run_converter_log_file $run_converter_result_file
}

# Run on arm64 platform:
function Run_arm64() {
    # Prepare the config file list
    local cropping_cfg_file_list=("$models_cropped_config")
    # Run converted models:
    # $1:cfgFileList; $2:modelPath; $3:dataPath; $4:logFile; $5:resultFile; $6:platform; $7:processor; $8:phoneId;
    Run_Benchmark "${cropping_cfg_file_list[*]}" . '/data/local/tmp' $run_arm64_log_file $run_benchmark_result_file 'arm64' 'CPU' $device_id
}

function Run_cropping() {
    # guard cropper
    echo "start Run cropping ... "
    cd ${basepath} || exit 1

    cropped_size_config="${basepath}"/../config/cropped_size.cfg

    cd ${arm64_path} || exit 1
    tar -zxf mindspore-lite-${version}-android-aarch64.tar.gz || exit 1
    cd mindspore-lite-${version}-android-aarch64 || exit 1
    cp -a ./runtime/lib/lib* "${cropper_test_path}"/ || exit 1
    cp -r "${x86_path}"/mindspore-lite-${version}-linux-x64/tools/cropper/ "${cropper_test_path}" || exit 1

    cd "${cropper_test_path}" || exit 1
    echo "${cropper_test_path}"

    ls -l libmindspore-lite.a || exit 1
    ls -l -h libmindspore-lite.a || exit 1
    ls -l -h libmindspore-lite.so || exit 1

    echo "./cropper/cropper --packageFile=./libmindspore-lite.a --configFile=./cropper/cropper_mapping_cpu.cfg --modelFile=${ms_models_path}/${model_name}.ms --outputFile=./libmindspore-lite-${model_name}.a"
    ./cropper/cropper --packageFile=./libmindspore-lite.a --configFile=./cropper/cropper_mapping_cpu.cfg --modelFile=${ms_models_path}/${model_name}.ms --outputFile=./libmindspore-lite-${model_name}.a

    if [ $? = 0 ]; then
        run_result='cropper_lib pass'; echo ${run_result} >> "${run_cropper_result}"
    else
        run_result='cropper_lib failed'; echo ${run_result} >> "${run_cropper_result}"; return 1
    fi
    echo "after cropped:"
    ls -l libmindspore-lite-${model_name}.a || exit 1
    ls -l -h libmindspore-lite-${model_name}.a || exit 1
    mkdir -p mm || exit 1
    cp libmindspore-lite-${model_name}.a mm/ || exit 1
    cd mm/ || exit 1
    ar -x libmindspore-lite-${model_name}.a || exit 1
    ar -d libmindspore-lite-${model_name}.a  *.S.o || exit 1
    cd ../ || exit 1
    echo "after ar -d libmindspore-lite-${model_name}.a  *.S.o"
    ls -l mm/libmindspore-lite-${model_name}.a || exit 1
    ls -l -h mm/libmindspore-lite-${model_name}.a || exit 1

    "${ANDROID_NDK}"/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ \
    --target=aarch64-none-linux-android21 \
    --gcc-toolchain="${ANDROID_NDK}"/toolchains/llvm/prebuilt/linux-x86_64 \
    --sysroot="${ANDROID_NDK}"/toolchains/llvm/prebuilt/linux-x86_64/sysroot \
    -fPIC -D_FORTIFY_SOURCE=2 -O2 -Wall -Werror -Wno-attributes \
    -Wno-deprecated-declarations    -Wno-missing-braces \
    -Wno-overloaded-virtual -std=c++17 -fPIC -fPIE -fstack-protector-strong  \
    -DANDROID -fdata-sections -ffunction-sections -funwind-tables \
    -fstack-protector-strong -no-canonical-prefixes -fno-addrsig -Wa,--noexecstack \
    -Wformat -Werror=format-security    -fomit-frame-pointer -fstrict-aliasing \
    -ffunction-sections  -fdata-sections -ffast-math -fno-rtti -fno-exceptions \
    -Wno-unused-private-field -O2 -DNDEBUG  -Wl,-z,relro -Wl,-z,now \
    -Wl,-z,noexecstack -s  -Wl,--exclude-libs,libgcc.a \
    -Wl,--exclude-libs,libatomic.a -Wl,--build-id -Wl,--warn-shared-textrel \
    -Wl,--fatal-warnings -Wl,--no-undefined -Qunused-arguments -Wl,-z,noexecstack  \
    -shared -Wl,-soname,libmindspore-lite.so -o libmindspore-lite.so \
    -Wl,--whole-archive ./mm/libmindspore-lite-${model_name}.a -Wl,--no-whole-archive  \
    -llog -ldl -latomic -lm

    if [ $? = 0 ]; then
        run_result='link_lib_to_so pass'; echo ${run_result} >> "${run_cropper_result}"
    else
        run_result='link_lib_to_so failed'; echo ${run_result} >> "${run_cropper_result}"; return 1
    fi

    ls -l libmindspore-lite.so  || exit 1
    ls -l -h libmindspore-lite.so  || exit 1
    so_size=`ls libmindspore-lite.so  -l|awk -F ' ' '{print $5}'`
    calib_size=`cat ${cropped_size_config}`
    echo "now size:${so_size}." >> "${run_cropper_result}";
    echo "calib_size size:${calib_size}." >> "${run_cropper_result}";

    if [[ ${so_size} -gt ${calib_size} ]];then
      echo ${so_size}
      add_size=$(((so_size-calib_size)/2))
      add_per=`awk 'BEGIN{printf "%.3f\n",'${add_size}'/'${calib_size}'}'`
      if [[ `echo ${add_per}|awk '{if($1 > 0.05) {printf "1"} else {printf "0"}}'` -eq 1 ]];then
        echo "calib_size failed." >> "${run_cropper_result}";
        run_result="Error: The increased basic framework code size has exceeded the threshold since the last review. Please check the code or review again.";
        echo ${run_result} >> "${run_cropper_result}";
        return 1
      fi
    fi
    echo "calib_size success." >> "${run_cropper_result}";
    return 0
}

basepath=$(pwd)
echo ${basepath}
#set -e

# Example:sh run_benchmark_arm64.sh -r /home/temp_test -m /home/temp_test/models -d "8KE5T19620002408" -e arm_cpu
while getopts "r:m:d:e:" opt; do
    case ${opt} in
        r)
            release_path=${OPTARG}
            echo "release_path is ${OPTARG}"
            ;;
        m)
            models_path=${OPTARG}
            echo "models_path is ${OPTARG}"
            ;;
        d)
            device_id=${OPTARG}
            echo "device_id is ${OPTARG}"
            ;;
        e)
            backend=${OPTARG}
            echo "backend is ${OPTARG}"
            ;;
        ?)
        echo "unknown para"
        exit 1;;
    esac
done

cropper_test_path="${basepath}"/cropper_test
rm -rf "${cropper_test_path}"
mkdir -p "${cropper_test_path}"

# mkdir train
x86_path=${release_path}/ubuntu_x86
arm64_path=${release_path}/android_aarch64/cropping
file_name=$(ls ${x86_path}/*linux-x64.tar.gz)
IFS="-" read -r -a file_name_array <<< "$file_name"
version=${file_name_array[2]}

# Set models config filepath
models_cropped_config=${basepath}/../config/models_cropping.cfg

ms_models_path=${basepath}/ms_models

# Write converter result to temp file
run_converter_log_file=${basepath}/run_converter_log.txt
echo ' ' > ${run_converter_log_file}

run_converter_result_file=${basepath}/run_converter_result.txt
echo ' ' > ${run_converter_result_file}

# Run converter
echo "start Run converter ..."
Run_Converter
Run_converter_status=$?
# Check converter result and return value
if [[ ${Run_converter_status} = 0 ]];then
    echo "Run converter success"
    Print_Converter_Result $run_converter_result_file
else
    echo "Run converter failed"
    cat ${run_converter_log_file}
    Print_Converter_Result $run_converter_result_file
    exit 1
fi

model_name=add_extend
cp ${models_path}/${model_name}.ms  ${ms_models_path}/  || exit 1

# Write benchmark result to temp file
run_benchmark_result_file=${basepath}/run_benchmark_result.txt
echo ' ' > ${run_benchmark_result_file}

run_arm64_log_file=${basepath}/run_arm64_log.txt
echo 'run arm64 logs: ' > ${run_arm64_log_file}

# Copy the MindSpore models:
echo "Push ficropper_configles to the arm and run benchmark"
benchmark_test_path=${basepath}/benchmark_test
rm -rf ${benchmark_test_path}
mkdir -p ${benchmark_test_path}
cp -a ${ms_models_path}/*.ms ${benchmark_test_path} || exit 1

# Push files to the phone
Push_Files $arm64_path "aarch64" $version $benchmark_test_path "adb_push_log.txt" $device_id

backend=${backend:-"all"}
isFailed=0

# Run on arm64
echo "start Run arm64 ..."
Run_arm64
Run_arm64_status=$?
# Run_arm64_PID=$!
# sleep 1

run_cropper_result="${basepath}"/run_cropper_result.txt
echo ' ' > "${run_cropper_result}"

Run_cropping
Run_cropping_status=$?

if [[ ${Run_arm64_status} != 0 ]];then
    echo "Run_arm64 failed"
    cat ${run_arm64_log_file}
    isFailed=1
fi
if [[ ${Run_cropping_status} != 0 ]];then
    echo "Run cropping failed"
    cat ${run_cropper_result}
    isFailed=1
fi

echo "Run_arm64 and Run_cropping is ended"
Print_Benchmark_Result $run_benchmark_result_file
exit ${isFailed}
