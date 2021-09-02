#!/bin/bash
source ./base_functions.sh

function Print_Cropper_Result() {
    MS_PRINT_TESTCASE_START_MSG
    while read line; do
        arr=("${line}")
        printf "%-20s %-100s %-7s\n" ${arr[0]} ${arr[1]} ${arr[2]}
    done < "${run_cropper_result}"
    MS_PRINT_TESTCASE_END_MSG
}

function Run_cropper() {
    cd ${arm64_path} || exit 1
    tar -zxf mindspore-lite-${version}-android-aarch64.tar.gz || exit 1
    cd mindspore-lite-${version}-android-aarch64 || exit 1
    cp -a ./runtime/third_party/hiai_ddk/lib/libhiai.so "${cropper_test_path}"/libhiai.so || exit 1
    cp -a ./runtime/third_party/hiai_ddk/lib/libhiai_ir.so "${cropper_test_path}"/libhiai_ir.so || exit 1
    cp -a ./runtime/third_party/hiai_ddk/lib/libhiai_ir_build.so "${cropper_test_path}"/libhiai_ir_build.so || exit 1

    cp -a ./runtime/lib/libmindspore-lite.a "${cropper_test_path}"/libmindspore-lite.a || exit 1
    cp -a ./tools/benchmark/benchmark "${cropper_test_path}"/benchmark || exit 1

    cp -r "${x86_path}"/mindspore-lite-${version}-linux-x64/tools/cropper/ "${cropper_test_path}" || exit 1

    cd "${cropper_test_path}" || exit 1
    echo "${cropper_test_path}"

    # adb push all needed files to the phone
    adb -s ${device_id} push "${cropper_test_path}" /data/local/tmp/ > adb_push_log.txt

    # run adb ,run session ,check the result:
    echo 'rm -rf /data/local/tmp/cropper_test' > adb_cmd.txt
    echo 'cd  /data/local/tmp/cropper_test' > adb_cmd.txt
    echo 'cp  /data/local/tmp/libc++_shared.so ./' >> adb_cmd.txt
    echo 'chmod 777 benchmark' >> adb_cmd.txt

    adb -s ${device_id} shell < adb_cmd.txt

    while read line; do
        model_line_info=${line}
        if [[ $model_line_info == \#* ]]; then
          continue
        fi
        model_name=`echo ${line}|awk -F ' ' '{print $1}'`

        echo "./cropper/cropper --packageFile=./libmindspore-lite.a --configFile=./cropper/cropper_mapping_npu.cfg --modelFile=${ms_models_path}/${model_name}.ms --outputFile=./libmindspore-lite-${model_name}.a"
         ./cropper/cropper --packageFile=./libmindspore-lite.a --configFile=./cropper/cropper_mapping_npu.cfg --modelFile=${ms_models_path}/${model_name}.ms --outputFile=./libmindspore-lite-${model_name}.a

        if [ $? = 0 ]; then
            run_result='cropper_lib: '${line}' pass'; echo ${run_result} >> "${run_cropper_result}"
        else
            run_result='cropper_lib: '${line}' failed'; echo ${run_result} >> "${run_cropper_result}"; return 1
        fi

       "${ANDROID_NDK}"/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ -Wl,--whole-archive ./libmindspore-lite-${model_name}.a \
       -Wl,--no-whole-archive --target=aarch64-none-linux-android21 \
       --gcc-toolchain="${ANDROID_NDK}"/toolchains/llvm/prebuilt/linux-x86_64 \
       --sysroot="${ANDROID_NDK}"/toolchains/llvm/prebuilt/linux-x86_64/sysroot \
       -fPIC -D_FORTIFY_SOURCE=2 -O2 -Wall -Werror -Wno-attributes -Wno-deprecated-declarations -Wno-missing-braces \
       -Wno-overloaded-virtual -std=c++17 -fPIC -fPIE -fstack-protector-strong -DANDROID -fdata-sections \
       -ffunction-sections -funwind-tables -fstack-protector-strong -no-canonical-prefixes -fno-addrsig \
       -Wa,--noexecstack -Wformat -Werror=format-security -fomit-frame-pointer -fstrict-aliasing \
       -ffunction-sections -fdata-sections -ffast-math -fno-rtti -fno-exceptions -Wno-unused-private-field \
       -O2 -DNDEBUG -Wl,-z,relro -Wl,-z,now -Wl,-z,noexecstack -s \
       -Wl,--exclude-libs,libgcc.a -Wl,--exclude-libs,libatomic.a -static-libstdc++ -Wl,--build-id \
       -Wl,--warn-shared-textrel -Wl,--fatal-warnings -Wl,--no-undefined -Qunused-arguments \
       -Wl,-z,noexecstack \
       -L "${cropper_test_path}" -lhiai -lhiai_ir -lhiai_ir_build \
       -shared -Wl,-soname,libmindspore-lite.so -o libmindspore-lite.so -llog -ldl -latomic -lm
        
        if [ $? = 0 ]; then
            run_result='link_lib_to_so: '${line}' pass'; echo ${run_result} >> "${run_cropper_result}"
        else
            run_result='link_lib_to_so: '${line}' failed'; echo ${run_result} >> "${run_cropper_result}"; return 1
        fi
        adb -s ${device_id} push "${cropper_test_path}"/libmindspore-lite.so /data/local/tmp/cropper_test > adb_push_log.txt

        echo "mindspore run cropper: ${model_name}, accuracy limit:4" >> "${run_cropper_log_file}"
        echo 'cd  /data/local/tmp/cropper_test' > adb_run_cmd.txt
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/cropper_test;./benchmark --device=GPU --modelFile=/data/local/tmp/benchmark_test/'${model_name}'.ms --loopCount=1 --warmUpLoopCount=0' >> "${run_cropper_log_file}"
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/cropper_test;./benchmark --device=GPU --modelFile=/data/local/tmp/benchmark_test/'${model_name}'.ms --loopCount=1 --warmUpLoopCount=0' >> adb_run_cmd.txt
        adb -s ${device_id} shell < adb_run_cmd.txt >> "${run_cropper_log_file}"
        if [ $? = 0 ]; then
            run_result='run_benchmark: '${model_name}' pass'; echo ${run_result} >> "${run_cropper_result}"
        else
            run_result='run_benchmark: '${model_name}' failed'; echo ${run_result} >> "${run_cropper_result}"; return 1
        fi
    done < ${cropper_config}
}

basepath=$(pwd)
echo "${basepath}"

# Example:sh run_cropper_nets.sh -r /home/temp_test -d "8KE5T19620002408"
while getopts "r:d:m:" opt; do
    case ${opt} in
        r)
            release_path=${OPTARG}
            echo "release_path is ${OPTARG}"
            ;;
        d)
            device_id=${OPTARG}
            echo "device_id is ${OPTARG}"
            ;;
        ?)
        echo "unknown para"
        exit 1;;
    esac
done

cropper_test_path="${basepath}"/cropper_test
rm -rf "${cropper_test_path}"
mkdir -p "${cropper_test_path}"

run_cropper_result="${basepath}"/run_cropper_result.txt
echo ' ' > "${run_cropper_result}"
run_cropper_log_file="${basepath}"/run_cropper_log.txt
echo 'run cropper logs: ' > "${run_cropper_log_file}"

cropper_config="${basepath}"/../config/models_cropper.cfg
arm64_path=${release_path}/android_aarch64/npu
x86_path=${release_path}/ubuntu_x86

# Write converter result to temp file
run_converter_log_file="${basepath}"/run_converter_log.txt
echo ' ' > "${run_converter_log_file}"

run_converter_result_file="${basepath}"/run_converter_result.txt
echo ' ' > "${run_converter_result_file}"

file_name=$(ls "${x86_path}"/*linux-x64.tar.gz)
IFS="-" read -r -a file_name_array <<< "$file_name"
version=${file_name_array[2]}
ms_models_path=${basepath}/ms_models

Run_cropper
Run_cropper_status=$?

if [[ $Run_cropper_status == 1 ]]; then
    cat "${run_cropper_log_file}"
    Print_Cropper_Result
    exit 1
fi

Print_Cropper_Result
exit 0
