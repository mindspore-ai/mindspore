/**
 * Copyright 2022 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "coder/generator/component/const_blocks/load_input.h"

namespace mindspore::lite::micro {
const char load_input_h[] = R"RAW(/**
 * Copyright 2022 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MINDSPORE_LITE_MICRO_LOAD_INPUT_H_
#define MINDSPORE_LITE_MICRO_LOAD_INPUT_H_

#ifdef __cplusplus
extern "C" {
#endif

void *ReadInputData(const char *real_input_path, int *size);

void SaveOutputData(char *final_name, unsigned char *output_data, unsigned int out_size);

int ReadInputsFile(char *path, void **buffers, const int *inputs_size, int inputs_num);

#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_LITE_MICRO_LOAD_INPUT_H_

)RAW";

const char load_input_c[] = R"RAW(/**
 * Copyright 2022 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "load_input.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "c_api/status_c.h"

void *ReadInputData(const char *real_input_path, int *size) {
  if (real_input_path == NULL) {
    return NULL;
  }
  if (strstr(real_input_path, ".bin") || strstr(real_input_path, ".net")) {
    FILE *file;
    file = fopen(real_input_path, "rb");
    if (!file) {
      printf("Can't find %s\n", real_input_path);
      return NULL;
    }
    int curr_file_posi = ftell(file);
    fseek(file, 0, SEEK_END);
    *size = ftell(file);
    unsigned char *buf = malloc((*size));
    (void)memset(buf, 0, (*size));
    fseek(file, curr_file_posi, SEEK_SET);
    int read_size = (int)(fread(buf, 1, *size, file));
    if (read_size != (*size)) {
      printf("read file failed, total file size: %d, read_size: %d\n", (*size), read_size);
      fclose(file);
      free(buf);
      return NULL;
    }
    fclose(file);
    return (void *)buf;
  } else {
    printf("input data file should be .bin , .net");
    return NULL;
  }
}

void SaveOutputData(char *final_name, unsigned char *output_data, unsigned int out_size) {
  FILE *output_file;
  output_file = fopen(final_name, "w");
  if (output_file == NULL) {
    printf("fopen output file: %s failed\n", final_name);
    return;
  }
  unsigned char str[out_size];
  for (unsigned int i = 0; i < out_size; ++i) {
    str[i] = output_data[i];
    fprintf(output_file, "%d\t", str[i]);
  }
  fclose(output_file);
}

int ReadInputsFile(char *path, void **buffers, const int *inputs_size, int inputs_num) {
  char *inputs_path[inputs_num];
  char *delim = ",";
  char *token;
  int i = 0;
  while ((token = strtok_r(path, delim, &path))) {
    if (i >= inputs_num) {
      printf("inputs num is error, need: %d\n", inputs_num);
      return kMSStatusLiteParamInvalid;
    }
    inputs_path[i] = token;
    printf("input %d: %s\n", i, inputs_path[i]);
    i++;
  }

  for (i = 0; i < inputs_num; ++i) {
    int size = 0;
    buffers[i] = ReadInputData(inputs_path[i], &size);
    if (size != inputs_size[i] || buffers[i] == NULL) {
      printf("size mismatch, %s, input: %d, needed: %d\n", inputs_path[i], size, inputs_size[i]);
      free(buffers[i]);
      return kMSStatusLiteError;
    }
  }
  return 0;
}
)RAW";

const char load_input_h_cortex[] = R"RAW(
/**
 * Copyright 2022 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MINDSPORE_LITE_MICRO_LOAD_INPUT_H_
#define MINDSPORE_LITE_MICRO_LOAD_INPUT_H_
#include "data.h"
#include "c_api/types_c.h"
#include "c_api/model_c.h"
#ifdef __cplusplus
extern "C" {
#endif

int SetDataToMSTensor(MSTensorHandleArray *inputs_handle, TensorArray *tensor_array);

#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_LITE_MICRO_LOAD_INPUT_H_
)RAW";

const char load_input_c_cortex[] = R"RAW(/**
 * Copyright 2022 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "load_input.h"
#include "data.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

//Set data pointer to MSTensor
int SetDataToMSTensor(MSTensorHandleArray *inputs_handle, TensorArray *tensor_array) {
  size_t inputs_num = inputs_handle->handle_num;
  if (inputs_num != tensor_array->tensors_size_) {
    printf("Tensor array num error, expected %d, but %d.", inputs_num, tensor_array->tensors_size_);
    return kMSStatusLiteError;
  }

  for (size_t i = 0; i < inputs_num; ++i) {
    MSTensorHandle tensor = inputs_handle->handle_list[i];
    int data_size = MSTensorGetDataSize(tensor);
    if(data_size != tensor_array->tensors_[i].data_size_){
      printf("Tensor %d data size failed, request %d, but it's %d.",
             i, data_size, tensor_array->tensors_[i].data_size_);
      return kMSStatusLiteError;
    }
    MSTensorSetData(tensor, tensor_array->tensors_[i].data_);
  }
  return kMSStatusSuccess;
}
)RAW";

const char data_h_cortex[] = R"RAW(/**
 * Copyright 2022 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MINDSPORE_LITE_MICRO_DATA_H_
#define MINDSPORE_LITE_MICRO_DATA_H_
#ifdef __cplusplus
extern "C" {
#endif
typedef struct Tensor {
  const char *tensor_name;
  int elemets_num_;
  int data_size_;
  float *data_;
} Tensor;

typedef struct TensorArray {
  Tensor *tensors_;
  int tensors_size_;
} TensorArray;

extern TensorArray g_calib_inputs;
extern TensorArray g_calib_outputs;
extern TensorArray g_inputs;
extern TensorArray g_outputs;
#ifdef __cplusplus
}
#endif
#endif //MINDSPORE_LITE_MICRO_DATA_H_
)RAW";

const char cortex_build_sh[] = R"RAW(#!/bin/bash
# Copyright 2022 Huawei Technologies Co., Ltd
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

set -e
BASEPATH=$(cd "$(dirname $0)"; pwd)
mkdir -p build

VERSION_STR=1.8.1
MINDSPORE_FILE_NAME="mindspore-lite-${VERSION_STR}-none-cortex-m7"
MINDSPORE_FILE="${MINDSPORE_FILE_NAME}.tar.gz"
MINDSPORE_LITE_DOWNLOAD_URL=\
"https://ms-release.obs.cn-north-4.myhuaweicloud.com/${VERSION_STR}/MindSpore/lite/none_cortex-m/${MINDSPORE_FILE}"
if [ ! -e ${BASEPATH}/${MINDSPORE_FILE} ]; then
  wget -c -O ${BASEPATH}/${MINDSPORE_FILE} --no-check-certificate ${MINDSPORE_LITE_DOWNLOAD_URL}
fi
if [ ! -e ${BASEPATH}/${MINDSPORE_FILE_NAME} ]; then
  tar xzf ${BASEPATH}/${MINDSPORE_FILE} -C ${BASEPATH}/
fi
cd build
cmake -DPKG_PATH=../${MINDSPORE_FILE_NAME}  -DCMAKE_TOOLCHAIN_FILE=../cortex-m7.toolchain.cmake  ..
make
cd ..
)RAW";

const char cortex_m7_toolchain[] = R"RAW(set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)

find_program(arm-none-eabi-gcc_EXE arm-none-eabi-gcc)
if(NOT arm-none-eabi-gcc_EXE)
    message(FATAL_ERROR "Required C COMPILER arm-none-eabi-gcc not found, "
            "please install the package and try building MindSpore again.")
else()
    message("Find C COMPILER PATH: ${arm-none-eabi-gcc_EXE}")
endif()

find_program(arm-none-eabi-g++_EXE arm-none-eabi-g++)
if(NOT arm-none-eabi-g++_EXE)
    message(FATAL_ERROR "Required CXX COMPILER arm-none-eabi-g++ not found, "
            "please install the package and try building MindSpore again.")
else()
    message("Find CXX COMPILER PATH: ${arm-none-eabi-g++_EXE}")
endif()

set(CMAKE_C_COMPILER arm-none-eabi-gcc)
set(CMAKE_CXX_COMPILER arm-none-eabi-g++)

set(CMAKE_CXX_FLAGS "-mcpu=cortex-m7 -mthumb -mfpu=fpv5-d16 -mfloat-abi=hard    ${CMAKE_CXX_FLAGS}")
set(CMAKE_C_FLAGS "-mcpu=cortex-m7 -mthumb -mfpu=fpv5-d16 -mfloat-abi=hard   ${CMAKE_C_FLAGS}")

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

set(CROSS_COMPILATION_ARM contex-m7)
set(CROSS_COMPILATION_ARCHITECTURE armv7-m)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}" CACHE STRING "c flags")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" CACHE STRING "c++ flags")

SET(CMAKE_C_COMPILER_WORKS TRUE)
SET(CMAKE_CXX_COMPILER_WORKS TRUE)
SET(CMAKE_TRY_COMPILE_TARGET_TYPE "STATIC_LIBRARY")
)RAW";
}  // namespace mindspore::lite::micro
