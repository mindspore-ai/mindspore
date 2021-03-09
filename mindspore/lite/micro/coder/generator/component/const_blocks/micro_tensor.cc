/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "coder/generator/component/const_blocks/micro_tensor.h"

namespace mindspore::lite::micro {

const char *micro_tensor_h = R"RAW(
/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MSMICRO_TENSOR_H
#define MSMICRO_TENSOR_H

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>

#define MICRO_INFO(content, args...) \
  { printf("[INFO] %s|%d: " #content "\r\n", __func__, __LINE__, ##args); }
#define MICRO_ERROR(content, args...) \
  { printf("[ERROR] %s|%d: " #content "\r\n", __func__, __LINE__, ##args); }

enum STATUS {
  RET_OK = 0,
  RET_ERROR = 1,
};

enum DataType {
  DataType_DT_FLOAT = 0,
  DataType_DT_FLOAT16 = 1,
  DataType_DT_INT8 = 2,
  DataType_DT_INT32 = 3,
  DataType_DT_UINT8 = 4,
  DataType_DT_INT16 = 5,
  DataType_DT_UINT32 = 8,
  DataType_DT_INT64 = 9,
  DataType_DT_UINT16 = 10,
  DataType_DT_UNDEFINED = 16,
  DataType_MIN = DataType_DT_FLOAT,
  DataType_MAX = DataType_DT_UNDEFINED
};

enum Format {
  Format_NCHW = 0,
  Format_NHWC = 1,
  Format_HWKC = 2,
  Format_HWCK = 3,
  Format_KCHW = 4,
  Format_CKHW = 5,
  Format_KHWC = 6,
  Format_CHWK = 7,
  Format_NC4HW4 = 100,
  Format_NUM_OF_FORMAT = 101,
  Format_MIN = Format_NCHW,
  Format_MAX = Format_NUM_OF_FORMAT
};

typedef struct {
  enum DataType type;
  enum Format format;
  int ndim;
  int *dim;
  void *data;
} MicroTensor;

typedef struct {
  int num;
  MicroTensor *tensor;
} MicroTensorList;

typedef struct {
  float in_scale;
  float out_scale;
  int in_zero_point;
  int out_zero_point;
} GraphQuantArgs;

#endif  // MSMICRO_TENSOR_H

)RAW";

}  // namespace mindspore::lite::micro
