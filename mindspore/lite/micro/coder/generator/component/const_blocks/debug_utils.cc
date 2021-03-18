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

#include "coder/generator/component/const_blocks/debug_utils.h"

namespace mindspore::lite::micro {

const char *debug_utils_h = R"RAW(
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

#ifndef MINDSPORE_LITE_MICRO_MICRODEBUGUTIL_H_
#define MINDSPORE_LITE_MICRO_MICRODEBUGUTIL_H_

#include <stdio.h>
#include <sys/time.h>
#include <time.h>
#include <stdint.h>

#define MICRO_INFO(content, args...) \
  { printf("[INFO] %s|%d: " #content "\r\n", __func__, __LINE__, ##args); }
#define MICRO_ERROR(content, args...) \
  { printf("[ERROR] %s|%d: " #content "\r\n", __func__, __LINE__, ##args); }

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

void PrintTensor(MicroTensor *tensor, FILE *output_file, const char *is_input);

void PrintTensorData(MicroTensor *tensor);

#endif  // MINDSPORE_LITE_MICRO_MICRODEBUGUTIL_H_

)RAW";

const char *debug_utils_c = R"RAW(
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

#include <inttypes.h>
#include "debug_utils.h"

#define UP_DIV(x, y) (((x) + (y) - (1)) / (y))

static const unsigned int kPrintNums = 20;
static const unsigned int kLineSplitNum = 44;
static const unsigned int kLineNum = 45;
unsigned int GetTensorElementSize(const MicroTensor *tensor) {
  unsigned int ans = 1;
  if (tensor->format == Format_NC4HW4) {
    for (unsigned int i = 0; i < tensor->ndim; ++i) {
      unsigned int dim = tensor->dim[i];
      if (i == 1) {
        dim = UP_DIV(dim, 4) * 4;
      }
      ans *= dim;
    }
  } else {
    for (unsigned int i = 0; i < tensor->ndim; ++i) {
      ans *= tensor->dim[i];
    }
  }
  return ans;
}

static const char *const TypeNames[] = {"DT_FLOAT", "DT_FLOAT16", "DT_INT8",   "DT_INT32", "DT_UINT8",     "DT_INT16",
                                        "",         "",           "DT_UINT32", "DT_INT64", "DT_UINT16",    "",
                                        "",         "",           "",          "",         "DT_UNDEFINED", ""};

const char *EnumNameFormat(enum Format e) {
  switch (e) {
    case Format_NCHW:
      return "NCHW";
    case Format_NHWC:
      return "NHWC";
    case Format_HWKC:
      return "HWKC";
    case Format_HWCK:
      return "HWCK";
    case Format_KCHW:
      return "KCHW";
    case Format_CKHW:
      return "CKHW";
    case Format_KHWC:
      return "KHWC";
    case Format_CHWK:
      return "CHWK";
    case Format_NC4HW4:
      return "NC4HW4";
    case Format_NUM_OF_FORMAT:
      return "NUM_OF_FORMAT";
    default:
      return "";
  }
}

void PrintTensorData(MicroTensor *tensor) {
  void *data = tensor->data;
  unsigned int elenums = GetTensorElementSize(tensor);
  if (data == NULL || elenums == 0) {
    MICRO_ERROR("print tensor data failed");
    return;
  }
  switch (tensor->type) {
    case DataType_DT_FLOAT: {
      float *addr = (float *)(data);
      for (int i = 0; i < elenums && i < kPrintNums; ++i) {
        printf("%f, ", addr[i]);
      }
      break;
    }
    case DataType_DT_INT32: {
      int32_t *addr = (int32_t *)(data);
      for (int i = 0; i < elenums && i < kPrintNums; ++i) {
        printf("%d, ", addr[i]);
      }
      break;
    }
    case DataType_DT_INT8: {
      int8_t *addr = (int8_t *)(data);
      for (int i = 0; i < elenums && i < kPrintNums; ++i) {
        printf("%d, ", addr[i]);
      }
      break;
    }
    case DataType_DT_UINT32: {
      uint32_t *addr = (uint32_t *)(data);
      for (int i = 0; i < elenums && i < kPrintNums; ++i) {
        printf("%u, ", addr[i]);
      }
      break;
    }
    case DataType_DT_UINT8: {
      uint8_t *addr = (uint8_t *)(data);
      for (int i = 0; i < elenums && i < kPrintNums; ++i) {
        printf("%u, ", addr[i]);
      }
      break;
    }
    default:
      MICRO_ERROR("unsupported data type %d", tensor->type);
  }
  printf("\n");
}

void PrintDataToFile(const void *data, const size_t elenums, const enum DataType type, FILE *file) {
  if (data == NULL || elenums == 0) {
    MICRO_ERROR("print tensor data to file failed");
    return;
  }
  switch (type) {
    case DataType_DT_FLOAT: {
      float *addr = (float *)(data);
      for (int i = 0; i < elenums; ++i) {
        fprintf(file, "%0.15f, ", addr[i]);
        if (i % kLineNum == kLineSplitNum) {
          fprintf(file, "\n");
        }
      }
      break;
    }
    case DataType_DT_INT32: {
      int32_t *addr = (int32_t *)(data);
      for (int i = 0; i < elenums; ++i) {
        fprintf(file, "%d, ", addr[i]);
        if (i % kLineNum == kLineSplitNum) {
          fprintf(file, "\n");
        }
      }
      break;
    }
    case DataType_DT_INT8: {
      int8_t *addr = (int8_t *)(data);
      for (int i = 0; i < elenums; ++i) {
        fprintf(file, "%d, ", addr[i]);
        if (i % kLineNum == kLineSplitNum) {
          fprintf(file, "\n");
        }
      }
      break;
    }
    case DataType_DT_UINT32: {
      uint32_t *addr = (uint32_t *)(data);
      for (int i = 0; i < elenums; ++i) {
        fprintf(file, "%u, ", addr[i]);
        if (i % kLineNum == kLineSplitNum) {
          fprintf(file, "\n");
        }
      }
      break;
    }
    case DataType_DT_UINT8: {
      uint8_t *addr = (uint8_t *)(data);
      for (int i = 0; i < elenums; ++i) {
        fprintf(file, "%u, ", addr[i]);
        if (i % kLineNum == kLineSplitNum) {
          fprintf(file, "\n");
        }
      }
      break;
    }
    default:
      MICRO_ERROR("unsupported data type %d", type);
  }
  fprintf(file, "\n");
}

void PrintTensor(MicroTensor *tensor, FILE *output_file, const char *is_input) {
  if (output_file == NULL) {
    MICRO_ERROR("output file is NULL");
    return;
  }
  fprintf(output_file, "%s ", is_input);
  for (int i = 0; i < tensor->ndim; ++i) {
    fprintf(output_file, "%u, ", tensor->dim[i]);
  }
  fprintf(output_file, "\n");

  const char *type = TypeNames[tensor->type];
  const char *format = EnumNameFormat(tensor->format);
  unsigned int tensorSize = GetTensorElementSize(tensor);
  fprintf(output_file, "%s type:%s, format:%s, elementSize: %u\n", is_input, type, format, tensorSize);
  fprintf(output_file, "%s Data:\n", is_input);
  PrintDataToFile(tensor->data, tensorSize, tensor->type, output_file);
  (void)fflush(output_file);
}

)RAW";

}  // namespace mindspore::lite::micro
