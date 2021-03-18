
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

uint64_t GetTimeUs() {
  const int USEC = 1000000;
  const int MSEC = 1000;
  struct timespec ts = {0, 0};
  if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
    return 0;
  }
  uint64_t retval = (uint64_t)((ts.tv_sec * USEC) + (ts.tv_nsec / MSEC));
  return retval;
}

