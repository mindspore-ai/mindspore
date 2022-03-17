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

#include "coder/generator/component/const_blocks/calib_output.h"

namespace mindspore::lite::micro {
const char *calib_header = R"RAW(
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

#ifndef MINDSPORE_LITE_MICRO_CALIB_OUTPUT_H_
#define MINDSPORE_LITE_MICRO_CALIB_OUTPUT_H_

#include "c_api/model_c.h"
#include "src/tensor.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef struct CalibTensor {
  char *tensor_name;
  int elemets_num_;
  float *data_;
} CalibTensor;
int ReadCalibData(const char *calib_data_path, CalibTensor **calib_tensots, int *calib_num);
int CompareOutputs(MSTensorHandleArray outputs, CalibTensor **calib_tensors, int calib_num);
void FreeCalibTensors(CalibTensor **calib_tensors, int calib_num);
#ifdef __cplusplus
}
#endif
#endif  // MINDSPORE_LITE_MICRO_CALIB_OUTPUT_H_
)RAW";

const char *calib_source = R"RAW(
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

#include "calib_output.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define kToleranceVal 0.0001f
#define kMaxOutput 5
#define kMaxTensorSize 400 * 400 * 4

int ReadCalibData(const char *calib_data_path, CalibTensor **calib_tensor_pointers, int *calib_num) {
  FILE *file = fopen(calib_data_path, "r");
  if (!file) {
    printf("Unable open %s", calib_data_path);
    return -1;
  }
  CalibTensor *calib_tensors = (CalibTensor *)malloc(kMaxOutput * sizeof(CalibTensor));
  // read line by line
  char line[kMaxTensorSize];
  char *p;
  int i = 0;
  int elements = 1;
  *calib_num = 0;
  while (fgets(line, kMaxTensorSize, file) != NULL) {
    if (i == 0) {
      elements = 1;
      int j = 0;
      int dims = 0;
      p = strtok(line, " ");
      calib_tensors[*calib_num].tensor_name = (char *)malloc(strlen(p));
      memcpy(calib_tensors[*calib_num].tensor_name, p, strlen(p));
      while (p != NULL) {
        if (j == 1) {
          dims = atoi(p);
        }
        if (j >= 2 && j - 2 < dims) {
          elements *= atoi(p);
          if (j - 2 == dims - 1) {
            calib_tensors[*calib_num].elemets_num_ = elements;
            break;
          }
        }
        p = strtok(NULL, " ");
        j++;
      }
      i++;
    } else {
      float *data = (float *)malloc(elements * sizeof(float));
      p = strtok(line, " ");
      int k = 0;
      while (p != NULL) {
        data[k++] = atof(p);
        p = strtok(NULL, " ");
        if (k == elements) {
          calib_tensors[*calib_num].data_ = data;
          break;
        }
      }
      i--;
      (*calib_num)++;
    }
  }
  *calib_tensor_pointers = calib_tensors;
  fclose(file);
  return 0;
}

int CompareOutputs(MSTensorHandleArray outputs, CalibTensor **calib_tensors, int calib_num) {
  if (outputs.handle_num != (size_t)calib_num) {
    printf("error, outputs and calibs size is mismatch\n");
    return -1;
  }
  float total_error = 0;
  size_t outputs_num = outputs.handle_num;
  for (size_t i = 0; i < outputs_num; ++i) {
    MicroTensor *output = (MicroTensor *)outputs.handle_list[i];
    if (!output || !output->data) {
      return -1;
    }
    CalibTensor *calib = calib_tensors[0];
    if (!calib || !calib[i].data_) {
      return -1;
    }
    if (strcmp(output->name, calib[i].tensor_name) != 0) {
      printf("warning, output tensor name is not equal to calib\n");
    }
    size_t elements = (size_t)MSTensorGetElementNum(output);
    if (elements != (size_t)calib[i].elemets_num_) {
      printf("error, output elements num is not equal to calib\n");
      return -1;
    }
    switch (output->type) {
      case kMSDataTypeNumberTypeFloat32: {
        float *float_output = (float *)output->data;
        for (size_t j = 0; j < elements; ++j) {
          if (isnan(float_output[j]) || isinf(float_output[j]) || isnan(calib[i].data_[j]) ||
              isinf(calib[i].data_[j])) {
            printf("error, output data is nan or inf\n");
            return -1;
          }
          total_error += fabsf(float_output[j] - calib[i].data_[j]);
        }
        break;
      }
      case kMSDataTypeNumberTypeInt8: {
        int8_t *int_output = (int8_t *)output->data;
        for (size_t j = 0; j < elements; ++j) {
          total_error += fabsf(int_output[j] - calib[i].data_[j]);
        }
        break;
      }
      case kMSDataTypeNumberTypeUInt8: {
        uint8_t *int_output = (uint8_t *)output->data;
        for (size_t j = 0; j < elements; ++j) {
          total_error += fabsf(int_output[j] - calib[i].data_[j]);
        }
        break;
      }
      case kMSDataTypeNumberTypeInt32:
      case kMSDataTypeNumberTypeUInt32: {
        int32_t *int_output = (int32_t *)output->data;
        for (size_t j = 0; j < elements; ++j) {
          total_error += fabsf(int_output[j] - calib[i].data_[j]);
        }
        break;
      }
      default: {
        printf("unsupported tensor data type\n");
      }
    }
  }
  if (total_error > kToleranceVal) {
    printf("compare outputs failed, total error: %f\n", total_error);
    return -1;
  }
  printf("compare outputs success, total error: %f\n", total_error);
  return 0;
}

void FreeCalibTensors(CalibTensor **calib_tensors_pointers, int calib_num) {
  CalibTensor *calib_tensors = *calib_tensors_pointers;
  if (calib_tensors) {
    for (int i = 0; i < calib_num; i++) {
      if (calib_tensors[i].data_) {
        free(calib_tensors[i].data_);
        calib_tensors[i].data_ = NULL;
      }
      if (calib_tensors[i].tensor_name) {
        free(calib_tensors[i].tensor_name);
        calib_tensors[i].tensor_name = NULL;
      }
    }
    free(calib_tensors);
    calib_tensors = NULL;
  }
}
)RAW";
}  // namespace mindspore::lite::micro
