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

#include "coder/generator/component/const_blocks/calib_output.h"

namespace mindspore::lite::micro {
const char *calib_header = R"RAW(/**
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
int CompareOutputs(MSTensorHandleArray outputs, CalibTensor **calib_tensors, int calib_num,
                   float cosine_distance_threshold);
void FreeCalibTensors(CalibTensor **calib_tensors, int calib_num);

#ifdef __cplusplus
}
#endif
#endif  // MINDSPORE_LITE_MICRO_CALIB_OUTPUT_H_
)RAW";

const char *calib_source = R"RAW(/**
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

#include "calib_output.h"
#include "c_api/status_c.h"
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
    return kMSStatusLiteError;
  }
  CalibTensor *calib_tensors = (CalibTensor *)malloc(kMaxOutput * sizeof(CalibTensor));
  if(calib_tensors == NULL) {
    printf("Malloc calib tensors failed.");
    return kMSStatusLiteError;
  }
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
      char* tensor_name = (char *)malloc(strlen(p)+1);
      if(tensor_name == NULL) {
        printf("Malloc tensor name failed.");
        return kMSStatusLiteError;
      }
      (void)strcpy(tensor_name, p);
      calib_tensors[*calib_num].tensor_name = tensor_name;
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
      if(data == NULL) {
        printf("Malloc tensor data failed.");
        return kMSStatusLiteError;
      }
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
  return kMSStatusSuccess;
}

int CompareOutputs(MSTensorHandleArray outputs, CalibTensor **calib_tensors, int calib_num,
                   float cosine_distance_threshold) {
  if (outputs.handle_num != (size_t)calib_num) {
    printf("error, outputs and calibs size is mismatch\n");
    return kMSStatusLiteError;
  }
  size_t outputs_num = outputs.handle_num;
  bool is_success = true;
  for (size_t i = 0; i < outputs_num; ++i) {
    MicroTensor *output = (MicroTensor *)outputs.handle_list[i];
    if (!output || !output->data) {
      return kMSStatusLiteError;
    }
    CalibTensor *calib = calib_tensors[0];
    if (!calib || !calib[i].data_) {
      return kMSStatusLiteError;
    }
    if (strcmp(output->name, calib[i].tensor_name) != 0) {
      printf("warning, output tensor name is not equal to calib\n");
    }
    size_t elements = (size_t)MSTensorGetElementNum(output);
    if (elements != (size_t)calib[i].elemets_num_) {
      printf("error, output elements num is not equal to calib\n");
      return kMSStatusLiteError;
    }
    float cosin = 0.f, dot = 0.f, normx = 0.f, normy = 0.f;
    switch (output->type) {
      case kMSDataTypeNumberTypeFloat32: {
        float *float_output = (float *)output->data;
        for (size_t j = 0; j < elements; ++j) {
          if (isnan(float_output[j]) || isinf(float_output[j]) || isnan(calib[i].data_[j]) ||
              isinf(calib[i].data_[j])) {
            printf("error, output data is nan or inf\n");
            return kMSStatusLiteError;
          }
          dot += float_output[j] * calib[i].data_[j];
          normx += float_output[j] * float_output[j];
          normy += calib[i].data_[j] * calib[i].data_[j];
        }
        break;
      }
      case kMSDataTypeNumberTypeInt8: {
        int8_t *int_output = (int8_t *)output->data;
        for (size_t j = 0; j < elements; ++j) {
          dot += (float) (int_output[j] * calib[i].data_[j]);
          normx += (float) (int_output[j] * int_output[j]);
          normy += (float)(calib[i].data_[j] * calib[i].data_[j]);
        }
        break;
      }
      case kMSDataTypeNumberTypeUInt8: {
        uint8_t *int_output = (uint8_t *)output->data;
        for (size_t j = 0; j < elements; ++j) {
          dot += (float) (int_output[j] * calib[i].data_[j]);
          normx += (float) (int_output[j] * int_output[j]);
          normy += (float)(calib[i].data_[j] * calib[i].data_[j]);
        }
        break;
      }
      case kMSDataTypeNumberTypeInt32:
      case kMSDataTypeNumberTypeUInt32: {
        int32_t *int_output = (int32_t *)output->data;
        for (size_t j = 0; j < elements; ++j) {
          dot += (float) (int_output[j] * calib[i].data_[j]);
          normx += (float) (int_output[j] * int_output[j]);
          normy += (float)(calib[i].data_[j] * calib[i].data_[j]);
        }
        break;
      }
      default: {
        printf("unsupported tensor data type.\n");
      }
    }
    cosin = dot / (sqrt(normx) * sqrt(normy));
    if (cosin < cosine_distance_threshold) {
      printf("cos-similarity of %s is %f, less than %f.\n", output->name, cosin, cosine_distance_threshold);
      is_success = false;
    } else {
      printf("cos-similarity of %s is %f.\n", output->name, cosin);
    }
  }
  if (!is_success) {
    printf("compare outputs failed.\n");
    return kMSStatusLiteError;
  }
  printf("compare outputs success.\n");
  return kMSStatusSuccess;
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

const char *calib_header_cortex = R"RAW(/**
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

#ifndef MINDSPORE_LITE_MICRO_CALIB_OUTPUT_H_
#define MINDSPORE_LITE_MICRO_CALIB_OUTPUT_H_

#include "data.h"
#include "c_api/model_c.h"
#include "src/tensor.h"
#ifdef __cplusplus
extern "C" {
#endif

int LoadCalibInputs(MSTensorHandleArray *inputs, TensorArray *tensor_array);
int CompareOutputs(MSTensorHandleArray *outputs, TensorArray *tensor_array);

#ifdef __cplusplus
}
#endif
#endif  // MINDSPORE_LITE_MICRO_CALIB_OUTPUT_H_
)RAW";

const char *calib_source_cortex = R"RAW(/**
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

#include "calib_output.h"
#include "load_input.h"
#include "data.h"
#include "c_api/status_c.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define kToleranceVal 0.0001f

int LoadCalibInputs(MSTensorHandleArray *inputs, TensorArray *tensor_array) {
  if (inputs->handle_num != tensor_array->tensors_size_) {
    printf("error, inputs and calibs size is mismatch.\n");
    return kMSStatusLiteError;
  }
  Tensor *calib_tensors = tensor_array->tensors_;
  if (calib_tensors == NULL) {
    printf("error, calib tensor is NULL.\n");
    return kMSStatusLiteError;
  }
  size_t inputs_num = inputs->handle_num;
  for (size_t i = 0; i < inputs_num; ++i) {
    MicroTensor *input = (MicroTensor *)inputs->handle_list[i];
    if (input == NULL){
      printf("error, input is nullptr.\n");
      return kMSStatusLiteError;
    }
    if(MSTensorGetMutableData(input) == NULL) {
      printf("error, malloc input data failed.\n");
      return kMSStatusLiteError;
    }
    if (calib_tensors[i].data_ == NULL) {
      return kMSStatusLiteError;
    }
    if (strcmp(input->name, calib_tensors[i].tensor_name) != 0) {
      printf("warning, input tensor name is not equal to calib_tensors\n");
    }
    size_t elements = (size_t)MSTensorGetElementNum(input);
    if (elements != (size_t)calib_tensors[i].elemets_num_) {
      printf("error, input elements num is not equal to calib_tensors\n");
      return kMSStatusLiteError;
    }
    if (calib_tensors[i].data_ == NULL) {
      printf("error, calib data is NULL.\n");
      return kMSStatusLiteError;
    }
    switch (input->type) {
      case kMSDataTypeNumberTypeFloat32: {
        float *float_input = (float *)MSTensorGetMutableData(input);;
        for (size_t j = 0; j < elements; ++j) {
          float_input[j] = calib_tensors[i].data_[j];
        }
        break;
      }
      case kMSDataTypeNumberTypeInt8: {
        int8_t *int_input = (int8_t *)MSTensorGetMutableData(input);;
        for (size_t j = 0; j < elements; ++j) {
          int_input[j] = calib_tensors[i].data_[j];
        }
        break;
      }
      case kMSDataTypeNumberTypeUInt8: {
        uint8_t *int_input = (uint8_t *)MSTensorGetMutableData(input);;
        for (size_t j = 0; j < elements; ++j) {
          int_input[j] = calib_tensors[i].data_[j];
        }
        break;
      }
      case kMSDataTypeNumberTypeInt32:
      case kMSDataTypeNumberTypeUInt32: {
        int32_t *int_input = (int32_t *)MSTensorGetMutableData(input);;
        for (size_t j = 0; j < elements; ++j) {
          int_input[j] = calib_tensors[i].data_[j];
        }
        break;
      }
      default: {
        printf("unsupported tensor data type\n");
      }
    }
  }

  printf("Successfully write the verification data into the input.\n");
  return kMSStatusSuccess;
}

int CompareOutputs(MSTensorHandleArray *outputs, TensorArray *tensor_array) {
  if (outputs->handle_num != tensor_array->tensors_size_) {
    printf("error, outputs and calibs size is mismatch\n");
    return kMSStatusLiteError;
  }
  Tensor *calib_tensors = tensor_array->tensors_;
  if (calib_tensors == NULL) {
    printf("error, calib tensor is NULL.\n");
    return kMSStatusLiteError;
  }
  bool is_success = true;
  size_t outputs_num = outputs->handle_num;
  for (size_t i = 0; i < outputs_num; ++i) {
    MicroTensor *output = (MicroTensor *)outputs->handle_list[i];
    float cosin = 0.f, dot = 0.f, normx = 0.f, normy = 0.f;
    if (output == NULL){
      printf("error, output is nullptr.\n");
      return kMSStatusLiteError;
    }
    if(output->data == NULL) {
      printf("error, output data is nullptr.\n");
      return kMSStatusLiteError;
    }
    if (!calib_tensors[i].data_) {
      return kMSStatusLiteError;
    }
    if (strcmp(output->name, calib_tensors[i].tensor_name) != 0) {
      printf("warning, output tensor name is not equal to calib_tensors\n");
    }
    size_t elements = (size_t)MSTensorGetElementNum(output);
    if (elements != (size_t)calib_tensors[i].elemets_num_) {
      printf("error, output elements num is not equal to calib_tensors\n");
      return kMSStatusLiteError;
    }
    if (calib_tensors[i].data_ == NULL) {
      printf("error, calib data is NULL.\n");
      return kMSStatusLiteError;
    }
    switch (output->type) {
      case kMSDataTypeNumberTypeFloat32: {
        float *float_output = (float *)output->data;
        for (size_t j = 0; j < elements; ++j) {
          if (isnan(float_output[j]) || isinf(float_output[j]) || isnan(calib_tensors[i].data_[j]) ||
              isinf(calib_tensors[i].data_[j])) {
            printf("error, output data is nan or inf\n");
            return kMSStatusLiteError;
          }
          dot += float_output[j] * calib_tensors[i].data_[j];
          normx += float_output[j] * float_output[j];
          normy += calib_tensors[i].data_[j] * calib_tensors[i].data_[j];
        }
        break;
      }
      case kMSDataTypeNumberTypeInt8: {
        int8_t *int_output = (int8_t *)output->data;
        for (size_t j = 0; j < elements; ++j) {
          dot += (float) (int_output[j] * calib_tensors[i].data_[j]);
          normx += (float) (int_output[j] * int_output[j]);
          normy += (float)(calib_tensors[i].data_[j] * calib_tensors[i].data_[j]);
        }
        break;
      }
      case kMSDataTypeNumberTypeUInt8: {
        uint8_t *int_output = (uint8_t *)output->data;
        for (size_t j = 0; j < elements; ++j) {
          dot += (float) (int_output[j] * calib_tensors[i].data_[j]);
          normx += (float) (int_output[j] * int_output[j]);
          normy += (float)(calib_tensors[i].data_[j] * calib_tensors[i].data_[j]);
        }
        break;
      }
      case kMSDataTypeNumberTypeInt32:
      case kMSDataTypeNumberTypeUInt32: {
        int32_t *int_output = (int32_t *)output->data;
        for (size_t j = 0; j < elements; ++j) {
          dot += (float) (int_output[j] * calib_tensors[i].data_[j]);
          normx += (float) (int_output[j] * int_output[j]);
          normy += (float)(calib_tensors[i].data_[j] * calib_tensors[i].data_[j]);
        }
        break;
      }
      default: {
        printf("unsupported tensor data type\n");
      }
    }
    cosin = dot / (sqrt(normx) * sqrt(normy));
    if (cosin < 0.9999) {
      printf("cos-similarity of %s is %f, less than 0.9999.\n", output->name, cosin);
      is_success = false;
    } else {
      printf("cos-similarity of %s is %f.\n", output->name, cosin);
    }
  }
  if (!is_success) {
    printf("compare outputs failed.\n");
    return kMSStatusLiteError;
  }
  printf("compare outputs success.\n");
  return kMSStatusSuccess;
}
)RAW";
}  // namespace mindspore::lite::micro
