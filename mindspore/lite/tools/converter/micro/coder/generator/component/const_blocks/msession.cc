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

#include "coder/generator/component/const_blocks/msession.h"

namespace mindspore::lite::micro {
const char model_runtime_other_source[] = R"RAW(
void MSTensorHandleArrayDestroy(MSTensorHandleArray inputs) {
  if (inputs.handle_list == NULL) {
    return;
  }
  for (int i = 0; i < inputs.handle_num; i++) {
    MicroTensor *micro_tensor = inputs.handle_list[i];
    if (!micro_tensor) {
      continue;
    }
    if (micro_tensor->data) {
      free(micro_tensor->data);
      micro_tensor->data = NULL;
    }
    if (micro_tensor->shape) {
      free(micro_tensor->shape);
      micro_tensor->shape = NULL;
    }
    free(micro_tensor);
    micro_tensor = NULL;
  }
  free(inputs.handle_list);
  inputs.handle_list = NULL;
}

MSStatus MSModelPredict(MSModelHandle model, const MSTensorHandleArray inputs, MSTensorHandleArray *outputs,
                        const MSKernelCallBackC before, const MSKernelCallBackC after) {
  MicroModel *micro_model = (MicroModel *)model;
  if (micro_model == NULL) {
    return kMSStatusLiteNullptr;
  }
  int inputs_num = inputs.handle_num;
  const void *inputs_data_array[inputs_num];
  for (int i = 0; i < inputs_num; i++) {
    inputs_data_array[i] = ((MicroTensor *)inputs.handle_list[i])->data;
  }
  SetInputs(inputs_data_array, inputs_num);

  Inference();

  int outputs_num = outputs->handle_num;
  void *outputs_data_array[outputs_num];
  for (int i = 0; i < outputs_num; i++) {
    outputs_data_array[i] = MSTensorGetMutableData(outputs->handle_list[i]);
  }
  CopyOutputsData(outputs_data_array, outputs_num);
  return kMSStatusSuccess;
}

MSTensorHandleArray MSModelGetInputs(const MSModelHandle model) {
  MicroModel *micro_model = (MicroModel *)model;
  return micro_model->inputs;
}

MSTensorHandleArray MSModelGetOutputs(const MSModelHandle model) {
  MicroModel *micro_model = (MicroModel *)model;
  return micro_model->outputs;
}

MSTensorHandle MSModelGetInputByTensorName(const MSModelHandle model, const char *tensor_name) {
  MicroModel *micro_model = (MicroModel *)model;
  if (micro_model == NULL || micro_model->inputs.handle_list == NULL) {
    return NULL;
  }
  for (size_t i = 0; i < micro_model->inputs.handle_num; i++) {
    MicroTensor *micro_tensor = (MicroTensor *)micro_model->inputs.handle_list[i];
    if (micro_tensor == NULL) {
      return NULL;
    }
    if (strcmp(micro_tensor->name, tensor_name)) {
      return micro_tensor;
    }
  }
  return NULL;
}

MSTensorHandle MSModelGetOutputByTensorName(const MSModelHandle model, const char *tensor_name) {
  MicroModel *micro_model = (MicroModel *)model;
  if (micro_model == NULL || micro_model->outputs.handle_list == NULL) {
    return NULL;
  }
  for (size_t i = 0; i < micro_model->outputs.handle_num; i++) {
    MicroTensor *micro_tensor = (MicroTensor *)micro_model->outputs.handle_list[i];
    if (micro_tensor == NULL) {
      return NULL;
    }
    if (strcmp(micro_tensor->name, tensor_name)) {
      return micro_tensor;
    }
  }
  return NULL;
}

MSStatus MSModelResize(MSModelHandle model, const MSTensorHandleArray inputs, MSShapeInfo *shape_infos,
                       size_t shape_info_num) {
  return kMSStatusLiteNotSupport;
}

)RAW";
const char model_runtime_init_source[] = R"RAW(
typedef struct {
  void *runtime_buffer;
  MSTensorHandleArray inputs;
  MSTensorHandleArray outputs;
} MicroModel;
MSModelHandle MSModelCreate() {
  MicroModel *micro_model = (MicroModel *)malloc(sizeof(MicroModel));
  if (micro_model == NULL) {
    return NULL;
  }
  int buffer_size = GetBufferSize();
  void *runtime_buffer = malloc(buffer_size);
  if (runtime_buffer == NULL) {
    return NULL;
  }
  micro_model->runtime_buffer = runtime_buffer;
  int ret = SetBuffer(runtime_buffer);
  if (ret != kMSStatusSuccess) {
    return NULL;
  }

)RAW";
}  // namespace mindspore::lite::micro
