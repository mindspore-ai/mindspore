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
#include <stdlib.h>
#include <stdio.h>
#include "include/c_api/model_c.h"

int GenerateInputDataWithRandom(MSTensorHandleArray inputs) {
  for (size_t i = 0; i < inputs.handle_num; ++i) {
    float *input_data = (float *)MSTensorGetMutableData(inputs.handle_list[i]);
    if (input_data == NULL) {
      printf("MSTensorGetMutableData failed.\n");
      return kMSStatusLiteError;
    }
    int64_t num = MSTensorGetElementNum(inputs.handle_list[i]);
    const int divisor = 10;
    for (size_t j = 0; j < num; j++) {
      input_data[j] = (float)(rand() % divisor) / divisor;  // 0--0.9f
    }
  }
  return kMSStatusSuccess;
}

int QuickStart(int argc, const char **argv) {
  if (argc < 2) {
    printf("Model file must be provided.\n");
    return kMSStatusLiteError;
  }

  // Create and init context, add CPU device info
  MSContextHandle context = MSContextCreate();
  if (context == NULL) {
    printf("MSContextCreate failed.\n");
    return kMSStatusLiteError;
  }
  const int thread_num = 2;
  MSContextSetThreadNum(context, thread_num);
  MSContextSetThreadAffinityMode(context, 1);

  MSDeviceInfoHandle cpu_device_info = MSDeviceInfoCreate(kMSDeviceTypeCPU);
  if (cpu_device_info == NULL) {
    printf("MSDeviceInfoCreate failed.\n");
    MSContextDestroy(&context);
    return kMSStatusLiteError;
  }
  MSDeviceInfoSetEnableFP16(cpu_device_info, false);
  MSContextAddDeviceInfo(context, cpu_device_info);

  // Create model
  MSModelHandle model = MSModelCreate();
  if (model == NULL) {
    printf("MSModelCreate failed.\n");
    MSContextDestroy(&context);
    return kMSStatusLiteError;
  }

  // Build model
  int ret = MSModelBuildFromFile(model, argv[1], kMSModelTypeMindIR, context);
  if (ret != kMSStatusSuccess) {
    printf("MSModelBuildFromFile failed, ret: %d.\n", ret);
    MSModelDestroy(&model);
    return ret;
  }

  // Get Inputs
  MSTensorHandleArray inputs = MSModelGetInputs(model);
  if (inputs.handle_list == NULL) {
    printf("MSModelGetInputs failed, ret: %d.\n", ret);
    MSModelDestroy(&model);
    return ret;
  }

  // Generate random data as input data.
  ret = GenerateInputDataWithRandom(inputs);
  if (ret != kMSStatusSuccess) {
    printf("GenerateInputDataWithRandom failed, ret: %d.\n", ret);
    MSModelDestroy(&model);
    return ret;
  }

  // Model Predict
  MSTensorHandleArray outputs;
  ret = MSModelPredict(model, inputs, &outputs, NULL, NULL);
  if (ret != kMSStatusSuccess) {
    printf("MSModelPredict failed, ret: %d.\n", ret);
    MSModelDestroy(&model);
    return ret;
  }

  // Print Output Tensor Data.
  for (size_t i = 0; i < outputs.handle_num; ++i) {
    MSTensorHandle tensor = outputs.handle_list[i];
    int64_t element_num = MSTensorGetElementNum(tensor);
    printf("Tensor name: %s, tensor size is %ld ,elements num: %ld.\n", MSTensorGetName(tensor),
           MSTensorGetDataSize(tensor), element_num);
    const float *data = (const float *)MSTensorGetData(tensor);
    printf("output data is:\n");
    const int max_print_num = 50;
    for (int j = 0; j < element_num && j <= max_print_num; ++j) {
      printf("%f ", data[j]);
    }
    printf("\n");
  }

  // Delete model.
  MSModelDestroy(&model);
  return kMSStatusSuccess;
}

int main(int argc, const char **argv) { return QuickStart(argc, argv); }
