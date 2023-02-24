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

#include <algorithm>
#include <random>
#include <iostream>
#include <fstream>
#include <cstring>
#include <memory>
#include "include/api/model.h"
#include "include/api/context.h"
#include "include/api/status.h"
#include "include/api/types.h"

template <typename T, typename Distribution>
void GenerateRandomData(int size, void *data, Distribution distribution) {
  std::mt19937 random_engine;
  int elements_num = size / sizeof(T);
  (void)std::generate_n(static_cast<T *>(data), elements_num,
                        [&distribution, &random_engine]() { return static_cast<T>(distribution(random_engine)); });
}

int GenerateInputDataWithRandom(std::vector<mindspore::MSTensor> inputs) {
  for (auto tensor : inputs) {
    auto input_data = tensor.MutableData();
    if (input_data == nullptr) {
      std::cerr << "MallocData for inTensor failed." << std::endl;
      return -1;
    }
    GenerateRandomData<float>(tensor.DataSize(), input_data, std::uniform_real_distribution<float>(0.1f, 1.0f));
  }
  return 0;
}

int QuickStart(int argc, const char **argv) {
  if (argc < 3) {
    std::cerr << "Model file and config file must be provided.\n";
    return -1;
  }
  // Read model file.
  std::string model_path = argv[1];
  if (model_path.empty()) {
    std::cerr << "Model path " << model_path << " is invalid.";
    return -1;
  }
  // Create and init context, add CPU device info
  auto context = std::make_shared<mindspore::Context>();
  if (context == nullptr) {
    std::cerr << "New context failed." << std::endl;
    return -1;
  }
  // return error message
  context->SetEnableParallel(true);
  if (context->GetEnableParallel()) {
    std::cerr << "GetEnableParallel failed." << std::endl;
    return -1;
  }
  // return error message
  context->SetMultiModalHW(true);
  if (context->GetMultiModalHW()) {
    std::cerr << "GetMultiModalHW failed." << std::endl;
    return -1;
  }
  auto &device_list = context->MutableDeviceInfo();

#ifdef ENABLE_ASCEND
  auto device_info = std::make_shared<mindspore::AscendDeviceInfo>();
  device_info->SetPrecisionMode("enforce_fp16");
#else
  auto device_info = std::make_shared<mindspore::GPUDeviceInfo>();
  // return error message
  device_info->SetEnableGLTexture(true);
  if (device_info->GetEnableGLTexture()) {
    std::cerr << "GetEnableGLTexture failed." << std::endl;
    return -1;
  }
  device_info->SetPrecisionMode("enforce_fp32");
#endif
  device_info->SetDeviceID(0);
  if (device_info == nullptr) {
    std::cerr << "New CPUDeviceInfo failed." << std::endl;
    return -1;
  }
  std::cout << "device_info->GetPrecisionMode(): " << device_info->GetPrecisionMode() << std::endl;
  device_list.push_back(device_info);

  mindspore::Model model;

  // config: [ascend_context]/[gpu_context] precision_mode=preferred_fp16
  auto config_ret = model.LoadConfig(argv[2]);
  if (config_ret != mindspore::kSuccess) {
    std::cerr << "LoadConfig failed while running ";
  }

  std::cout << "Before build device_info->GetPrecisionMode(): " << device_info->GetPrecisionMode() << std::endl;

  // Build model
  auto build_ret = model.Build(model_path, mindspore::kMindIR, context);
  if (build_ret != mindspore::kSuccess) {
    std::cerr << "Build model error " << build_ret << std::endl;
    return -1;
  }

  std::cout << "After build device_info->GetPrecisionMode(): " << device_info->GetPrecisionMode() << std::endl;

  // Get Input
  auto inputs = model.GetInputs();
  // Generate random data as input data.
  if (GenerateInputDataWithRandom(inputs) != 0) {
    std::cerr << "Generate Random Input Data failed." << std::endl;
    return -1;
  }

  // Model Predict
  std::vector<mindspore::MSTensor> outputs;
  auto predict_ret = model.Predict(inputs, &outputs);
  if (predict_ret != mindspore::kSuccess) {
    std::cerr << "Predict error " << predict_ret << std::endl;
    return -1;
  }

  // Print Output Tensor Data.
  constexpr int kNumPrintOfOutData = 50;
  for (auto &tensor : outputs) {
    std::cout << "tensor name is:" << tensor.Name() << " tensor size is:" << tensor.DataSize()
              << " tensor elements num is:" << tensor.ElementNum() << std::endl;
    auto out_data = reinterpret_cast<const float *>(tensor.Data().get());
    std::cout << "output data is:";
    for (int i = 0; i < tensor.ElementNum() && i <= kNumPrintOfOutData; i++) {
      std::cout << out_data[i] << " ";
    }
    std::cout << std::endl;
  }
#ifdef ENABLE_ASCEND
  if (device_info->GetPrecisionMode() == "enforce_fp32") {
    std::cout << "Right Ascend GetPrecisionMode" << std::endl;
  }
#else
  if (device_info->GetPrecisionMode() == "preferred_fp16") {
    std::cout << "Right GPU GetPrecisionMode" << std::endl;
  }
#endif
  std::cout << "Success Predict" << std::endl;
  return 0;
}

int main(int argc, const char **argv) { return QuickStart(argc, argv); }
