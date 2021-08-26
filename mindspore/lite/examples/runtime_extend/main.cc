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

#include <iostream>
#include <cstring>
#include <memory>
#include <string>
#include <vector>
#include "include/api/status.h"
#include "include/api/context.h"
#include "include/api/model.h"

namespace mindspore {
namespace lite {
namespace {
constexpr int kNumPrintOfOutData = 20;
Status FillInputData(const std::vector<mindspore::MSTensor> &inputs) {
  for (auto tensor : inputs) {
    auto input_data = tensor.MutableData();
    if (input_data == nullptr) {
      std::cerr << "MallocData for inTensor failed.\n";
      return kLiteError;
    }
    std::vector<float> temp(tensor.ElementNum(), 1.0f);
    memcpy(input_data, temp.data(), tensor.DataSize());
  }
  return kSuccess;
}
}  // namespace

Status CompileAndRun(int argc, const char **argv) {
  if (argc < 2) {
    std::cerr << "Model file must be provided.\n";
    return kLiteError;
  }
  // generate context.
  auto context = std::make_shared<mindspore::Context>();
  if (context == nullptr) {
    std::cerr << "New context failed while running.\n";
    return kLiteError;
  }
  auto &device_list = context->MutableDeviceInfo();
  std::shared_ptr<CPUDeviceInfo> device_info = std::make_shared<CPUDeviceInfo>();
  device_info->SetProvider("Tutorial");
  device_info->SetProviderDevice("Tutorial");
  device_list.push_back(device_info);

  // build model.
  std::string model_file = std::string(argv[1]);
  mindspore::Model model;
  auto ret = model.Build(model_file, kMindIR, context);
  if (ret != kSuccess) {
    std::cerr << "build model failed.\n";
    return kLiteError;
  }

  // fill input data.
  auto inputs = model.GetInputs();
  ret = FillInputData(inputs);
  if (ret != kSuccess) {
    std::cerr << "Generate Random Input Data failed.\n";
    return ret;
  }

  // run model.
  std::vector<MSTensor> outputs;
  ret = model.Predict(inputs, &outputs);
  if (ret != kSuccess) {
    std::cerr << "run model failed.\n";
    return ret;
  }

  // display output result.
  for (auto tensor : outputs) {
    std::cout << "tensor name is:" << tensor.Name() << " tensor size is:" << tensor.DataSize()
              << " tensor elements num is:" << tensor.ElementNum() << std::endl;
    auto out_data = std::static_pointer_cast<const float>(tensor.Data());
    std::cout << "output data is:";
    for (int i = 0; i < tensor.ElementNum() && i <= kNumPrintOfOutData; i++) {
      std::cout << out_data.get()[i] << " ";
    }
    std::cout << std::endl;
  }
  return kSuccess;
}
}  // namespace lite
}  // namespace mindspore

int main(int argc, const char **argv) {
  auto ret = mindspore::lite::CompileAndRun(argc, argv);
  if (ret != mindspore::kSuccess) {
    std::cerr << "run failed.\n";
    return -1;
  }
  std::cout << "run success.\n";
  return 0;
}
