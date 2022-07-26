/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include <string>
#include <iostream>

#include "include/api/model.h"
#include "src/extendrt/utils/cloud_infer_test.h"

const int test_input_shape = 7;
const int test_input_value = 3;

int main(int argc, const char **argv) {
  CloudInferTestFlags flags;
  mindspore::lite::Option<std::string> err = flags.ParseFlags(argc, argv);
  if (err.IsSome()) {
    std::cerr << err.Get() << std::endl;
    std::cerr << flags.Usage() << std::endl;
    return mindspore::lite::RET_ERROR;
  }
  mindspore::Model ms_model;
  auto context = std::make_shared<mindspore::Context>();
  auto &device_info = context->MutableDeviceInfo();
  auto device_context = std::make_shared<mindspore::CPUDeviceInfo>();
  device_context->SetProvider("tensorrt");
  device_context->SetAllocator(nullptr);
  device_info.emplace_back(device_context);
  ms_model.Build(flags.model_file_, mindspore::kMindIR, context);

  auto inputs = ms_model.GetInputs();
  for (auto input : inputs) {
    auto input_data = input.MutableData();
    if (input_data == nullptr) {
      std::cout << "input data is nullptr" << std::endl;
      return -1;
    }
    if (static_cast<int>(input.DataType()) == mindspore::kNumberTypeInt32) {
      int *int_data = static_cast<int *>(input_data);
      // only test int32 data type
      for (int i = 0; i < input.ElementNum(); i++) {
        int_data[i] = test_input_value;
      }
    }
  }
  std::vector<mindspore::MSTensor> outputs;
  auto ret = ms_model.Predict(inputs, &outputs);
  if (ret != mindspore::kSuccess) {
    std::cout << "Preict failed with ret " << ret << std::endl;
    return -1;
  }

  for (auto output : outputs) {
    auto z = static_cast<bool *>(output.MutableData());
    for (int i = 0; i < test_input_shape; i++) {
      std::cout << z[i] << " ";
    }
    std::cout << std::endl;
  }

  std::cout << "hello  cloud inference!" << std::endl;
  return 0;
}
