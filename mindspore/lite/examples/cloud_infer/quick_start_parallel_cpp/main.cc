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
#include "include/api/model_parallel_runner.h"
#include "include/api/context.h"
#include "include/api/status.h"
#include "include/api/types.h"
namespace {
constexpr int kNumPrintOfOutData = 50;
constexpr int kNumWorkers = 2;
constexpr int kElementsNum = 1001;
constexpr int64_t MAX_MALLOC_SIZE = static_cast<size_t>(2000) * 1024 * 1024;
}  // namespace

template <typename T, typename Distribution>
void GenerateRandomData(int size, void *data, Distribution distribution) {
  std::mt19937 random_engine;
  int elements_num = size / sizeof(T);
  (void)std::generate_n(static_cast<T *>(data), elements_num,
                        [&distribution, &random_engine]() { return static_cast<T>(distribution(random_engine)); });
}

int SetInputDataWithRandom(std::vector<mindspore::MSTensor> inputs) {
  if (inputs.size() != 1) {
    std::cerr << "input size must be 1.\n";
    return -1;
  }
  size_t size = inputs[0].DataSize();
  if (size == 0 || size > MAX_MALLOC_SIZE) {
    std::cerr << "malloc size is wrong" << std::endl;
    return {};
  }
  // user need malloc data for parallel predict input data;
  void *input_data = malloc(size);
  if (input_data == nullptr) {
    std::cerr << "malloc failed" << std::endl;
    return {};
  }
  GenerateRandomData<float>(size, input_data, std::uniform_real_distribution<float>(0.1f, 1.0f));
  inputs.at(0).SetData(input_data);
  return 0;
}

int QuickStart(int argc, const char **argv) {
  if (argc < 2) {
    std::cerr << "Model file must be provided.\n";
    return -1;
  }
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
  auto &device_list = context->MutableDeviceInfo();
  auto device_info = std::make_shared<mindspore::CPUDeviceInfo>();
  if (device_info == nullptr) {
    std::cerr << "New CPUDeviceInfo failed." << std::endl;
    return -1;
  }
  device_list.push_back(device_info);

  // Create model
  auto model_runner = new (std::nothrow) mindspore::ModelParallelRunner();
  if (model_runner == nullptr) {
    std::cerr << "New Model failed." << std::endl;
    return -1;
  }
  auto runner_config = std::make_shared<mindspore::RunnerConfig>();
  if (runner_config == nullptr) {
    std::cerr << "runner config is nullptr." << std::endl;
    return -1;
  }
  runner_config->SetContext(context);
  runner_config->SetWorkersNum(kNumWorkers);
  // Build model
  auto build_ret = model_runner->Init(model_path, runner_config);
  if (build_ret != mindspore::kSuccess) {
    delete model_runner;
    std::cerr << "Build model error " << build_ret << std::endl;
    return -1;
  }

  // Get Input
  auto inputs = model_runner->GetInputs();
  if (inputs.empty()) {
    delete model_runner;
    std::cerr << "model input is empty." << std::endl;
    return -1;
  }
  // set random data to input data.
  auto ret = SetInputDataWithRandom(inputs);
  if (ret != 0) {
    delete model_runner;
    std::cerr << "set input data failed." << std::endl;
    return -1;
  }
  // Get Output
  auto outputs = model_runner->GetOutputs();
  for (auto &output : outputs) {
    size_t size = kElementsNum * sizeof(float);
    if (size == 0 || size > MAX_MALLOC_SIZE) {
      std::cerr << "malloc size is wrong" << std::endl;
      return -1;
    }
    auto out_data = malloc(size);
    output.SetShape({1, kElementsNum});
    output.SetData(out_data);
  }

  // Model Predict
  auto predict_ret = model_runner->Predict(inputs, &outputs);
  if (predict_ret != mindspore::kSuccess) {
    delete model_runner;
    std::cerr << "Predict error " << predict_ret << std::endl;
    return -1;
  }

  // Print Output Tensor Data
  for (auto tensor : outputs) {
    std::cout << "tensor name is:" << tensor.Name() << " tensor size is:" << tensor.DataSize()
              << " tensor elements num is:" << tensor.ElementNum() << std::endl;
    auto out_data = reinterpret_cast<const float *>(tensor.Data().get());
    std::cout << "output data is:";
    constexpr int print_max = 50;
    for (int i = 0; i < tensor.ElementNum() && i <= print_max; i++) {
      std::cout << out_data[i] << " ";
    }
    std::cout << std::endl;
  }

  // Delete model runner.
  delete model_runner;
  return mindspore::kSuccess;
}

int main(int argc, const char **argv) { return QuickStart(argc, argv); }
