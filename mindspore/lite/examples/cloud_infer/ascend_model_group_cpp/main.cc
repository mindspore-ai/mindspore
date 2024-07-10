/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include <memory>
#include <thread>
#include <tuple>
#include <vector>
#include "include/api/model.h"
#include "include/api/context.h"
#include "include/api/status.h"
#include "include/api/types.h"
#include "include/api/model_group.h"

struct ThreadArgs {
  std::vector<std::string> model_paths;
  std::vector<mindspore::MSTensor> inputs;
  std::vector<mindspore::MSTensor> outputs;
  std::shared_ptr<mindspore::Context> context;
  std::shared_ptr<mindspore::ModelGroup> model_group;
  int index;
};

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

void ThreadFunc(ThreadArgs threadArg) {
  mindspore::Model model1;
  mindspore::Model model2;
  std::cout << threadArg.model_paths[0] << std::endl;
  std::cout << threadArg.model_paths[1] << std::endl;
  threadArg.model_group->AddModel(threadArg.model_paths);
  std::cout << "add model finished!" << std::endl;
  threadArg.model_group->CalMaxSizeOfWorkspace(mindspore::kMindIR, threadArg.context);
  std::cout << "cal maxsizeofworkspace finished!" << std::endl;
  std::cout << "threadid:" << threadArg.index << std::endl;
  auto build_ret = model1.Build(threadArg.model_paths[0], mindspore::kMindIR, threadArg.context);
  if (build_ret != mindspore::kSuccess) {
    std::cerr << "Build model error " << build_ret << std::endl;
  }
  build_ret = model2.Build(threadArg.model_paths[1], mindspore::kMindIR, threadArg.context);
  if (build_ret != mindspore::kSuccess) {
    std::cerr << "Build model error " << build_ret << std::endl;
  }
  std::cout << "build model finished!" << std::endl;
  auto inputs = model1.GetInputs();
  // Generate random data as input data.
  if (GenerateInputDataWithRandom(inputs) != 0) {
    std::cerr << "Generate Random Input Data failed." << std::endl;
  }
  std::cout << "get input finished!" << std::endl;
  model1.Predict(inputs, &(threadArg.outputs));
  std::cout << "predict model 1 finished!" << std::endl;
  model2.Predict(inputs, &(threadArg.outputs));
  std::cout << "predict model 2 finished!" << std::endl;
}

int QuickStart() {
  // Read model file.
  std::string model_path1 = "path_to_model1";
  std::string model_path2 = "path_to_model2";

  // Create and init context, add CPU device info
  auto context = std::make_shared<mindspore::Context>();
  auto &device_list = context->MutableDeviceInfo();
  auto device_info = std::make_shared<mindspore::AscendDeviceInfo>();
  if (device_info == nullptr) {
    std::cerr << "New AscendDeviceInfo failed." << std::endl;
    return -1;
  }
  device_list.push_back(device_info);
  // share weight
  // auto model_group = std::make_shared<mindspore::ModelGroup>(mindspore::ModelGroupFlag::kShareWeight);
  // share weight and workmem
  // auto model_group = std::make_shared<mindspore::ModelGroup>(mindspore::ModelGroupFlag::kShareWeightAndWorkspace);
  // share workmem
  auto model_group = std::make_shared<mindspore::ModelGroup>();
  std::vector<std::string> model_path_list = {model_path1, model_path2};
  std::vector<mindspore::MSTensor> outputs;
  std::vector<std::thread> threads;
  ThreadArgs args1, args2;
  args1.model_paths = model_path_list;
  args1.outputs = outputs;
  args1.index = 1;
  args1.context = context;

  args1.model_group = model_group;
  std::cout << "Create args1 finished" << std::endl;
  args2.model_paths = model_path_list;
  args2.outputs = outputs;
  args2.index = 2;
  args2.context = context;
  args2.model_group = model_group;
  std::cout << "Create args2 finished" << std::endl;
  threads.emplace_back(ThreadFunc, args1);
  threads.emplace_back(ThreadFunc, args2);
  for (auto &t : threads) {
    t.join();
  }
  std::cout << "finished!" << std::endl;
  return 0;
}

int main(int argc, const char **argv) { return QuickStart(); }
