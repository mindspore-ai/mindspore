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
#include "include/api/model_parallel_runner.h"
#include <memory>
#include "common/common_test.h"
#include "src/common/file_utils.h"

namespace mindspore {
namespace {
const char in_data_path[] = "./mobilenetv2.ms.bin";
const char model_path[] = "./mobilenetv2.ms";
const size_t kInputDataSize = 1 * 224 * 224 * 3 * sizeof(float);
const size_t kOutputDataSize = 1 * 1001 * sizeof(float);

void SetInputTensorData(std::vector<MSTensor> *inputs) {
  ASSERT_EQ(inputs->size(), 1);
  auto &input = inputs->front();
  auto data_size = input.DataSize();
  ASSERT_EQ(data_size, kInputDataSize);
  size_t size;
  auto bin_buf = lite::ReadFile(in_data_path, &size);
  ASSERT_NE(bin_buf, nullptr);
  ASSERT_EQ(size, kInputDataSize);
  input.SetData(bin_buf);
  return;
}
}  // namespace

class ModelParallelRunnerTest : public mindspore::CommonTest {
 public:
  ModelParallelRunnerTest() {}
};

TEST_F(ModelParallelRunnerTest, InitWithoutRunnerConfig) {
  ModelParallelRunner runner;
  auto status = runner.Init(model_path);
  ASSERT_EQ(status, kSuccess);
}

TEST_F(ModelParallelRunnerTest, RunnerConfigWithWorkNum) {
  auto config = std::make_shared<RunnerConfig>();
  ASSERT_NE(nullptr, config);

  config->SetWorkersNum(2);
  ModelParallelRunner runner;
  auto status = runner.Init(model_path, config);
  ASSERT_EQ(status, kSuccess);
}

TEST_F(ModelParallelRunnerTest, RunnerConfigWithContext) {
  auto config = std::make_shared<RunnerConfig>();
  ASSERT_NE(nullptr, config);

  auto context = std::make_shared<Context>();
  ASSERT_NE(nullptr, context);
  context->SetThreadNum(1);
  auto &device_list = context->MutableDeviceInfo();
  auto device_info = std::make_shared<mindspore::CPUDeviceInfo>();
  ASSERT_NE(nullptr, device_info);
  device_list.push_back(device_info);
  ASSERT_EQ(device_list.size(), 1);

  config->SetContext(context);
  config->SetWorkersNum(2);
  ModelParallelRunner runner;
  auto status = runner.Init(model_path, config);
  ASSERT_EQ(status, kSuccess);
}

TEST_F(ModelParallelRunnerTest, RunnerGetInput) {
  auto config = std::make_shared<RunnerConfig>();
  ASSERT_NE(nullptr, config);

  auto context = std::make_shared<Context>();
  ASSERT_NE(nullptr, context);
  context->SetThreadNum(1);
  auto &device_list = context->MutableDeviceInfo();
  auto device_info = std::make_shared<mindspore::CPUDeviceInfo>();
  ASSERT_NE(nullptr, device_info);
  device_list.push_back(device_info);
  ASSERT_EQ(device_list.size(), 1);

  config->SetContext(context);
  config->SetWorkersNum(2);
  ModelParallelRunner runner;
  auto status = runner.Init(model_path, config);
  ASSERT_EQ(status, kSuccess);
  auto inputs = runner.GetInputs();
  ASSERT_EQ(inputs.size(), 1);
}

TEST_F(ModelParallelRunnerTest, RunnerGetOutput) {
  auto config = std::make_shared<RunnerConfig>();
  ASSERT_NE(nullptr, config);

  auto context = std::make_shared<Context>();
  ASSERT_NE(nullptr, context);
  context->SetThreadNum(1);
  auto &device_list = context->MutableDeviceInfo();
  auto device_info = std::make_shared<mindspore::CPUDeviceInfo>();
  ASSERT_NE(nullptr, device_info);
  device_list.push_back(device_info);
  ASSERT_EQ(device_list.size(), 1);

  config->SetContext(context);
  config->SetWorkersNum(2);
  ModelParallelRunner runner;
  auto status = runner.Init(model_path, config);
  ASSERT_EQ(status, kSuccess);
  auto outputs = runner.GetOutputs();
  ASSERT_EQ(outputs.size(), 1);
}

TEST_F(ModelParallelRunnerTest, PredictWithoutInput) {
  auto config = std::make_shared<RunnerConfig>();
  ASSERT_NE(nullptr, config);

  auto context = std::make_shared<Context>();
  ASSERT_NE(nullptr, context);
  context->SetThreadNum(2);
  auto &device_list = context->MutableDeviceInfo();
  auto device_info = std::make_shared<mindspore::CPUDeviceInfo>();
  ASSERT_NE(nullptr, device_info);
  device_list.push_back(device_info);
  ASSERT_EQ(device_list.size(), 1);

  config->SetContext(context);
  config->SetWorkersNum(2);
  ModelParallelRunner runner;
  auto status = runner.Init(model_path, config);
  ASSERT_EQ(status, kSuccess);
  std::vector<MSTensor> inputs;
  std::vector<MSTensor> outputs;
  status = runner.Predict(inputs, &outputs);
  ASSERT_NE(status, kSuccess);
  status = runner.Predict(inputs, &outputs);
  ASSERT_NE(status, kSuccess);
}

TEST_F(ModelParallelRunnerTest, RunnerPredict) {
  auto config = std::make_shared<RunnerConfig>();
  ASSERT_NE(nullptr, config);

  auto context = std::make_shared<Context>();
  ASSERT_NE(nullptr, context);
  auto &device_list = context->MutableDeviceInfo();
  auto device_info = std::make_shared<mindspore::CPUDeviceInfo>();
  ASSERT_NE(nullptr, device_info);
  device_list.push_back(device_info);
  ASSERT_EQ(device_list.size(), 1);

  config->SetContext(context);
  config->SetWorkersNum(2);
  ModelParallelRunner runner;
  auto status = runner.Init(model_path, config);
  ASSERT_EQ(status, kSuccess);

  auto inputs = runner.GetInputs();
  SetInputTensorData(&inputs);
  std::vector<MSTensor> outputs;
  status = runner.Predict(inputs, &outputs);
  ASSERT_EQ(status, kSuccess);
  // free user data
  for (auto &tensor : inputs) {
    char *data = static_cast<char *>(tensor.MutableData());
    delete[] data;
    tensor.SetData(nullptr);
  }

  inputs = runner.GetInputs();
  SetInputTensorData(&inputs);
  outputs.clear();
  status = runner.Predict(inputs, &outputs);
  ASSERT_EQ(status, kSuccess);
  // free user data
  for (auto &tensor : inputs) {
    char *data = static_cast<char *>(tensor.MutableData());
    delete[] data;
    tensor.SetData(nullptr);
  }
}

TEST_F(ModelParallelRunnerTest, RunnerInitByBuf) {
  auto config = std::make_shared<RunnerConfig>();
  ASSERT_NE(nullptr, config);

  auto context = std::make_shared<Context>();
  ASSERT_NE(nullptr, context);
  auto &device_list = context->MutableDeviceInfo();
  auto device_info = std::make_shared<mindspore::CPUDeviceInfo>();
  ASSERT_NE(nullptr, device_info);
  device_list.push_back(device_info);
  ASSERT_EQ(device_list.size(), 1);
  config->SetContext(context);
  config->SetWorkersNum(2);
  ModelParallelRunner runner;

  size_t size = 0;
  auto model_buf = lite::ReadFile(model_path, &size);
  ASSERT_NE(nullptr, model_buf);
  auto status = runner.Init(model_buf, size, config);
  delete[] model_buf;  // after init, users can release buf data
  ASSERT_EQ(status, kSuccess);
  auto inputs = runner.GetInputs();
  SetInputTensorData(&inputs);
  std::vector<MSTensor> outputs;
  for (auto &tensor : outputs) {
    auto tensor_size = tensor.DataSize();
    ASSERT_NE(0, tensor_size);
    ASSERT_EQ(tensor_size, kOutputDataSize);
    auto data = malloc(tensor_size);
    ASSERT_NE(nullptr, data);
    tensor.SetShape({1, 1001});
    tensor.SetData(data);
  }
  status = runner.Predict(inputs, &outputs);
  ASSERT_EQ(status, kSuccess);
  // free user data
  for (auto &tensor : inputs) {
    char *input_data = static_cast<char *>(tensor.MutableData());
    delete[] input_data;
    tensor.SetData(nullptr);
  }
  for (auto &tensor : outputs) {
    auto *output_data = tensor.MutableData();
    free(output_data);
    tensor.SetData(nullptr);
  }
}
}  // namespace mindspore
