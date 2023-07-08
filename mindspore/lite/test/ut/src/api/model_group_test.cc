/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "include/api/model_group.h"
#include <memory>
#include <string>
#include <vector>
#include "src/common/file_utils.h"

namespace mindspore {
namespace {
const char in_data_path[] = "./mobilenetv2.ms.bin";
const char model_path_1[] = "./mobilenetv2.ms";
const char model_path_2[] = "./mobilenetv2.ms";
const size_t kInputDataSize = 1 * 224 * 224 * 3 * sizeof(float);
const size_t kOutputDataSize = 1 * 1001 * sizeof(float);
const size_t kNumber2 = 2;

void SetInputTensorData(std::vector<MSTensor> *inputs) {
  ASSERT_NE(inputs, nullptr);

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

class ModelGroupTest : public mindspore::CommonTest {
 public:
  ModelGroupTest() {}
};

TEST_F(ModelGroupTest, ModelGroupAddModel) {
  // create model group
  auto model_group = std::make_shared<ModelGroup>();
  ASSERT_NE(nullptr, model_group);

  // add model for model group
  std::vector<std::string> model_paths;
  model_paths.push_back(model_path_1);
  model_paths.push_back(model_path_2);
  model_group->AddModel(model_paths);
  ASSERT_EQ(model_paths.size(), kNumber2);
}

TEST_F(ModelGroupTest, ModelGroupTestApi) {
  // create model group
  auto model_group = std::make_shared<ModelGroup>();
  ASSERT_NE(nullptr, model_group);

  // add model for model group
  std::vector<std::string> model_paths;
  model_paths.push_back(model_path_1);
  model_paths.push_back(model_path_2);
  model_group->AddModel(model_paths);
  ASSERT_EQ(model_paths.size(), kNumber2);

  // init model group
  auto context = std::make_shared<Context>();
  context->SetThreadNum(1);
  ASSERT_NE(nullptr, context);
  // add ascend context device info
  std::shared_ptr<AscendDeviceInfo> ascend_device_info = std::make_shared<AscendDeviceInfo>();
  ASSERT_NE(nullptr, ascend_device_info);
  auto &device_list = context->MutableDeviceInfo();
  ASSERT_NE(nullptr, device_list);
  device_list.push_back(ascend_device_info);
  auto status = model_group->CalMaxSizeOfWorkspace(ModelType::kMindIR, context);
  ASSERT_EQ(status, kSuccess);
}

TEST_F(ModelGroupTest, ModelGroupTestAPIERROR) {
  // create model group
  auto model_group = std::make_shared<ModelGroup>();
  ASSERT_NE(nullptr, model_group);

  // add model for model group
  std::vector<std::string> model_paths;
  model_paths.push_back(model_path_1);
  model_paths.push_back(model_path_2);
  model_group->AddModel(model_paths);
  ASSERT_EQ(model_paths.size(), kNumber2);

  // init model group
  auto context = std::make_shared<Context>();
  context->SetThreadNum(1);
  ASSERT_NE(nullptr, context);
  // add ascend context device info
  std::shared_ptr<AscendDeviceInfo> ascend_device_info = std::make_shared<AscendDeviceInfo>();
  ASSERT_NE(nullptr, ascend_device_info);
  auto &device_list = context->MutableDeviceInfo();
  ASSERT_NE(nullptr, device_list);
  device_list.push_back(ascend_device_info);
  // error ut
  auto status = model_group->CalMaxSizeOfWorkspace(ModelType::kMindIR_Lite, context);
  ASSERT_NE(status, kSuccess);
}

TEST_F(ModelGroupTest, ModelGroupInit) {
  // create model group
  auto model_group = std::make_shared<ModelGroup>();
  ASSERT_NE(nullptr, model_group);
  // add model for model group
  std::vector<std::string> model_paths;
  model_paths.push_back(model_path_1);
  model_paths.push_back(model_path_2);
  model_group->AddModel(model_paths);
  // init model group
  auto context = std::make_shared<Context>();
  ASSERT_NE(nullptr, context);
  std::shared_ptr<AscendDeviceInfo> ascend_device_info = std::make_shared<AscendDeviceInfo>();
  ASSERT_NE(nullptr, ascend_device_info);
  auto &device_list = context->MutableDeviceInfo();
  ASSERT_NE(nullptr, device_list);
  device_list.push_back(ascend_device_info);
  auto status = model_group->CalMaxSizeOfWorkspace(ModelType::kMindIR, context);
  ASSERT_EQ(status, kSuccess);

  // new a model for predict.
  auto model1 = std::make_shared<Model>();
  ASSERT_NE(nullptr, model1);
  auto model_context = std::make_shared<Context>();
  ASSERT_NE(nullptr, model_context);
  std::shared_ptr<AscendDeviceInfo> model_ascend_device_info = std::make_shared<AscendDeviceInfo>();
  ASSERT_NE(nullptr, model_ascend_device_info);
  auto &model_device_list = context->MutableDeviceInfo();
  ASSERT_NE(nullptr, model_device_list);
  model_device_list.push_back(model_ascend_device_info);
  status = model1->Build(model_path_1, ModelType::kMindIR, model_context);
  ASSERT_EQ(status, kSuccess);

  // new model 2, model 1 and model 2 shared workPtr.
  auto model_2 = std::make_shared<Model>();
  ASSERT_NE(nullptr, model_2);
  auto model_context_2 = std::make_shared<Context>();
  ASSERT_NE(nullptr, model_context_2);

  std::shared_ptr<AscendDeviceInfo> model_ascend_device_info_2 = std::make_shared<AscendDeviceInfo>();
  ASSERT_NE(nullptr, model_ascend_device_info_2);
  auto &model_device_list_2 = context->MutableDeviceInfo();
  ASSERT_NE(nullptr, model_device_list_2);
  model_device_list_2.push_back(model_ascend_device_info_2);
  status = model2->Build(model_path_2, ModelType::kMindIR, model_context_2);
  ASSERT_EQ(status, kSuccess);
}

TEST_F(ModelGroupTest, ModelGroupPredict) {
  // create model group
  auto model_group = std::make_shared<ModelGroup>();
  ASSERT_NE(nullptr, model_group);

  // add model for model group
  std::vector<std::string> model_paths;
  model_paths.push_back(model_path_1);
  model_paths.push_back(model_path_2);
  model_group->AddModel(model_paths);

  // init model group
  auto context = std::make_shared<Context>();
  ASSERT_NE(nullptr, context);
  std::shared_ptr<AscendDeviceInfo> ascend_device_info = std::make_shared<AscendDeviceInfo>();
  ASSERT_NE(nullptr, ascend_device_info);
  auto &device_list = context->MutableDeviceInfo();
  ASSERT_NE(nullptr, device_list);
  device_list.push_back(ascend_device_info);
  auto status = model_group->CalMaxSizeOfWorkspace(ModelType::kMindIR, context);
  ASSERT_EQ(status, kSuccess);

  // new model
  auto model1 = std::make_shared<Model>();
  ASSERT_NE(nullptr, model1);
  auto model_context = std::make_shared<Context>();
  ASSERT_NE(nullptr, model_context);
  std::shared_ptr<AscendDeviceInfo> model_ascend_device_info = std::make_shared<AscendDeviceInfo>();
  ASSERT_NE(nullptr, model_ascend_device_info);
  auto &model_device_list = context->MutableDeviceInfo();
  ASSERT_NE(nullptr, model_device_list);
  model_device_list.push_back(model_ascend_device_info);
  status = model1->Build(model_path_1, ModelType::kMindIR, model_context);
  ASSERT_EQ(status, kSuccess);
  auto inputs_1 = model1->GetInputs();
  SetInputTensorData(&inputs_1);
  std::vector<MSTensor> outputs_1;
  status = model1->Predict(inputs_1, &outputs_1);
  ASSERT_EQ(status, kSuccess);

  // new model 2
  auto model_2 = std::make_shared<Model>();
  ASSERT_NE(nullptr, model_2);
  auto model_context_2 = std::make_shared<Context>();
  ASSERT_NE(nullptr, model_context_2);
  std::shared_ptr<AscendDeviceInfo> model_ascend_device_info_2 = std::make_shared<AscendDeviceInfo>();
  ASSERT_NE(nullptr, model_ascend_device_info_2);
  auto &model_device_list_2 = context->MutableDeviceInfo();
  ASSERT_NE(nullptr, model_device_list_2);
  model_device_list_2.push_back(model_ascend_device_info_2);
  status = model2->Build(model_path_2, ModelType::kMindIR, model_context_2);
  ASSERT_EQ(status, kSuccess);
  auto inputs_2 = model_2->GetInputs();
  SetInputTensorData(&inputs_2);
  std::vector<MSTensor> outputs_2;
  status = model_2->Predict(inputs_2, &outputs_2);
  ASSERT_EQ(status, kSuccess);
}
}  // namespace mindspore
