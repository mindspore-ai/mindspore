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
#include "common/common_test.h"
#include "include/api/model.h"
#include "include/api/status.h"

namespace mindspore {
class RuntimeConvert : public mindspore::CommonTest {
 public:
  RuntimeConvert() = default;
};

TEST_F(RuntimeConvert, relu1) {
  Model model;
  auto context = std::make_shared<mindspore::Context>();
  context->MutableDeviceInfo().push_back(std::make_shared<mindspore::CPUDeviceInfo>());
  Status build_ret = model.Build("./relu.mindir", mindspore::kMindIR, context);
  ASSERT_EQ(build_ret, Status::OK());

  auto inputs = model.GetInputs();
  auto in = inputs[0];
  std::vector<float> in_float = {1.0, 2.0, -3.0, -4.0};
  memcpy(inputs[0].MutableData(), in_float.data(), in.DataSize());
  auto outputs = model.GetOutputs();

  auto predict_ret = model.Predict(inputs, &outputs);
  ASSERT_EQ(predict_ret, Status::OK());

  /* checkout output */
  auto out = outputs[0];
  void *out_data = out.MutableData();
  float *fp32_data = reinterpret_cast<float *>(out_data);
  ASSERT_LE(fp32_data[0], 1.0);
  ASSERT_LE(fp32_data[1], 2.0);
  ASSERT_LE(fp32_data[2], 3.0);
  ASSERT_LE(fp32_data[3], 4.0);
}

TEST_F(RuntimeConvert, relu2) {
  Model model;
  auto context = std::make_shared<mindspore::Context>();
  context->MutableDeviceInfo().push_back(std::make_shared<mindspore::CPUDeviceInfo>());
  Status build_ret = model.Build("./relu.mindir", mindspore::kMindIR_Lite, context);
  ASSERT_NE(build_ret, Status::OK());
}

TEST_F(RuntimeConvert, relu3) {
  size_t size;
  char *mindir_buf = lite::ReadFile("./relu.mindir", &size);
  ASSERT_NE(mindir_buf, nullptr);

  Model model;
  auto context = std::make_shared<mindspore::Context>();
  context->MutableDeviceInfo().push_back(std::make_shared<mindspore::CPUDeviceInfo>());
  Status build_ret = model.Build(mindir_buf, size, mindspore::kMindIR, context);
  ASSERT_EQ(build_ret, Status::OK());

  auto inputs = model.GetInputs();
  auto in = inputs[0];
  std::vector<float> in_float = {1.0, 2.0, -3.0, -4.0};
  memcpy(inputs[0].MutableData(), in_float.data(), in.DataSize());
  auto outputs = model.GetOutputs();

  auto predict_ret = model.Predict(inputs, &outputs);
  ASSERT_EQ(predict_ret, Status::OK());

  /* checkout output */
  auto out = outputs[0];
  void *out_data = out.MutableData();
  float *fp32_data = reinterpret_cast<float *>(out_data);
  ASSERT_LE(fp32_data[0], 1.0);
  ASSERT_LE(fp32_data[1], 2.0);
  ASSERT_LE(fp32_data[2], 3.0);
  ASSERT_LE(fp32_data[3], 4.0);
}
}  // namespace mindspore
