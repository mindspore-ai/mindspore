/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include <vector>
#include "common/common_test.h"
#include "include/api/model.h"
#include "include/api/serialization.h"
#include "include/api/context.h"

using namespace mindspore;

static const char tensor_add_file[] = "/home/workspace/mindspore_dataset/tensor_add/tensor_add.mindir";
static const std::vector<float> input_data_1 = {1, 2, 3, 4};
static const std::vector<float> input_data_2 = {2, 3, 4, 5};
static const std::vector<float> input_data_shape5_1 = {1, 2, 3, 4, 5};
static const std::vector<float> input_data_shape5_2 = {2, 3, 4, 5, 6};

class TestTensorAdd : public ST::Common {
 public:
  TestTensorAdd() {}
};

TEST_F(TestTensorAdd, BuildOptionOutputType_Only310) {
  GlobalContext::SetGlobalDeviceTarget(kDeviceTypeAscend310);

  auto ascend310_context = std::make_shared<ModelContext>();
  ModelContext::SetOutputType(ascend310_context, DataType::kNumberTypeUInt8);
  auto graph = Serialization::LoadModel(tensor_add_file, ModelType::kMindIR);
  Model tensor_add((GraphCell(graph)), ascend310_context);
  Status ret = tensor_add.Build();
  ASSERT_TRUE(ret == kSuccess);

  // prepare input
  std::vector<MSTensor> inputs;
  std::vector<MSTensor> outputs;
  std::vector<MSTensor> origin_inputs = tensor_add.GetInputs();
  inputs.emplace_back(origin_inputs[0].Name(), origin_inputs[0].DataType(), origin_inputs[0].Shape(),
                      input_data_1.data(), sizeof(float) * input_data_1.size());
  inputs.emplace_back(origin_inputs[1].Name(), origin_inputs[1].DataType(), origin_inputs[1].Shape(),
                      input_data_2.data(), sizeof(float) * input_data_2.size());

  // infer
  ret = tensor_add.Predict(inputs, &outputs);
  ASSERT_TRUE(ret == kSuccess);

  // print
  for (auto &o : outputs) {
    ASSERT_EQ(o.DataType(), DataType::kNumberTypeUInt8);
    const uint8_t *p = reinterpret_cast<const uint8_t *>(o.MutableData());
    for (size_t i = 0; i < o.DataSize() / sizeof(uint8_t); ++i) {
      ASSERT_LE(std::abs(static_cast<float>(p[i]) - (input_data_1[i] + input_data_2[i])), 1e-4);
    }
  }
}
