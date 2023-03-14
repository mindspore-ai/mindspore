/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

static const char tensor_add_file[] = "/home/workspace/mindspore_dataset/mindir/add/add.mindir";
static const std::vector<float> input_data_1 = {1, 2, 3, 4};
static const std::vector<float> input_data_2 = {2, 3, 4, 5};

class TestAdd : public ST::Common {
 public:
  TestAdd() {}
};

TEST_F(TestAdd, InferMindIR) {
  auto context = ContextAutoSet();

  Graph graph;
  ASSERT_TRUE(Serialization::Load(tensor_add_file, ModelType::kMindIR, &graph));
  Model tensor_add;
  ASSERT_TRUE(tensor_add.Build(GraphCell(graph), context) == kSuccess);

  // get model inputs
  std::vector<MSTensor> origin_inputs = tensor_add.GetInputs();
  ASSERT_EQ(origin_inputs.size(), 2);

  // prepare input
  std::vector<MSTensor> outputs;
  std::vector<MSTensor> inputs;
  inputs.emplace_back(origin_inputs[0].Name(), origin_inputs[0].DataType(), origin_inputs[0].Shape(),
                      input_data_1.data(), sizeof(float) * input_data_1.size());
  inputs.emplace_back(origin_inputs[1].Name(), origin_inputs[1].DataType(), origin_inputs[1].Shape(),
                      input_data_2.data(), sizeof(float) * input_data_2.size());

  // infer
  ASSERT_TRUE(tensor_add.Predict(inputs, &outputs) == kSuccess);

  // assert input
  inputs = tensor_add.GetInputs();
  ASSERT_EQ(inputs.size(), 2);
  auto after_input_data_1 = inputs[0].Data();
  auto after_input_data_2 = inputs[1].Data();
  const float *p = reinterpret_cast<const float *>(after_input_data_1.get());
  for (size_t i = 0; i < inputs[0].DataSize() / sizeof(float); ++i) {
    ASSERT_LE(std::abs(p[i] - input_data_1[i]), 1e-4);
  }
  p = reinterpret_cast<const float *>(after_input_data_2.get());
  for (size_t i = 0; i < inputs[0].DataSize() / sizeof(float); ++i) {
    ASSERT_LE(std::abs(p[i] - input_data_2[i]), 1e-4);
  }

  // assert output
  for (auto &buffer : outputs) {
    auto buffer_data = buffer.Data();
    p = reinterpret_cast<const float *>(buffer_data.get());
    for (size_t i = 0; i < buffer.DataSize() / sizeof(float); ++i) {
      ASSERT_LE(std::abs(p[i] - (input_data_1[i] + input_data_2[i])), 1e-4);
    }
  }
}
