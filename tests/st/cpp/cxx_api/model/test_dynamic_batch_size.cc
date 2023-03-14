/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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
#include <map>
#include "common/common_test.h"
#include "include/api/model.h"
#include "include/api/serialization.h"
#include "include/api/context.h"

using namespace mindspore;

static const char tensor_add_file[] = "/home/workspace/mindspore_dataset/mindir/add/add.mindir";
static const float input_data_1[2][2] = {{1, 2}, {3, 4}};
static const float input_data_2[2][2] = {{2, 3}, {4, 5}};
static const float input_data_3[1] = {2};

class TestDynamicBatchSize : public ST::Common {
 public:
  TestDynamicBatchSize() {}
};

TEST_F(TestDynamicBatchSize, InferMindIR) {
#ifdef ENABLE_ACL
  auto context = ContextAutoSet();
  ASSERT_TRUE(context != nullptr);
  ASSERT_TRUE(context->MutableDeviceInfo().size() == 1);
  auto ascend310_info = context->MutableDeviceInfo()[0]->Cast<AscendDeviceInfo>();
  ASSERT_TRUE(ascend310_info != nullptr);

  std::map<int, std::vector<int>> input_shape;
  input_shape.insert(std::make_pair(0, std::vector<int>{-1, 2}));
  input_shape.insert(std::make_pair(1, std::vector<int>{-1, 2}));
  std::vector<size_t> dynamic_batch_size = {1, 2, 4, 8};
  ascend310_info->SetDynamicBatchSize(dynamic_batch_size);
  ascend310_info->SetInputShapeMap(input_shape);

  Graph graph;
  ASSERT_TRUE(Serialization::Load(tensor_add_file, ModelType::kMindIR, &graph) == kSuccess);
  Model tensor_add;
  ASSERT_TRUE(tensor_add.Build(GraphCell(graph), context) == kSuccess);

  // get model inputs
  std::vector<MSTensor> origin_inputs = tensor_add.GetInputs();
  ASSERT_EQ(origin_inputs.size() - 1, 2);

  // prepare input
  std::vector<MSTensor> outputs;
  std::vector<MSTensor> inputs;
  size_t row = sizeof(input_data_1) / sizeof(input_data_1[0]);
  size_t col = sizeof(input_data_1[0]) / sizeof(input_data_1[0][0]);
  inputs.emplace_back(origin_inputs[0].Name(), origin_inputs[0].DataType(), origin_inputs[0].Shape(), input_data_1,
                      sizeof(float) * row * col);
  inputs.emplace_back(origin_inputs[1].Name(), origin_inputs[1].DataType(), origin_inputs[1].Shape(), input_data_2,
                      sizeof(float) * row * col);
  inputs.emplace_back(origin_inputs[2].Name(), origin_inputs[2].DataType(), origin_inputs[2].Shape(), input_data_3,
                      sizeof(float) * 1);

  // infer
  ASSERT_TRUE(tensor_add.Predict(inputs, &outputs) == kSuccess);

  // assert input
  inputs = tensor_add.GetInputs();
  ASSERT_EQ(inputs.size() - 1, 2);
  auto after_input_data_1 = inputs[0].Data();
  auto after_input_data_2 = inputs[1].Data();
  const float *p = reinterpret_cast<const float *>(after_input_data_1.get());
  float input_data1[inputs[0].DataSize() / sizeof(float)] = {0};
  float input_data2[inputs[1].DataSize() / sizeof(float)] = {0};
  size_t k = 0, t = 0;
  for (size_t i = 0; i < row; i++)
    for (size_t j = 0; j < col; j++) {
      input_data1[k++] = input_data_1[i][j];
      input_data2[t++] = input_data_2[i][j];
    }
  for (size_t i = 0; i < inputs[0].DataSize() / sizeof(float); ++i) {
    ASSERT_LE(std::abs(p[i] - input_data1[i]), 1e-4);
  }
  p = reinterpret_cast<const float *>(after_input_data_2.get());
  for (size_t i = 0; i < inputs[1].DataSize() / sizeof(float); ++i) {
    ASSERT_LE(std::abs(p[i] - input_data2[i]), 1e-4);
  }

  // assert output
  for (auto &buffer : outputs) {
    auto buffer_data = buffer.Data();
    p = reinterpret_cast<const float *>(buffer_data.get());
    for (size_t i = 0; i < buffer.DataSize() / sizeof(float); ++i) {
      ASSERT_LE(std::abs(p[i] - (input_data1[i] + input_data2[i])), 1e-4);
    }
  }
#endif  // ENABLE_ACL
}
