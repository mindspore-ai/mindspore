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
#include "common/common_test.h"
#include "include/api/model.h"
#include "include/api/serialization.h"
#include "include/api/context.h"

using namespace mindspore;

static constexpr char kIfbyIfFile[] = "/home/workspace/mindspore_dataset/mindir/control/ifbyif.mindir";
static constexpr char kSimpleWhileFile[] = "/home/workspace/mindspore_dataset/mindir/control/simple_while.mindir";
static constexpr char kMixIfWhileFile[] = "/home/workspace/mindspore_dataset/mindir/control/mix_while_if.mindir";
static constexpr char kRecursiveFile[] = "/home/workspace/mindspore_dataset/mindir/control/fibonacci.mindir";
static constexpr char kSingleForFile[] = "/home/workspace/mindspore_dataset/mindir/control/single_for.mindir";
static constexpr char kSingleOrFile[] = "/home/workspace/mindspore_dataset/mindir/control/single_or.mindir";
static constexpr char kSingleSwitchFile[] = "/home/workspace/mindspore_dataset/mindir/control/switch_layer_net.mindir";
static constexpr float kConstValue = 0.1234;
static const std::vector<float> input_data(2 * 3 * 4 * 5, kConstValue);

class TestControl : public ST::Common {
 public:
  TestControl() {}
};

TEST_F(TestControl, InferIfbyIf) {
  auto context = ContextAutoSet();

  Graph graph;
  ASSERT_TRUE(Serialization::Load(kIfbyIfFile, ModelType::kMindIR, &graph));
  Model control_model;
  ASSERT_TRUE(control_model.Build(GraphCell(graph), context) == kSuccess);

  // assert inputs
  std::vector<MSTensor> inputs_before = control_model.GetInputs();
  ASSERT_EQ(5, inputs_before.size());
  EXPECT_EQ(inputs_before[0].DataType(), DataType::kNumberTypeFloat32);
  EXPECT_EQ(inputs_before[1].DataType(), DataType::kNumberTypeFloat32);
  EXPECT_EQ(inputs_before[2].DataType(), DataType::kNumberTypeBool);
  EXPECT_EQ(inputs_before[3].DataType(), DataType::kNumberTypeBool);
  EXPECT_EQ(inputs_before[4].DataType(), DataType::kNumberTypeFloat32);
  ASSERT_EQ(inputs_before[0].DataSize(), sizeof(float));
  ASSERT_EQ(inputs_before[1].DataSize(), sizeof(float));
  ASSERT_EQ(inputs_before[2].DataSize(), sizeof(bool));
  ASSERT_EQ(inputs_before[3].DataSize(), sizeof(bool));
  ASSERT_EQ(inputs_before[4].DataSize(), sizeof(float) * input_data.size());
  ASSERT_EQ(inputs_before[0].Shape().size(), 1);
  EXPECT_EQ(inputs_before[0].Shape()[0], 1);
  ASSERT_EQ(inputs_before[1].Shape().size(), 1);
  EXPECT_EQ(inputs_before[1].Shape()[0], 1);
  ASSERT_EQ(inputs_before[2].Shape().size(), 1);
  EXPECT_EQ(inputs_before[2].Shape()[0], 1);
  ASSERT_EQ(inputs_before[3].Shape().size(), 1);
  EXPECT_EQ(inputs_before[3].Shape()[0], 1);
  ASSERT_EQ(inputs_before[4].Shape().size(), 4);
  EXPECT_EQ(inputs_before[4].Shape()[0], 2);
  EXPECT_EQ(inputs_before[4].Shape()[1], 3);
  EXPECT_EQ(inputs_before[4].Shape()[2], 4);
  EXPECT_EQ(inputs_before[4].Shape()[3], 5);

  // assert outputs
  std::vector<MSTensor> outputs_before = control_model.GetOutputs();
  ASSERT_EQ(1, outputs_before.size());
  EXPECT_EQ(outputs_before[0].DataType(), DataType::kNumberTypeFloat32);
  ASSERT_TRUE(outputs_before[0].DataSize() == sizeof(float) * input_data.size());
  ASSERT_EQ(outputs_before[0].Shape().size(), 4);
  EXPECT_EQ(outputs_before[0].Shape()[0], 2);
  EXPECT_EQ(outputs_before[0].Shape()[1], 3);
  EXPECT_EQ(outputs_before[0].Shape()[2], 4);
  EXPECT_EQ(outputs_before[0].Shape()[3], 5);

  // prepare input
  std::vector<MSTensor> outputs;
  std::vector<MSTensor> inputs;

  float x = 2.345678, y = 1.234567;
  bool cond1 = true, cond2 = false;
  inputs.emplace_back(inputs_before[0].Name(), inputs_before[0].DataType(), inputs_before[0].Shape(), &x,
                      sizeof(float));
  inputs.emplace_back(inputs_before[1].Name(), inputs_before[1].DataType(), inputs_before[1].Shape(), &y,
                      sizeof(float));
  inputs.emplace_back(inputs_before[2].Name(), inputs_before[2].DataType(), inputs_before[2].Shape(), &cond1,
                      sizeof(bool));
  inputs.emplace_back(inputs_before[3].Name(), inputs_before[3].DataType(), inputs_before[3].Shape(), &cond2,
                      sizeof(bool));
  inputs.emplace_back(inputs_before[4].Name(), inputs_before[4].DataType(), inputs_before[4].Shape(), input_data.data(),
                      sizeof(float) * input_data.size());

  // infer
  ASSERT_TRUE(control_model.Predict(inputs, &outputs) == kSuccess);

  // assert output
  ASSERT_TRUE(outputs.size() == 1);
  auto out = outputs[0];
  ASSERT_TRUE(out.DataSize() == sizeof(float) * input_data.size());
  auto out_data = out.Data();
  auto p = reinterpret_cast<const float *>(out_data.get());
  for (size_t i = 0; i < out.DataSize() / sizeof(float); ++i) {
    ASSERT_LE(std::abs(p[i] - kConstValue * 24), 1e-3);
  }
}

TEST_F(TestControl, InferSimpleWhile) {
  auto context = ContextAutoSet();

  Graph graph;
  ASSERT_TRUE(Serialization::Load(kSimpleWhileFile, ModelType::kMindIR, &graph));
  Model control_model;
  ASSERT_TRUE(control_model.Build(GraphCell(graph), context) == kSuccess);

  // assert inputs
  std::vector<MSTensor> inputs_before = control_model.GetInputs();
  ASSERT_EQ(3, inputs_before.size());
  EXPECT_EQ(inputs_before[0].DataType(), DataType::kNumberTypeBool);
  EXPECT_EQ(inputs_before[1].DataType(), DataType::kNumberTypeBool);
  EXPECT_EQ(inputs_before[2].DataType(), DataType::kNumberTypeFloat32);
  ASSERT_EQ(inputs_before[0].DataSize(), sizeof(bool));
  ASSERT_EQ(inputs_before[1].DataSize(), sizeof(bool));
  ASSERT_EQ(inputs_before[2].DataSize(), sizeof(float) * input_data.size());
  ASSERT_EQ(inputs_before[0].Shape().size(), 1);
  EXPECT_EQ(inputs_before[0].Shape()[0], 1);
  ASSERT_EQ(inputs_before[1].Shape().size(), 1);
  EXPECT_EQ(inputs_before[1].Shape()[0], 1);
  ASSERT_EQ(inputs_before[2].Shape().size(), 4);
  EXPECT_EQ(inputs_before[2].Shape()[0], 2);
  EXPECT_EQ(inputs_before[2].Shape()[1], 3);
  EXPECT_EQ(inputs_before[2].Shape()[2], 4);
  EXPECT_EQ(inputs_before[2].Shape()[3], 5);

  // assert outputs
  std::vector<MSTensor> outputs_before = control_model.GetOutputs();
  ASSERT_EQ(1, outputs_before.size());
  EXPECT_EQ(outputs_before[0].DataType(), DataType::kNumberTypeFloat32);
  ASSERT_TRUE(outputs_before[0].DataSize() == sizeof(float) * input_data.size());
  ASSERT_EQ(outputs_before[0].Shape().size(), 4);
  EXPECT_EQ(outputs_before[0].Shape()[0], 2);
  EXPECT_EQ(outputs_before[0].Shape()[1], 3);
  EXPECT_EQ(outputs_before[0].Shape()[2], 4);
  EXPECT_EQ(outputs_before[0].Shape()[3], 5);

  // prepare input
  std::vector<MSTensor> outputs;
  std::vector<MSTensor> inputs;
  {
    bool x = true, y = false;
    inputs.emplace_back(inputs_before[0].Name(), inputs_before[0].DataType(), inputs_before[0].Shape(), &x,
                        sizeof(bool));
    inputs.emplace_back(inputs_before[1].Name(), inputs_before[1].DataType(), inputs_before[1].Shape(), &y,
                        sizeof(bool));
    inputs.emplace_back(inputs_before[2].Name(), inputs_before[2].DataType(), inputs_before[2].Shape(),
                        input_data.data(), sizeof(float) * input_data.size());
  }

  // infer
  ASSERT_TRUE(control_model.Predict(inputs, &outputs) == kSuccess);

  // assert output
  ASSERT_TRUE(outputs.size() == 1);
  auto out = outputs[0];
  ASSERT_TRUE(out.DataSize() == sizeof(float) * input_data.size());
  auto out_data = out.Data();
  auto p = reinterpret_cast<const float *>(out_data.get());
  for (size_t i = 0; i < out.DataSize() / sizeof(float); ++i) {
    ASSERT_LE(std::abs(p[i] - kConstValue * 3), 1e-3);
  }
}

TEST_F(TestControl, InferRecursive) {
  auto context = ContextAutoSet();

  Graph graph;
  ASSERT_TRUE(Serialization::Load(kRecursiveFile, ModelType::kMindIR, &graph));
  Model control_model;
  ASSERT_TRUE(control_model.Build(GraphCell(graph), context) == kSuccess);

  // assert inputs
  std::vector<MSTensor> inputs_before = control_model.GetInputs();
  ASSERT_EQ(1, inputs_before.size());
  EXPECT_EQ(inputs_before[0].DataType(), DataType::kNumberTypeInt32);
  ASSERT_EQ(inputs_before[0].DataSize(), sizeof(int32_t));
  ASSERT_EQ(inputs_before[0].Shape().size(), 1);
  EXPECT_EQ(inputs_before[0].Shape()[0], 1);

  // assert outputs
  std::vector<MSTensor> outputs_before = control_model.GetOutputs();
  ASSERT_EQ(1, outputs_before.size());
  EXPECT_EQ(outputs_before[0].DataType(), DataType::kNumberTypeInt32);
  ASSERT_TRUE(outputs_before[0].DataSize() == sizeof(int32_t));
  ASSERT_EQ(outputs_before[0].Shape().size(), 1);
  EXPECT_EQ(outputs_before[0].Shape()[0], 1);


  // prepare input
  std::vector<MSTensor> outputs;
  std::vector<MSTensor> inputs;
  {
    int32_t x = 7;
    inputs.emplace_back(inputs_before[0].Name(), inputs_before[0].DataType(), inputs_before[0].Shape(), &x,
                        sizeof(int32_t));
  }

  // infer
  ASSERT_TRUE(control_model.Predict(inputs, &outputs) == kSuccess);

  // assert output
  ASSERT_TRUE(outputs.size() == 1);
  auto out = outputs[0];
  ASSERT_TRUE(out.DataSize() == sizeof(int32_t));
  auto out_data = out.Data();
  auto p = reinterpret_cast<const int32_t *>(out_data.get());
  ASSERT_EQ(*p, 21);
}

TEST_F(TestControl, InferMixedWhileIf) {
  auto context = ContextAutoSet();

  Graph graph;
  ASSERT_TRUE(Serialization::Load(kMixIfWhileFile, ModelType::kMindIR, &graph));
  Model control_model;
  ASSERT_TRUE(control_model.Build(GraphCell(graph), context) == kSuccess);

  // assert inputs
  std::vector<MSTensor> inputs_before = control_model.GetInputs();
  ASSERT_EQ(inputs_before.size(), 5);
  EXPECT_EQ(inputs_before[0].DataType(), DataType::kNumberTypeInt32);
  EXPECT_EQ(inputs_before[1].DataType(), DataType::kNumberTypeInt32);
  EXPECT_EQ(inputs_before[2].DataType(), DataType::kNumberTypeInt32);
  EXPECT_EQ(inputs_before[3].DataType(), DataType::kNumberTypeInt32);
  EXPECT_EQ(inputs_before[4].DataType(), DataType::kNumberTypeInt32);
  ASSERT_EQ(inputs_before[0].DataSize(), sizeof(int32_t));
  ASSERT_EQ(inputs_before[1].DataSize(), sizeof(int32_t));
  ASSERT_EQ(inputs_before[2].DataSize(), sizeof(int32_t));
  ASSERT_EQ(inputs_before[3].DataSize(), sizeof(int32_t));
  ASSERT_EQ(inputs_before[4].DataSize(), sizeof(int32_t));
  ASSERT_EQ(inputs_before[0].Shape().size(), 1);
  EXPECT_EQ(inputs_before[0].Shape()[0], 1);
  ASSERT_EQ(inputs_before[1].Shape().size(), 1);
  EXPECT_EQ(inputs_before[1].Shape()[0], 1);
  ASSERT_EQ(inputs_before[2].Shape().size(), 1);
  EXPECT_EQ(inputs_before[2].Shape()[0], 1);
  ASSERT_EQ(inputs_before[3].Shape().size(), 1);
  EXPECT_EQ(inputs_before[3].Shape()[0], 1);
  ASSERT_EQ(inputs_before[4].Shape().size(), 1);
  EXPECT_EQ(inputs_before[4].Shape()[0], 1);

  // assert outputs
  std::vector<MSTensor> outputs_before = control_model.GetOutputs();
  ASSERT_EQ(1, outputs_before.size());
  EXPECT_EQ(outputs_before[0].DataType(), DataType::kNumberTypeInt32);
  ASSERT_TRUE(outputs_before[0].DataSize() == sizeof(int32_t));
  ASSERT_EQ(outputs_before[0].Shape().size(), 1);
  EXPECT_EQ(outputs_before[0].Shape()[0], 1);

  // prepare input
  std::vector<MSTensor> outputs;
  std::vector<MSTensor> inputs;
  {
    int32_t x = 2, y = 14, z = 1, c2 = 14, c4 = 0;
    inputs.emplace_back(inputs_before[0].Name(), inputs_before[0].DataType(), inputs_before[0].Shape(), &x,
                        sizeof(int32_t));
    inputs.emplace_back(inputs_before[1].Name(), inputs_before[1].DataType(), inputs_before[1].Shape(), &y,
                        sizeof(int32_t));
    inputs.emplace_back(inputs_before[2].Name(), inputs_before[2].DataType(), inputs_before[2].Shape(), &z,
                        sizeof(int32_t));
    inputs.emplace_back(inputs_before[3].Name(), inputs_before[3].DataType(), inputs_before[3].Shape(), &c2,
                        sizeof(int32_t));
    inputs.emplace_back(inputs_before[4].Name(), inputs_before[4].DataType(), inputs_before[4].Shape(), &c4,
                        sizeof(int32_t));
  }

  // infer
  ASSERT_TRUE(control_model.Predict(inputs, &outputs) == kSuccess);

  // assert output
  ASSERT_TRUE(outputs.size() == 1);
  auto out = outputs[0];
  ASSERT_TRUE(out.DataSize() == sizeof(int32_t));
  auto out_data = out.Data();
  auto p = reinterpret_cast<const int32_t *>(out_data.get());
  ASSERT_EQ(*p, 350);
}

TEST_F(TestControl, InferSingleFor) {
  auto context = ContextAutoSet();

  Graph graph;
  ASSERT_TRUE(Serialization::Load(kSingleForFile, ModelType::kMindIR, &graph));
  Model control_model;
  ASSERT_TRUE(control_model.Build(GraphCell(graph), context) == kSuccess);

  // assert inputs
  std::vector<MSTensor> inputs_before = control_model.GetInputs();
  ASSERT_EQ(inputs_before.size(), 3);
  EXPECT_EQ(inputs_before[0].DataType(), DataType::kNumberTypeInt32);
  EXPECT_EQ(inputs_before[1].DataType(), DataType::kNumberTypeInt32);
  EXPECT_EQ(inputs_before[2].DataType(), DataType::kNumberTypeInt32);
  ASSERT_EQ(inputs_before[0].DataSize(), sizeof(int32_t));
  ASSERT_EQ(inputs_before[1].DataSize(), sizeof(int32_t));
  ASSERT_EQ(inputs_before[2].DataSize(), sizeof(int32_t));
  ASSERT_EQ(inputs_before[0].Shape().size(), 1);
  EXPECT_EQ(inputs_before[0].Shape()[0], 1);
  ASSERT_EQ(inputs_before[1].Shape().size(), 1);
  EXPECT_EQ(inputs_before[1].Shape()[0], 1);
  ASSERT_EQ(inputs_before[2].Shape().size(), 1);
  EXPECT_EQ(inputs_before[2].Shape()[0], 1);

  // assert outputs
  std::vector<MSTensor> outputs_before = control_model.GetOutputs();
  ASSERT_EQ(1, outputs_before.size());
  EXPECT_EQ(outputs_before[0].DataType(), DataType::kNumberTypeInt32);
  ASSERT_TRUE(outputs_before[0].DataSize() == sizeof(int32_t));
  ASSERT_EQ(outputs_before[0].Shape().size(), 1);
  EXPECT_EQ(outputs_before[0].Shape()[0], 1);

  // prepare input
  std::vector<MSTensor> outputs;
  std::vector<MSTensor> inputs;
  {
    int32_t x = 2, y = 5, z = 4;
    inputs.emplace_back(inputs_before[0].Name(), inputs_before[0].DataType(), inputs_before[0].Shape(), &x,
                        sizeof(int32_t));
    inputs.emplace_back(inputs_before[1].Name(), inputs_before[1].DataType(), inputs_before[1].Shape(), &y,
                        sizeof(int32_t));
    inputs.emplace_back(inputs_before[2].Name(), inputs_before[2].DataType(), inputs_before[2].Shape(), &z,
                        sizeof(int32_t));
  }

  // infer
  ASSERT_TRUE(control_model.Predict(inputs, &outputs) == kSuccess);

  // assert output
  ASSERT_TRUE(outputs.size() == 1);
  auto out = outputs[0];
  ASSERT_TRUE(out.DataSize() == sizeof(int32_t));
  auto out_data = out.Data();
  auto p = reinterpret_cast<const int32_t *>(out_data.get());
  ASSERT_EQ(*p, 125);
}

TEST_F(TestControl, InferSingleOr) {
  auto context = ContextAutoSet();

  Graph graph;
  ASSERT_TRUE(Serialization::Load(kSingleOrFile, ModelType::kMindIR, &graph));
  Model control_model;
  ASSERT_TRUE(control_model.Build(GraphCell(graph), context) == kSuccess);

  // assert inputs
  std::vector<MSTensor> inputs_before = control_model.GetInputs();
  ASSERT_EQ(inputs_before.size(), 2);
  EXPECT_EQ(inputs_before[0].DataType(), DataType::kNumberTypeFloat32);
  EXPECT_EQ(inputs_before[1].DataType(), DataType::kNumberTypeFloat32);
  ASSERT_EQ(inputs_before[0].DataSize(), sizeof(float) * 2);
  ASSERT_EQ(inputs_before[1].DataSize(), sizeof(float) * 2);
  ASSERT_EQ(inputs_before[0].Shape().size(), 1);
  EXPECT_EQ(inputs_before[0].Shape()[0], 2);
  ASSERT_EQ(inputs_before[1].Shape().size(), 1);
  EXPECT_EQ(inputs_before[1].Shape()[0], 2);

  // assert outputs
  std::vector<MSTensor> outputs_before = control_model.GetOutputs();
  ASSERT_EQ(1, outputs_before.size());
  EXPECT_EQ(outputs_before[0].DataType(), DataType::kNumberTypeFloat32);
  ASSERT_TRUE(outputs_before[0].DataSize() == sizeof(float));

  // prepare input
  std::vector<MSTensor> outputs;
  std::vector<MSTensor> inputs;
  {
    static const std::vector<float> input_data1 = {0, 1};
    static const std::vector<float> input_data2 = {0, 0};
    inputs.emplace_back(inputs_before[0].Name(), inputs_before[0].DataType(), inputs_before[0].Shape(),
                        input_data1.data(), sizeof(float) * input_data1.size());
    inputs.emplace_back(inputs_before[1].Name(), inputs_before[1].DataType(), inputs_before[1].Shape(),
                        input_data2.data(), sizeof(int32_t) * input_data2.size());
  }

  // infer
  ASSERT_TRUE(control_model.Predict(inputs, &outputs) == kSuccess);

  // assert outputs
  std::vector<MSTensor> outputs_after = control_model.GetOutputs();
  ASSERT_EQ(1, outputs_after.size());
  EXPECT_EQ(outputs_after[0].DataType(), DataType::kNumberTypeFloat32);
  ASSERT_TRUE(outputs_after[0].DataSize() == sizeof(float));
  EXPECT_EQ(outputs_after[0].Shape().size(), outputs_before[0].Shape().size());

  // assert output
  ASSERT_TRUE(outputs.size() == 1);
  auto out = outputs[0];
  ASSERT_TRUE(out.DataSize() == sizeof(float));
  auto out_data = out.Data();
  auto p = reinterpret_cast<const float *>(out_data.get());
  ASSERT_EQ(*p, 1);
}

TEST_F(TestControl, InferSingleSwitch) {
  auto context = ContextAutoSet();

  Graph graph;
  ASSERT_TRUE(Serialization::Load(kSingleSwitchFile, ModelType::kMindIR, &graph));
  Model control_model;
  ASSERT_TRUE(control_model.Build(GraphCell(graph), context) == kSuccess);

  // assert inputs
  std::vector<MSTensor> inputs_before = control_model.GetInputs();
  ASSERT_EQ(inputs_before.size(), 3);
  EXPECT_EQ(inputs_before[0].DataType(), DataType::kNumberTypeFloat32);
  EXPECT_EQ(inputs_before[1].DataType(), DataType::kNumberTypeInt32);
  EXPECT_EQ(inputs_before[2].DataType(), DataType::kNumberTypeInt32);
  ASSERT_EQ(inputs_before[0].DataSize(), sizeof(float) * 224 * 224);
  ASSERT_EQ(inputs_before[1].DataSize(), sizeof(int32_t));
  ASSERT_EQ(inputs_before[2].DataSize(), sizeof(int32_t));
  ASSERT_EQ(inputs_before[0].Shape().size(), 4);
  EXPECT_EQ(inputs_before[0].Shape()[0], 1);
  EXPECT_EQ(inputs_before[0].Shape()[1], 1);
  EXPECT_EQ(inputs_before[0].Shape()[2], 224);
  EXPECT_EQ(inputs_before[0].Shape()[3], 224);
  ASSERT_EQ(inputs_before[1].Shape().size(), 1);
  EXPECT_EQ(inputs_before[1].Shape()[0], 1);
  ASSERT_EQ(inputs_before[2].Shape().size(), 1);
  EXPECT_EQ(inputs_before[2].Shape()[0], 1);

  // assert outputs
  std::vector<MSTensor> outputs_before = control_model.GetOutputs();
  ASSERT_EQ(1, outputs_before.size());
  EXPECT_EQ(outputs_before[0].DataType(), DataType::kNumberTypeFloat32);
  ASSERT_TRUE(outputs_before[0].DataSize() == sizeof(float) * 224 * 224);
  ASSERT_EQ(outputs_before[0].Shape().size(), 4);
  EXPECT_EQ(outputs_before[0].Shape()[0], 1);
  EXPECT_EQ(outputs_before[0].Shape()[1], 1);
  EXPECT_EQ(outputs_before[0].Shape()[2], 224);
  EXPECT_EQ(outputs_before[0].Shape()[3], 224);

  // prepare input
  std::vector<MSTensor> outputs;
  std::vector<MSTensor> inputs;
  {
    static const std::vector<float> input_data1(1 * 1 * 224 * 224, 1);
    int32_t index1 = 0;
    int32_t index2 = -1;
    inputs.emplace_back(inputs_before[0].Name(), inputs_before[0].DataType(), inputs_before[0].Shape(),
                        input_data1.data(), sizeof(float) * input_data1.size());
    inputs.emplace_back(inputs_before[1].Name(), inputs_before[1].DataType(), inputs_before[1].Shape(), &index1,
                        sizeof(int32_t));
    inputs.emplace_back(inputs_before[2].Name(), inputs_before[2].DataType(), inputs_before[2].Shape(), &index2,
                        sizeof(int32_t));
  }

  // infer
  ASSERT_TRUE(control_model.Predict(inputs, &outputs) == kSuccess);

  // assert output
  ASSERT_TRUE(outputs.size() == 1);
  auto out = outputs[0];
  ASSERT_TRUE(out.DataSize() == sizeof(float) * 224 * 224);
  auto out_data = out.Data();
  auto p = reinterpret_cast<const float *>(out_data.get());
  for (size_t i = 0; i < out.DataSize() / sizeof(float); ++i) {
    ASSERT_EQ(p[i], 1);
  }
}
