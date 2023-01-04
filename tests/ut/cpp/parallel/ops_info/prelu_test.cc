/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#include <list>
#include <vector>
#include "common/common_test.h"
#include "frontend/parallel/strategy.h"
#include "frontend/parallel/ops_info/prelu_info.h"
#include "frontend/parallel/device_manager.h"
#include "frontend/parallel/step_parallel.h"

namespace mindspore {
namespace parallel {

class PReLUInfo;
using PReLUInfoPtr = std::shared_ptr<PReLUInfo>;
PReLUInfoPtr prelu;
PReLUInfoPtr prelu_2d;

class TestPReLUInfo : public UT::Common {
 public:
  TestPReLUInfo() {}
  void SetUp();
  void TearDown() {}
};

void TestPReLUInfo::SetUp() {
  RankList dev_list;

  for (int32_t i = 0; i < 1050; i++) {
    dev_list.push_back(i);
  }

  RankList stage_map;
  stage_map.push_back(1024);
  stage_map.push_back(26);
  int32_t local_dev = 0;
  // create a new g_device_manager
  g_device_manager = std::make_shared<DeviceManager>();
  g_device_manager->Init(dev_list, local_dev, stage_map, "hccl");
  Shapes inputs_shape = {{64, 4, 8, 16}, {4}};
  Shapes outputs_shape = {{64, 4, 8, 16}};
  mindspore::HashMap<std::string, ValuePtr> attr;
  prelu = std::make_shared<PReLUInfo>("prelu_info", inputs_shape, outputs_shape, attr);

  Shapes inputs_shape_2d = {{1024, 4}, {4}};
  Shapes outputs_shape_2d = {{1024, 4}};
  mindspore::HashMap<std::string, ValuePtr> attr_2d;
  prelu_2d = std::make_shared<PReLUInfo>("prelu_info", inputs_shape_2d, outputs_shape_2d, attr_2d);
}

TEST_F(TestPReLUInfo, InferDevMatrixShape1) {
  Strategies inputs = {{2, 1, 8, 16}, {1}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  prelu->Init(strategy, nullptr);
  Shape dev_matrix_shape = prelu->dev_matrix_shape();

  Shape expect = {2, 1, 8, 16, 4};
  ASSERT_EQ(dev_matrix_shape, expect);
}

TEST_F(TestPReLUInfo, InferSliceShape1) {
  Strategies str = {{2, 1, 8, 16}, {1}};
  StrategyPtr strategy = NewStrategy(0, str);

  prelu->Init(strategy, nullptr);
  std::vector<TensorInfo> inputs = prelu->inputs_tensor_info();
  std::vector<TensorInfo> outputs = prelu->outputs_tensor_info();

  Shape input_slice_shape_expect = {32, 4, 1, 1};
  Shape param_slice_shape_expect = {4};
  Shape output_slice_shape_expect = {32, 4, 1, 1};

  TensorInfo input_tensor_info = inputs.at(0);
  TensorInfo param_tensor_info = inputs.at(1);
  TensorInfo output_tensor_info = outputs.at(0);

  Shape input_slice_shape = input_tensor_info.slice_shape();
  Shape output_slice_shape = output_tensor_info.slice_shape();

  ASSERT_EQ(input_slice_shape, input_slice_shape_expect);
  ASSERT_EQ(output_slice_shape, output_slice_shape_expect);
}

TEST_F(TestPReLUInfo, GetTensorLayout1) {
  Strategies str = {{2, 1, 8, 16}, {1}};
  StrategyPtr strategy = NewStrategy(0, str);

  prelu->Init(strategy, nullptr);
  std::vector<TensorInfo> inputs = prelu->inputs_tensor_info();
  std::vector<TensorInfo> outputs = prelu->outputs_tensor_info();

  TensorMap input_expect = {4, 3, 2, 1};
  TensorMap param_expect = {2};
  TensorMap output_expect = {4, 3, 2, 1};

  TensorInfo input_tensor_info = inputs.at(0);
  TensorInfo param_tensor_info = inputs.at(1);
  TensorInfo output_tensor_info = outputs.at(0);

  Map input_tensor_map = input_tensor_info.tensor_layout().origin_tensor_map();
  Map param_tensor_map = param_tensor_info.tensor_layout().origin_tensor_map();
  Map output_tensor_map = output_tensor_info.tensor_layout().origin_tensor_map();

  ASSERT_EQ(input_tensor_map.array(), input_expect);
  ASSERT_EQ(output_tensor_map.array(), output_expect);
}

TEST_F(TestPReLUInfo, GetMirrorOPs1) {
  Strategies str = {{2, 1, 2, 2}, {1}};
  StrategyPtr strategy = NewStrategy(0, str);
  prelu->Init(strategy, nullptr);
  MirrorOps mirror_ops = prelu->mirror_ops();
  OperatorVector mirror_op = mirror_ops.at(1);
  OperatorArgs operator_args = mirror_op.at(0).second;
  std::string arg0_name = operator_args.first.at(0).first;
  ValuePtr arg0_value = operator_args.first.at(0).second;
  std::string group = arg0_value->cast<StringImmPtr>()->ToString();

  ASSERT_EQ(mirror_op.at(0).first, "_MirrorOperator");
  ASSERT_EQ(mirror_op.size(), 1);
  ASSERT_EQ(arg0_name, "group");
}

TEST_F(TestPReLUInfo, CheckStrategy1) {
  // Success: {{2,1,8,16},{1}}
  Strategies inputs = {{2, 1, 8, 16}};
  StrategyPtr strategy = NewStrategy(0, inputs);
  Status ret = prelu->Init(strategy, nullptr);
  ASSERT_EQ(ret, FAILED);
}

TEST_F(TestPReLUInfo, CheckStrategy2) {
  Strategies inputs = {{2, 4, 8, 16}, {4}};
  StrategyPtr strategy = NewStrategy(0, inputs);
  Status ret = prelu->Init(strategy, nullptr);
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(TestPReLUInfo, AutoStrategy1) {
  ASSERT_EQ(prelu->GenerateStrategies(0), Status::SUCCESS);
  std::vector<std::shared_ptr<StrategyWithCost>> sc = prelu->GetStrategyCost();

  Shapes splittable_inputs = {{1, 0, 1, 1}, {0}};
  std::vector<StrategyPtr> sp_vector;
  Shapes inputs_shape = {{64, 4, 8, 16}, {4}};
  GenerateStrategiesForIndependentInputs(0, inputs_shape, splittable_inputs, &sp_vector);
  for (auto stra : sp_vector) {
    auto stra0 = stra->GetInputDim()[0];
    auto stra1 = stra->GetInputDim()[1];
    ASSERT_EQ(stra0[1], 1);
    ASSERT_EQ(stra1[0], 1);
  }
}

TEST_F(TestPReLUInfo, InferDevMatrixShape_2d1) {
  Strategies inputs = {{128, 1}, {1}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  prelu_2d->Init(strategy, nullptr);
  Shape dev_matrix_shape = prelu_2d->dev_matrix_shape();

  Shape expect = {128, 1, 8};
  ASSERT_EQ(dev_matrix_shape, expect);
}

TEST_F(TestPReLUInfo, InferSliceShape_2d1) {
  Strategies str = {{128, 1}, {1}};
  StrategyPtr strategy = NewStrategy(0, str);

  prelu_2d->Init(strategy, nullptr);
  std::vector<TensorInfo> inputs = prelu_2d->inputs_tensor_info();
  std::vector<TensorInfo> outputs = prelu_2d->outputs_tensor_info();

  Shape input_slice_shape_expect = {8, 4};
  Shape param_slice_shape_expect = {4};
  Shape output_slice_shape_expect = {8, 4};

  TensorInfo input_tensor_info = inputs.at(0);
  TensorInfo param_tensor_info = inputs.at(1);
  TensorInfo output_tensor_info = outputs.at(0);

  Shape input_slice_shape = input_tensor_info.slice_shape();
  Shape output_slice_shape = output_tensor_info.slice_shape();

  ASSERT_EQ(input_slice_shape, input_slice_shape_expect);
  ASSERT_EQ(output_slice_shape, output_slice_shape_expect);
}

TEST_F(TestPReLUInfo, GetTensorLayout_2d1) {
  Strategies str = {{128, 1}, {1}};
  StrategyPtr strategy = NewStrategy(0, str);

  prelu_2d->Init(strategy, nullptr);
  std::vector<TensorInfo> inputs = prelu_2d->inputs_tensor_info();
  std::vector<TensorInfo> outputs = prelu_2d->outputs_tensor_info();

  TensorMap input_expect = {2, 1};
  TensorMap param_expect = {0};
  TensorMap output_expect = {2, 1};

  TensorInfo input_tensor_info = inputs.at(0);
  TensorInfo param_tensor_info = inputs.at(1);
  TensorInfo output_tensor_info = outputs.at(0);

  Map input_tensor_map = input_tensor_info.tensor_layout().origin_tensor_map();
  Map param_tensor_map = param_tensor_info.tensor_layout().origin_tensor_map();
  Map output_tensor_map = output_tensor_info.tensor_layout().origin_tensor_map();

  ASSERT_EQ(input_tensor_map.array(), input_expect);
  ASSERT_EQ(output_tensor_map.array(), output_expect);
}

TEST_F(TestPReLUInfo, GetMirrorOPs_2d1) {
  Strategies str = {{128, 1}, {1}};
  StrategyPtr strategy = NewStrategy(0, str);
  prelu_2d->Init(strategy, nullptr);
  MirrorOps mirror_ops = prelu_2d->mirror_ops();
  OperatorVector mirror_op = mirror_ops.at(1);
  OperatorArgs operator_args = mirror_op.at(0).second;
  std::string arg0_name = operator_args.first.at(0).first;
  ValuePtr arg0_value = operator_args.first.at(0).second;
  std::string group = arg0_value->cast<StringImmPtr>()->ToString();

  ASSERT_EQ(mirror_op.at(0).first, "_MirrorOperator");
  ASSERT_EQ(mirror_op.size(), 1);
  ASSERT_EQ(arg0_name, "group");
}

TEST_F(TestPReLUInfo, CheckStrategy_2d1) {
  // Success: {{2,1,8,16},{1}}
  Strategies inputs = {{128, 1}};
  StrategyPtr strategy = NewStrategy(0, inputs);
  Status ret = prelu_2d->Init(strategy, nullptr);
  ASSERT_EQ(ret, FAILED);
}

TEST_F(TestPReLUInfo, CheckStrategy_2d2) {
  Strategies inputs = {{128, 4}, {4}};
  StrategyPtr strategy = NewStrategy(0, inputs);
  Status ret = prelu_2d->Init(strategy, nullptr);
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(TestPReLUInfo, AutoStrategy_2d1) {
  ASSERT_EQ(prelu_2d->GenerateStrategies(0), Status::SUCCESS);
  std::vector<std::shared_ptr<StrategyWithCost>> sc = prelu_2d->GetStrategyCost();

  Shapes splittable_inputs = {{1, 0}, {0}};
  std::vector<StrategyPtr> sp_vector;
  Shapes inputs_shape = {{1024, 4}, {4}};
  GenerateStrategiesForIndependentInputs(0, inputs_shape, splittable_inputs, &sp_vector);
  for (auto stra : sp_vector) {
    auto stra0 = stra->GetInputDim()[0];
    auto stra1 = stra->GetInputDim()[1];
    ASSERT_EQ(stra0[1], 1);
    ASSERT_EQ(stra1[0], 1);
  }
}
}  // namespace parallel
}  // namespace mindspore
