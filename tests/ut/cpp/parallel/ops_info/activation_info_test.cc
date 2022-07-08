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
#include "frontend/parallel/ops_info/activation_info.h"
#include "frontend/parallel/device_manager.h"
#include "frontend/parallel/step_parallel.h"

namespace mindspore {
namespace parallel {

class ActivationInfo;
using ActivationInfoPtr = std::shared_ptr<ActivationInfo>;
ActivationInfoPtr activation;

class TestActivationInfo : public UT::Common {
 public:
  TestActivationInfo() {}
  void SetUp();
  void TearDown() {}
};

void TestActivationInfo::SetUp() {
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

  ValuePtr relu = MakeValue(std::string("relu"));
  mindspore::HashMap<std::string, ValuePtr> attr = {{"activation_type", relu}};

  Shapes inputs_shape = {{2, 4, 8, 16}};
  Shapes outputs_shape = {{2, 4, 8, 16}};

  activation = std::make_shared<ActivationInfo>("activation_info", inputs_shape, outputs_shape, attr);
}

TEST_F(TestActivationInfo, InferDevMatrixShape1) {
  Strategies inputs = {{2, 4, 8, 16}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  activation->Init(strategy, nullptr);
  Shape dev_matrix_shape = activation->dev_matrix_shape();

  Shape expect = {2, 4, 8, 16};
  ASSERT_EQ(dev_matrix_shape, expect);
}

TEST_F(TestActivationInfo, InferSliceShape1) {
  Strategies str = {{2, 4, 8, 16}};
  StrategyPtr strategy = NewStrategy(0, str);

  activation->Init(strategy, nullptr);
  std::vector<TensorInfo> inputs = activation->inputs_tensor_info();
  std::vector<TensorInfo> outputs = activation->outputs_tensor_info();

  Shape input_slice_shape_expect = {1, 1, 1, 1};
  Shape output_slice_shape_expect = {1, 1, 1, 1};

  TensorInfo input_tensor_info = inputs.at(0);
  TensorInfo output_tensor_info = outputs.at(0);

  Shape input_slice_shape = input_tensor_info.slice_shape();
  Shape output_slice_shape = output_tensor_info.slice_shape();

  ASSERT_EQ(input_slice_shape, input_slice_shape_expect);
  ASSERT_EQ(output_slice_shape, output_slice_shape_expect);
}

TEST_F(TestActivationInfo, GetTensorLayout1) {
  Strategies str = {{2, 4, 8, 16}};
  StrategyPtr strategy = NewStrategy(0, str);

  activation->Init(strategy, nullptr);
  std::vector<TensorInfo> inputs = activation->inputs_tensor_info();
  std::vector<TensorInfo> outputs = activation->outputs_tensor_info();

  TensorMap input_expect = {3, 2, 1, 0};
  TensorMap output_expect = {3, 2, 1, 0};

  TensorInfo input_tensor_info = inputs.at(0);
  TensorInfo output_tensor_info = outputs.at(0);

  Map input_tensor_map = input_tensor_info.tensor_layout().origin_tensor_map();
  Map output_tensor_map = output_tensor_info.tensor_layout().origin_tensor_map();

  ASSERT_EQ(input_tensor_map.array(), input_expect);
  ASSERT_EQ(output_tensor_map.array(), output_expect);
}

TEST_F(TestActivationInfo, GetForwardOp1) {
  Strategies inputs = {{2, 4, 8, 16}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  activation->Init(strategy, nullptr);
  OperatorVector forward_op = activation->forward_op();
  size_t size = forward_op.size();

  ASSERT_EQ(size, 0);
}

TEST_F(TestActivationInfo, GetMirrorOPs1) {
  Strategies inputs = {{1, 4, 8, 16}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  activation->Init(strategy, nullptr);
  MirrorOps mirror_ops = activation->mirror_ops();

  OperatorVector mirror_op = mirror_ops.at(0);

  OperatorArgs operator_args = mirror_op.at(0).second;

  std::string arg0_name = operator_args.first.at(0).first;
  ValuePtr arg0_value = operator_args.first.at(0).second;
  std::string group = arg0_value->cast<StringImmPtr>()->ToString();

  ASSERT_EQ(mirror_op.at(0).first, "_MirrorOperator");
  ASSERT_EQ(mirror_op.size(), 1);
  ASSERT_EQ(arg0_name, "group");
}

TEST_F(TestActivationInfo, GetMirrorOPs2) {
  Strategies inputs = {{2, 4, 8, 16}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  activation->Init(strategy, nullptr);
  MirrorOps mirror_ops = activation->mirror_ops();

  size_t size = mirror_ops.size();

  ASSERT_EQ(size, 0);
}

TEST_F(TestActivationInfo, CheckStrategy1) {
  // Success: {{2,4,8,16}}
  Strategies inputs = {{2, 2, 8, 16}, {2, 4, 16, 1}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  Status ret = activation->Init(strategy, nullptr);
  ASSERT_EQ(ret, FAILED);
}

TEST_F(TestActivationInfo, CheckStrategy2) {
  // Success: {{2,4,8,16}}
  Strategies inputs = {{2, 4, 8}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  Status ret = activation->Init(strategy, nullptr);
  ASSERT_EQ(ret, FAILED);
}

}  // namespace parallel
}  // namespace mindspore
