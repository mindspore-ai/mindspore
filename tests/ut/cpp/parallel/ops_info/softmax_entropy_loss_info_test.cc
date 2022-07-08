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
#include "frontend/parallel/ops_info/loss_info.h"
#include "frontend/parallel/device_manager.h"
#include "frontend/parallel/tensor_layout/tensor_redistribution.h"

namespace mindspore {
namespace parallel {

class SoftmaxCrossEntropyWithLogitsInfo;
using LossPtr = std::shared_ptr<SoftmaxCrossEntropyWithLogitsInfo>;
LossPtr loss;

class TestSoftmaxLoss : public UT::Common {
 public:
  TestSoftmaxLoss() {}
  void SetUp();
  void TearDown() {}
};

void TestSoftmaxLoss::SetUp() {
  RankList dev_list;

  for (int32_t i = 0; i < 65; i++) {
    dev_list.push_back(i);
  }

  RankList stage_map;
  stage_map.push_back(64);
  stage_map.push_back(1);

  int32_t local_dev = 0;

  // create a new g_device_manager
  g_device_manager = std::make_shared<DeviceManager>();
  g_device_manager->Init(dev_list, local_dev, stage_map, "hccl");

  ValuePtr is_grad = MakeValue(true);
  mindspore::HashMap<std::string, ValuePtr> attr = {{"is_grad", is_grad}};

  Shapes inputs_shape = {{2, 4, 8, 16}, {2, 4, 8, 16}};
  Shapes outputs_shape = {{2}, {2, 4, 8, 16}};

  loss = std::make_shared<SoftmaxCrossEntropyWithLogitsInfo>("CrossEntropy_info", inputs_shape, outputs_shape, attr);
}

TEST_F(TestSoftmaxLoss, InferDevMatrixShape1) {
  Strategies inputs = {{2, 4, 8, 1}, {2, 4, 8, 1}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  loss->Init(strategy, nullptr);
  Shape dev_matrix_shape = loss->dev_matrix_shape();

  Shape expect = {2, 4, 8, 1};
  ASSERT_EQ(dev_matrix_shape, expect);
}

TEST_F(TestSoftmaxLoss, InferSliceShape1) {
  Strategies str = {{2, 4, 8, 1}, {2, 4, 8, 1}};
  StrategyPtr strategy = NewStrategy(0, str);

  loss->Init(strategy, nullptr);
  std::vector<TensorInfo> inputs = loss->inputs_tensor_info();
  std::vector<TensorInfo> outputs = loss->outputs_tensor_info();

  Shape input_slice_shape_expect = {1, 1, 1, 16};
  Shape label_slice_shape_expect = {1, 1, 1, 16};
  Shape output_0_slice_shape_expect = {1};
  Shape output_1_slice_shape_expect = {1, 1, 1, 16};

  TensorInfo input_tensor_info = inputs.at(0);
  TensorInfo label_tensor_info = inputs.at(1);
  TensorInfo output_0_tensor_info = outputs.at(0);
  TensorInfo output_1_tensor_info = outputs.at(1);

  Shape input_slice_shape = input_tensor_info.slice_shape();
  Shape label_slice_shape = label_tensor_info.slice_shape();
  Shape output_0_slice_shape = output_0_tensor_info.slice_shape();
  Shape output_1_slice_shape = output_1_tensor_info.slice_shape();

  ASSERT_EQ(input_slice_shape, input_slice_shape_expect);
  ASSERT_EQ(label_slice_shape, label_slice_shape_expect);
  ASSERT_EQ(output_0_slice_shape, output_0_slice_shape_expect);
  ASSERT_EQ(output_1_slice_shape, output_1_slice_shape_expect);
}

TEST_F(TestSoftmaxLoss, GetTensorLayout1) {
  Strategies str = {{2, 4, 8, 1}, {2, 4, 8, 1}};
  StrategyPtr strategy = NewStrategy(0, str);

  loss->Init(strategy, nullptr);
  std::vector<TensorInfo> inputs = loss->inputs_tensor_info();
  std::vector<TensorInfo> outputs = loss->outputs_tensor_info();

  TensorMap input_expect = {3, 2, 1, 0};
  TensorMap label_expect = {3, 2, 1, 0};
  TensorMap output_0_expect = {3};
  TensorMap output_1_expect = {3, 2, 1, 0};

  TensorInfo input_tensor_info = inputs.at(0);
  TensorInfo label_tensor_info = inputs.at(1);
  TensorInfo output_0_tensor_info = outputs.at(0);
  TensorInfo output_1_tensor_info = outputs.at(1);

  Map input_tensor_map = input_tensor_info.tensor_layout().origin_tensor_map();
  Map label_tensor_map = label_tensor_info.tensor_layout().origin_tensor_map();
  Map output_0_tensor_map = output_0_tensor_info.tensor_layout().origin_tensor_map();
  Map output_1_tensor_map = output_1_tensor_info.tensor_layout().origin_tensor_map();

  ASSERT_EQ(input_tensor_map.array(), input_expect);
  ASSERT_EQ(label_tensor_map.array(), label_expect);
  ASSERT_EQ(output_0_tensor_map.array(), output_0_expect);
  ASSERT_EQ(output_1_tensor_map.array(), output_1_expect);
}

TEST_F(TestSoftmaxLoss, GetForwardOp1) {
  Strategies inputs = {{2, 4, 8, 1}, {2, 4, 8, 1}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  loss->Init(strategy, nullptr);
  OperatorVector forward_op = loss->forward_op();
  size_t size = forward_op.size();

  ASSERT_EQ(size, 0);
}

TEST_F(TestSoftmaxLoss, GetMirrorOPs1) {
  Strategies inputs = {{2, 4, 8, 1}, {2, 4, 8, 1}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  loss->Init(strategy, nullptr);
  MirrorOps mirror_ops = loss->mirror_ops();

  size_t size = mirror_ops.size();

  ASSERT_EQ(size, 0);
}

TEST_F(TestSoftmaxLoss, GetVirtualDivOPs1) {
  Strategies inputs = {{1, 4, 8, 1}, {1, 4, 8, 1}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  loss->Init(strategy, nullptr);
  OperatorVector virtual_div_op = loss->virtual_div_op();

  OperatorArgs operator_args = virtual_div_op.at(0).second;

  std::string arg0_name = operator_args.first.at(0).first;
  ValuePtr arg0_value = operator_args.first.at(0).second;
  int64_t divisor = arg0_value->cast<Int64ImmPtr>()->value();

  ASSERT_EQ(virtual_div_op.at(0).first, "_VirtualDiv");
  ASSERT_EQ(virtual_div_op.size(), 1);
  ASSERT_EQ(arg0_name, "divisor");
  ASSERT_EQ(divisor, 2);
}

TEST_F(TestSoftmaxLoss, CheckStrategy1) {
  // Success: {{2,4,8,16}}
  Strategies inputs = {{2, 2, 8, 16}, {2, 4, 16, 1}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  Status ret = loss->Init(strategy, nullptr);
  ASSERT_EQ(ret, FAILED);
}

TEST_F(TestSoftmaxLoss, CheckStrategy2) {
  // Success: {{2,4,8,16}}
  Strategies inputs = {{2, 4, 8}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  Status ret = loss->Init(strategy, nullptr);
  ASSERT_EQ(ret, FAILED);
}

}  // namespace parallel
}  // namespace mindspore
