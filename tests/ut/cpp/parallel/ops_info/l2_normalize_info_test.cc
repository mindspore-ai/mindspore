/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include "frontend/parallel/ops_info/l2_normalize_info.h"
#include "frontend/parallel/device_manager.h"
#include "frontend/parallel/step_parallel.h"

namespace mindspore {
namespace parallel {

class L2NormalizeInfo;
using L2NormalizeInfoPtr = std::shared_ptr<L2NormalizeInfo>;
L2NormalizeInfoPtr norm;

class TestL2NormalizeInfo : public UT::Common {
 public:
  TestL2NormalizeInfo() {}
  void SetUp();
  void TearDown() {}
};

void TestL2NormalizeInfo::SetUp() {
  RankList dev_list;

  for (int32_t i = 0; i < 34; i++) {
    dev_list.push_back(i);
  }

  RankList stage_map;
  stage_map.push_back(32);
  stage_map.push_back(2);

  int32_t local_dev = 0;

  // create a new g_device_manager
  g_device_manager = std::make_shared<DeviceManager>();
  g_device_manager->Init(dev_list, local_dev, stage_map, "hccl");

  ValuePtr axis = MakeValue(std::vector<int64_t>{1});
  std::unordered_map<std::string, ValuePtr> attr = {{AXIS, axis}};

  Shapes inputs_shape = {{32, 64, 96}};
  Shapes outputs_shape = {{32, 64, 96}};

  norm = std::make_shared<L2NormalizeInfo>("l2_normalize_info", inputs_shape, outputs_shape, attr);
}

TEST_F(TestL2NormalizeInfo, InferDevMatrixShape1) {
  Strategys inputs = {{4, 1, 8}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  norm->Init(strategy);
  Shape dev_matrix_shape = norm->dev_matrix_shape();

  Shape expect = {4, 1, 8};
  ASSERT_EQ(dev_matrix_shape, expect);
}

TEST_F(TestL2NormalizeInfo, InferSliceShape1) {
  Strategys str = {{4, 1, 8}};
  StrategyPtr strategy = NewStrategy(0, str);

  norm->Init(strategy);
  std::vector<TensorInfo> inputs = norm->inputs_tensor_info();
  std::vector<TensorInfo> outputs = norm->outputs_tensor_info();

  Shape input_slice_shape_expect = {8, 64, 12};
  Shape output_slice_shape_expect = {8, 64, 12};

  TensorInfo input_tensor_info = inputs.at(0);
  TensorInfo output_tensor_info = outputs.at(0);

  Shape input_slice_shape = input_tensor_info.slice_shape();
  Shape output_slice_shape = output_tensor_info.slice_shape();

  ASSERT_EQ(input_slice_shape, input_slice_shape_expect);
  ASSERT_EQ(output_slice_shape, output_slice_shape_expect);
}

TEST_F(TestL2NormalizeInfo, GetTensorLayout1) {
  Strategys str = {{4, 1, 8}};
  StrategyPtr strategy = NewStrategy(0, str);

  norm->Init(strategy);
  std::vector<TensorInfo> inputs = norm->inputs_tensor_info();
  std::vector<TensorInfo> outputs = norm->outputs_tensor_info();

  TensorMap input_expect = {2, 1, 0};
  TensorMap output_expect = {2, 1, 0};

  TensorInfo input_tensor_info = inputs.at(0);
  TensorInfo output_tensor_info = outputs.at(0);

  Map input_tensor_map = input_tensor_info.tensor_layout().origin_tensor_map();
  Map output_tensor_map = output_tensor_info.tensor_layout().origin_tensor_map();

  ASSERT_EQ(input_tensor_map.array(), input_expect);
  ASSERT_EQ(output_tensor_map.array(), output_expect);
}

TEST_F(TestL2NormalizeInfo, GetForwardOp1) {
  Strategys inputs = {{4, 1, 8}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  norm->Init(strategy);
  OperatorVector forward_op = norm->forward_op();
  size_t size = forward_op.size();

  ASSERT_EQ(size, 0);
}

TEST_F(TestL2NormalizeInfo, GetMirrorOPs1) {
  Strategys inputs = {{4, 1, 8}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  norm->Init(strategy);
  MirrorOps mirror_ops = norm->mirror_ops();

  size_t size = mirror_ops.size();

  ASSERT_EQ(size, 0);
}

TEST_F(TestL2NormalizeInfo, CheckStrategy1) {
  Strategys inputs = {{4, 1, 8}, {4, 1, 8}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  Status ret = norm->Init(strategy);
  ASSERT_EQ(ret, FAILED);
}

TEST_F(TestL2NormalizeInfo, CheckStrategy2) {
  Strategys inputs = {{4, 2, 3}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  Status ret = norm->Init(strategy);
  ASSERT_EQ(ret, FAILED);
}

TEST_F(TestL2NormalizeInfo, CheckStrategy3) {
  Strategys inputs = {{4, 2, 3, 4}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  Status ret = norm->Init(strategy);
  ASSERT_EQ(ret, FAILED);
}

TEST_F(TestL2NormalizeInfo, CheckStrategy4) {
  Strategys inputs = {{4, 1, 8}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  Status ret = norm->Init(strategy);
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(TestL2NormalizeInfo, mirror_ops) {
  Strategys inputs = {{2, 1, 8}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  norm->Init(strategy);
  MirrorOps mirror_ops = norm->mirror_ops();
  OperatorVector mirror_op = mirror_ops.at(0);

  OperatorArgs operator_args = mirror_op.at(0).second;
  std::string arg0_name = operator_args.first.at(0).first;
  ValuePtr arg0_value = operator_args.first.at(0).second;
  std::string group = arg0_value->cast<StringImmPtr>()->ToString();

  ASSERT_EQ(mirror_op.at(0).first, "_MirrorOperator");
  ASSERT_EQ(mirror_op.size(), 1);
  ASSERT_EQ(arg0_name, "group");
}

}  // namespace parallel
}  // namespace mindspore
