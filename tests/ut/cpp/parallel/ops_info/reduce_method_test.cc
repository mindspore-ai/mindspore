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
#include "frontend/parallel/ops_info/reduce_method_info.h"
#include "frontend/parallel/ops_info/reduce_base_method_info.h"
#include "common/py_func_graph_fetcher.h"
#include "frontend/parallel/device_manager.h"
#include "frontend/parallel/step_parallel.h"

namespace mindspore {
namespace parallel {

using ReduceSumInfoPtr = std::shared_ptr<ReduceSumInfo>;
ReduceSumInfoPtr reduce_sum;

class TestReduceSumInfo : public UT::Common {
 public:
  TestReduceSumInfo() {}
  void SetUp();
  void TearDown() {}
};

void TestReduceSumInfo::SetUp() {
  UT::InitPythonPath();
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

  Shapes inputs_shape = {{16, 32, 64}};
  Shapes outputs_shape = {{16, 32}};
  ValuePtr value = MakeValue(static_cast<int64_t>(-1));
  ValuePtr value0;
  std::vector<ValuePtr> val = {value0, value};
  ValuePtr keep_dims = MakeValue(false);
  mindspore::HashMap<std::string, ValuePtr> attr = {{KEEP_DIMS, keep_dims}};

  reduce_sum = std::make_shared<ReduceSumInfo>("sum_info", inputs_shape, outputs_shape, attr);
  reduce_sum->set_input_value(val);
}

TEST_F(TestReduceSumInfo, DISABLED_InferDevMatrixShape1) {
  Strategies inputs = {{4, 8, 1}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  reduce_sum->Init(strategy, nullptr);
  Shape dev_matrix_shape = reduce_sum->dev_matrix_shape();

  Shape expect = {4, 8, 1};
  ASSERT_EQ(dev_matrix_shape, expect);
}

TEST_F(TestReduceSumInfo, DISABLED_InferSliceShape1) {
  Strategies str = {{4, 8, 1}};
  StrategyPtr strategy = NewStrategy(0, str);

  reduce_sum->Init(strategy, nullptr);
  std::vector<TensorInfo> inputs = reduce_sum->inputs_tensor_info();
  std::vector<TensorInfo> outputs = reduce_sum->outputs_tensor_info();

  Shape input_slice_shape_expect = {4, 4, 64};
  Shape output_slice_shape_expect = {4, 4};

  TensorInfo input_tensor_info = inputs.at(0);
  TensorInfo output_tensor_info = outputs.at(0);

  Shape input_slice_shape = input_tensor_info.slice_shape();
  Shape output_slice_shape = output_tensor_info.slice_shape();

  ASSERT_EQ(input_slice_shape, input_slice_shape_expect);
  ASSERT_EQ(output_slice_shape, output_slice_shape_expect);
}

TEST_F(TestReduceSumInfo, DISABLED_GetTensorLayout1) {
  Strategies str = {{4, 8, 1}};
  StrategyPtr strategy = NewStrategy(0, str);

  reduce_sum->Init(strategy, nullptr);
  std::vector<TensorInfo> inputs = reduce_sum->inputs_tensor_info();
  std::vector<TensorInfo> outputs = reduce_sum->outputs_tensor_info();

  TensorMap input_expect = {2, 1, 0};
  TensorMap output_expect = {2, 1};

  TensorInfo input_tensor_info = inputs.at(0);
  TensorInfo output_tensor_info = outputs.at(0);

  Map input_tensor_map = input_tensor_info.tensor_layout().origin_tensor_map();
  Map output_tensor_map = output_tensor_info.tensor_layout().origin_tensor_map();

  ASSERT_EQ(input_tensor_map.array(), input_expect);
  ASSERT_EQ(output_tensor_map.array(), output_expect);
}

TEST_F(TestReduceSumInfo, DISABLED_GetForwardOp1) {
  Strategies inputs = {{4, 8, 1}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  reduce_sum->Init(strategy, nullptr);
  OperatorVector forward_op = reduce_sum->forward_op();
  size_t size = forward_op.size();

  ASSERT_EQ(size, 0);
}

TEST_F(TestReduceSumInfo, DISABLED_GetForwardOp2) {
  Strategies inputs = {{4, 4, 2}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  reduce_sum->Init(strategy, nullptr);
  OperatorVector forward_op = reduce_sum->forward_op();
  OperatorArgs operator_args = forward_op.at(0).second;
  OperatorAttrs operator_attrs = operator_args.first;

  std::string arg0_name = operator_attrs.at(0).first;
  ValuePtr arg0_value = operator_attrs.at(0).second;
  std::string op_value = arg0_value->cast<StringImmPtr>()->ToString();

  std::string arg1_name = operator_attrs.at(1).first;
  ValuePtr arg1_value = operator_attrs.at(1).second;
  std::string group_value = arg1_value->cast<StringImmPtr>()->ToString();

  ASSERT_EQ(forward_op.at(0).first, "AllReduce");
  ASSERT_EQ(forward_op.size(), 1);
  ASSERT_EQ(arg0_name, "op");
  ASSERT_EQ(op_value, "sum");
  ASSERT_EQ(arg1_name, "group");
}

TEST_F(TestReduceSumInfo, DISABLED_GetMirrorOPs1) {
  Strategies inputs = {{4, 8, 1}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  reduce_sum->Init(strategy, nullptr);
  MirrorOps mirror_ops = reduce_sum->mirror_ops();

  size_t size = mirror_ops.size();

  ASSERT_EQ(size, 0);
}

TEST_F(TestReduceSumInfo, DISABLED_GetMirrorOPs2) {
  Strategies inputs = {{4, 4, 1}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  reduce_sum->Init(strategy, nullptr);
  MirrorOps mirror_ops = reduce_sum->mirror_ops();
  OperatorVector mirror_op = mirror_ops.at(0);
  OperatorArgs operator_args = mirror_op.at(0).second;
  OperatorAttrs operator_attrs = operator_args.first;

  std::string arg0_name = operator_attrs.at(0).first;
  ValuePtr arg0_value = operator_attrs.at(0).second;
  std::string group = arg0_value->cast<StringImmPtr>()->ToString();

  ASSERT_EQ(mirror_op.at(0).first, "_MirrorOperator");
  ASSERT_EQ(mirror_op.size(), 1);
  ASSERT_EQ(arg0_name, "group");
}

TEST_F(TestReduceSumInfo, DISABLED_CheckStrategy1) {
  Strategies inputs = {{2, 2, 8, 16}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  Status ret = reduce_sum->Init(strategy, nullptr);
  ASSERT_EQ(ret, FAILED);
}

TEST_F(TestReduceSumInfo, DISABLED_CheckStrategy2) {
  Strategies inputs = {{2, 4, 8}, {2, 4, 8}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  Status ret = reduce_sum->Init(strategy, nullptr);
  ASSERT_EQ(ret, FAILED);
}

TEST_F(TestReduceSumInfo, DISABLED_CheckStrategy3) {
  Strategies inputs = {{4, 4, 2}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  Status ret = reduce_sum->Init(strategy, nullptr);
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(TestReduceSumInfo, DISABLED_CheckStrategy4) {
  Strategies inputs = {{4, 8, 1}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  Status ret = reduce_sum->Init(strategy, nullptr);
  ASSERT_EQ(ret, SUCCESS);
}
}  // namespace parallel
}  // namespace mindspore
