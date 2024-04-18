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
#include "frontend/parallel/ops_info/reshape_info.h"
#include "frontend/parallel/device_manager.h"
#include "frontend/parallel/step_parallel.h"

namespace mindspore {
namespace parallel {

class ReshapeInfo;
using ReshapeInfoPtr = std::shared_ptr<ReshapeInfo>;
ReshapeInfoPtr reshape;

class TestReshapeInfo : public UT::Common {
 public:
  TestReshapeInfo() {}
  void SetUp();
  void TearDown() {}
};

void TestReshapeInfo::SetUp() {
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

  mindspore::HashMap<std::string, ValuePtr> attr;

  Shapes inputs_shape = {{32, 512, 7, 7}};
  Shapes outputs_shape = {{32, 25088}};
  std::vector<int64_t> axis = {32, 25088};
  ValuePtr val0;
  ValuePtr val1 = MakeValue(axis);
  std::vector<ValuePtr> val = {val0, val1};

  reshape = std::make_shared<ReshapeInfo>("reshape_info", inputs_shape, outputs_shape, attr);
  reshape->set_input_value(val);
}

TEST_F(TestReshapeInfo, DISABLED_InferDevMatrixShape1) {
  Strategies inputs = {{4, 1, 1, 1}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  reshape->Init(strategy, nullptr);
  Shape dev_matrix_shape = reshape->dev_matrix_shape();

  Shape expect = {4, 1, 1, 1, 8};
  ASSERT_EQ(dev_matrix_shape, expect);
}

TEST_F(TestReshapeInfo, DISABLED_InferDevMatrixShape2) {
  Strategies inputs = {{32, 1, 1, 1}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  reshape->Init(strategy, nullptr);
  Shape dev_matrix_shape = reshape->dev_matrix_shape();

  Shape expect = {32, 1, 1, 1};
  ASSERT_EQ(dev_matrix_shape, expect);
}

TEST_F(TestReshapeInfo, DISABLED_InferSliceShape1) {
  Strategies str = {{4, 1, 1, 1}};
  StrategyPtr strategy = NewStrategy(0, str);

  reshape->Init(strategy, nullptr);
  std::vector<TensorInfo> inputs = reshape->inputs_tensor_info();
  std::vector<TensorInfo> outputs = reshape->outputs_tensor_info();

  Shape input_slice_shape_expect = {8, 512, 7, 7};
  Shape output_slice_shape_expect = {32, 25088};

  TensorInfo input_tensor_info = inputs.at(0);
  TensorInfo output_tensor_info = outputs.at(0);

  Shape input_slice_shape = input_tensor_info.slice_shape();
  Shape output_slice_shape = output_tensor_info.slice_shape();

  ASSERT_EQ(input_slice_shape, input_slice_shape_expect);
  ASSERT_EQ(output_slice_shape, output_slice_shape_expect);
}

TEST_F(TestReshapeInfo, DISABLED_InferSliceShape2) {
  Strategies str = {{32, 1, 1, 1}};
  StrategyPtr strategy = NewStrategy(0, str);

  reshape->Init(strategy, nullptr);
  std::vector<TensorInfo> inputs = reshape->inputs_tensor_info();
  std::vector<TensorInfo> outputs = reshape->outputs_tensor_info();

  Shape input_slice_shape_expect = {1, 512, 7, 7};
  Shape output_slice_shape_expect = {32, 25088};

  TensorInfo input_tensor_info = inputs.at(0);
  TensorInfo output_tensor_info = outputs.at(0);

  Shape input_slice_shape = input_tensor_info.slice_shape();
  Shape output_slice_shape = output_tensor_info.slice_shape();

  ASSERT_EQ(input_slice_shape, input_slice_shape_expect);
  ASSERT_EQ(output_slice_shape, output_slice_shape_expect);
}

TEST_F(TestReshapeInfo, DISABLED_GetTensorLayout1) {
  Strategies str = {{4, 1, 1, 1}};
  StrategyPtr strategy = NewStrategy(0, str);

  reshape->Init(strategy, nullptr);
  std::vector<TensorInfo> inputs = reshape->inputs_tensor_info();
  std::vector<TensorInfo> outputs = reshape->outputs_tensor_info();

  TensorMap input_expect = {4, 3, 2, 1};
  TensorMap output_expect = {-1, -1};

  TensorInfo input_tensor_info = inputs.at(0);
  TensorInfo output_tensor_info = outputs.at(0);

  Map input_tensor_map = input_tensor_info.tensor_layout().origin_tensor_map();
  Map output_tensor_map = output_tensor_info.tensor_layout().origin_tensor_map();

  ASSERT_EQ(input_tensor_map.array(), input_expect);
  ASSERT_EQ(output_tensor_map.array(), output_expect);
}

TEST_F(TestReshapeInfo, DISABLED_GetTensorLayout2) {
  Strategies str = {{32, 1, 1, 1}};
  StrategyPtr strategy = NewStrategy(0, str);

  reshape->Init(strategy, nullptr);
  std::vector<TensorInfo> inputs = reshape->inputs_tensor_info();
  std::vector<TensorInfo> outputs = reshape->outputs_tensor_info();

  TensorMap input_expect = {3, 2, 1, 0};
  TensorMap output_expect = {-1, -1};

  TensorInfo input_tensor_info = inputs.at(0);
  TensorInfo output_tensor_info = outputs.at(0);

  Map input_tensor_map = input_tensor_info.tensor_layout().origin_tensor_map();
  Map output_tensor_map = output_tensor_info.tensor_layout().origin_tensor_map();

  ASSERT_EQ(input_tensor_map.array(), input_expect);
  ASSERT_EQ(output_tensor_map.array(), output_expect);
}

TEST_F(TestReshapeInfo, DISABLED_GetForwardOp1) {
  Strategies inputs = {{4, 1, 1, 1}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  reshape->Init(strategy, nullptr);
  OperatorVector forward_op = reshape->forward_op();
  size_t size = forward_op.size();

  ASSERT_EQ(size, 0);
}

TEST_F(TestReshapeInfo, DISABLED_GetMirrorOPs1) {
  Strategies inputs = {{4, 1, 1, 1}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  reshape->Init(strategy, nullptr);
  MirrorOps mirror_ops = reshape->mirror_ops();

  size_t size = mirror_ops.size();

  ASSERT_EQ(size, 2);
}

TEST_F(TestReshapeInfo, DISABLED_CheckStrategy1) {
  Strategies inputs = {{1, 4, 8}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  Status ret = reshape->Init(strategy, nullptr);
  ASSERT_EQ(ret, FAILED);
}

TEST_F(TestReshapeInfo, DISABLED_CheckStrategy2) {
  Strategies inputs = {{2, 4, 8}, {2, 4, 8}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  Status ret = reshape->Init(strategy, nullptr);
  ASSERT_EQ(ret, FAILED);
}

TEST_F(TestReshapeInfo, DISABLED_CheckStrategy3) {
  Strategies inputs = {{4, 1, 1, 1}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  Status ret = reshape->Init(strategy, nullptr);
  ASSERT_EQ(ret, SUCCESS);
}
}  // namespace parallel
}  // namespace mindspore
