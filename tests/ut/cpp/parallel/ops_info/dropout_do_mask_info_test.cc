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
#include "parallel/strategy.h"
#include "parallel/ops_info/dropout_do_mask_info.h"
#include "parallel/device_manager.h"
#include "parallel/step_parallel.h"

namespace mindspore {
namespace parallel {

class DropoutDoMaskInfo;
using DropoutDoMaskInfoPtr = std::shared_ptr<DropoutDoMaskInfo>;
DropoutDoMaskInfoPtr do_mask;

class TestDropoutDoMaskInfo : public UT::Common {
 public:
  TestDropoutDoMaskInfo() {}
  void SetUp();
  void TearDown() {}
};

void TestDropoutDoMaskInfo::SetUp() {
  std::vector<int32_t> dev_list;

  for (int32_t i = 0; i < 34; i++) {
    dev_list.push_back(i);
  }

  std::vector<int32_t> stage_map;
  stage_map.push_back(32);
  stage_map.push_back(2);

  int32_t local_dev = 0;

  // create a new g_device_manager
  g_device_manager = std::make_shared<DeviceManager>();
  g_device_manager->Init(dev_list, local_dev, stage_map, "hccl");

  std::unordered_map<std::string, ValuePtr> attr;

  Shapes inputs_shape = {{32, 128}, {64}, {}};
  Shapes outputs_shape = {{32, 128}};
  do_mask = std::make_shared<DropoutDoMaskInfo>("do_mask_info", inputs_shape, outputs_shape, attr);
}

TEST_F(TestDropoutDoMaskInfo, InferDevMatrixShape) {
  std::vector<Dimensions> stra = {{4, 8}};
  StrategyPtr strategy = NewStrategy(0, stra);

  do_mask->Init(strategy);
  std::vector<int32_t> dev_matrix_shape = do_mask->dev_matrix_shape();

  std::vector<int32_t> expect = {4, 8};
  ASSERT_EQ(dev_matrix_shape, expect);
}

TEST_F(TestDropoutDoMaskInfo, InferSliceShape) {
  std::vector<Dimensions> stra = {{4, 8}};
  StrategyPtr strategy = NewStrategy(0, stra);

  do_mask->Init(strategy);
  std::vector<TensorInfo> inputs = do_mask->inputs_tensor_info();
  std::vector<TensorInfo> outputs = do_mask->outputs_tensor_info();

  Shape input_a_slice_shape_expect = {8, 16};
  Shape input_b_slice_shape_expect = {64};
  Shape output_slice_shape_expect = {8, 16};

  TensorInfo input_a_tensor_info = inputs.at(0);
  TensorInfo input_b_tensor_info = inputs.at(1);
  TensorInfo output_tensor_info = outputs.at(0);
  Shape input_a_slice_shape = input_a_tensor_info.slice_shape();
  Shape input_b_slice_shape = input_b_tensor_info.slice_shape();
  Shape output_slice_shape = output_tensor_info.slice_shape();

  ASSERT_EQ(input_a_slice_shape, input_a_slice_shape_expect);
  ASSERT_EQ(input_b_slice_shape, input_b_slice_shape_expect);
  ASSERT_EQ(output_slice_shape, output_slice_shape_expect);
}

TEST_F(TestDropoutDoMaskInfo, GetTensorLayout) {
  std::vector<Dimensions> stra = {{4, 8}};
  StrategyPtr strategy = NewStrategy(0, stra);

  do_mask->Init(strategy);
  std::vector<TensorInfo> inputs = do_mask->inputs_tensor_info();
  std::vector<TensorInfo> outputs = do_mask->outputs_tensor_info();

  TensorMap input_a_map_expect = {1, 0};
  TensorMap input_b_map_expect = {-1};
  TensorMap output_map_expect = {1, 0};

  TensorInfo input_a_tensor_info = inputs.at(0);
  TensorInfo input_b_tensor_info = inputs.at(1);
  TensorInfo output_tensor_info = outputs.at(0);
  Map input_a_tensor_map = input_a_tensor_info.tensor_layout().origin_tensor_map();
  Map input_b_tensor_map = input_b_tensor_info.tensor_layout().origin_tensor_map();
  Map output_tensor_map = output_tensor_info.tensor_layout().origin_tensor_map();

  ASSERT_EQ(input_a_tensor_map.array(), input_a_map_expect);
  ASSERT_EQ(input_b_tensor_map.array(), input_b_map_expect);
  ASSERT_EQ(output_tensor_map.array(), output_map_expect);
}

TEST_F(TestDropoutDoMaskInfo, GetForwardOp) {
  std::vector<Dimensions> stra = {{4, 8}};
  StrategyPtr strategy = NewStrategy(0, stra);

  do_mask->Init(strategy);
  OperatorVector forward_op = do_mask->forward_op();
  size_t size = forward_op.size();

  ASSERT_EQ(size, 0);
}

TEST_F(TestDropoutDoMaskInfo, CheckStrategy1) {
  std::vector<Dimensions> stra = {{4, 8, 2}};
  StrategyPtr strategy = NewStrategy(0, stra);

  Status ret = do_mask->Init(strategy);
  ASSERT_EQ(ret, FAILED);
}

TEST_F(TestDropoutDoMaskInfo, CheckStrategy2) {
  std::vector<Dimensions> stra = {{8, 8}};
  StrategyPtr strategy = NewStrategy(0, stra);

  Status ret = do_mask->Init(strategy);
  ASSERT_EQ(ret, FAILED);
}

TEST_F(TestDropoutDoMaskInfo, CheckStrategy3) {
  std::vector<Dimensions> stra = {{4, 8}, {4, 8}};
  StrategyPtr strategy = NewStrategy(0, stra);

  Status ret = do_mask->Init(strategy);
  ASSERT_EQ(ret, FAILED);
}

TEST_F(TestDropoutDoMaskInfo, CheckStrategy4) {
  std::vector<Dimensions> stra = {{4, 8}};
  StrategyPtr strategy = NewStrategy(0, stra);

  Status ret = do_mask->Init(strategy);
  ASSERT_EQ(ret, SUCCESS);
}
}  // namespace parallel
}  // namespace mindspore
