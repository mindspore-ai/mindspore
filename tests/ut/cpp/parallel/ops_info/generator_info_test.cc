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
#include "parallel/ops_info/generator_info.h"
#include "parallel/device_manager.h"
#include "parallel/step_parallel.h"

namespace mindspore {
namespace parallel {

class DropoutGenMaskInfo;
using DropoutGenMaskInfoPtr = std::shared_ptr<DropoutGenMaskInfo>;
DropoutGenMaskInfoPtr gen_mask;

class TestDropoutGenMaskInfo : public UT::Common {
 public:
  TestDropoutGenMaskInfo() {}
  void SetUp();
  void TearDown() {}
};

void TestDropoutGenMaskInfo::SetUp() {
  std::vector<int32_t> dev_list;

  for (int32_t i = 0; i < 10; i++) {
    dev_list.push_back(i);
  }

  std::vector<int32_t> stage_map;
  stage_map.push_back(8);
  stage_map.push_back(2);

  int32_t local_dev = 0;

  // create a new g_device_manager
  g_device_manager = std::make_shared<DeviceManager>();
  g_device_manager->Init(dev_list, local_dev, stage_map, "hccl");

  std::unordered_map<std::string, ValuePtr> attr;

  Shapes inputs_shape;
  Shapes outputs_shape = {{128}};
  std::vector<int> shape = {32, 128};
  ValuePtr val0 = MakeValue(shape);
  ValuePtr val1;
  std::vector<ValuePtr> val = {val0, val1};
  gen_mask = std::make_shared<DropoutGenMaskInfo>("gen_mask_info", inputs_shape, outputs_shape, attr);
  gen_mask->set_input_value(val);
}

TEST_F(TestDropoutGenMaskInfo, InferDevMatrixShape) {
  std::vector<Dimensions> stra = {{8, 1}};
  StrategyPtr strategy = NewStrategy(0, stra);

  gen_mask->Init(strategy);
  std::vector<int32_t> dev_matrix_shape = gen_mask->dev_matrix_shape();

  std::vector<int32_t> expect = {8, 1};
  ASSERT_EQ(dev_matrix_shape, expect);
}

TEST_F(TestDropoutGenMaskInfo, InferSliceShape) {
  std::vector<Dimensions> stra = {{8, 1}};
  StrategyPtr strategy = NewStrategy(0, stra);

  gen_mask->Init(strategy);
  std::vector<TensorInfo> outputs = gen_mask->outputs_tensor_info();

  Shape output_slice_shape_expect = {128};

  TensorInfo output_tensor_info = outputs.at(0);
  Shape output_slice_shape = output_tensor_info.slice_shape();

  ASSERT_EQ(output_slice_shape, output_slice_shape_expect);
}

TEST_F(TestDropoutGenMaskInfo, GetTensorLayout) {
  std::vector<Dimensions> stra = {{8, 1}};
  StrategyPtr strategy = NewStrategy(0, stra);

  gen_mask->Init(strategy);
  std::vector<TensorInfo> outputs = gen_mask->outputs_tensor_info();

  TensorMap output_map_expect = {-1};

  TensorInfo output_tensor_info = outputs.at(0);
  Map output_tensor_map = output_tensor_info.tensor_layout().origin_tensor_map();

  ASSERT_EQ(output_tensor_map.array(), output_map_expect);
}

TEST_F(TestDropoutGenMaskInfo, GetForwardOp) {
  std::vector<Dimensions> stra = {{8, 1}};
  StrategyPtr strategy = NewStrategy(0, stra);

  gen_mask->Init(strategy);
  OperatorVector forward_op = gen_mask->forward_op();
  size_t size = forward_op.size();

  ASSERT_EQ(size, 0);
}

TEST_F(TestDropoutGenMaskInfo, CheckStrategy1) {
  std::vector<Dimensions> stra = {{4, 8, 2}, {2, 3}};
  StrategyPtr strategy = NewStrategy(0, stra);

  Status ret = gen_mask->Init(strategy);
  ASSERT_EQ(ret, FAILED);
}

TEST_F(TestDropoutGenMaskInfo, CheckStrategy2) {
  std::vector<Dimensions> stra = {{8, 1}};
  StrategyPtr strategy = NewStrategy(0, stra);

  Status ret = gen_mask->Init(strategy);
  ASSERT_EQ(ret, SUCCESS);
}
}  // namespace parallel
}  // namespace mindspore
