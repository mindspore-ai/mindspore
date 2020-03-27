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
#include "optimizer/parallel/strategy.h"
#include "optimizer/parallel/ops_info/transpose_info.h"
#include "optimizer/parallel/device_manager.h"
#include "optimizer/parallel/step_parallel.h"

namespace mindspore {
namespace parallel {

class TransposeInfo;
using TransposeInfoPtr = std::shared_ptr<TransposeInfo>;
TransposeInfoPtr transpose;

class TestTransposeInfo : public UT::Common {
 public:
  TestTransposeInfo() {}
  void SetUp();
  void TearDown() {}
};

void TestTransposeInfo::SetUp() {
  std::list<int32_t> dev_list;

  for (int32_t i = 0; i < 34; i++) {
    dev_list.push_back(i);
  }

  std::list<int32_t> stage_map;
  stage_map.push_back(32);
  stage_map.push_back(2);

  int32_t local_dev = 0;

  // create a new g_device_manager
  g_device_manager = std::make_shared<DeviceManager>();
  g_device_manager->Init(dev_list, local_dev, stage_map, "hccl");

  std::unordered_map<std::string, ValuePtr> attr;

  Shapes inputs_shape = {{128, 64}};
  Shapes outputs_shape = {{64, 128}};
  std::vector<int> axis = {1, 0};
  ValuePtr val0;
  ValuePtr val1 = MakeValue(axis);
  std::vector<ValuePtr> val = {val0, val1};

  transpose = std::make_shared<TransposeInfo>("transpose_info", inputs_shape, outputs_shape, attr);
  transpose->set_input_value(val);
}

TEST_F(TestTransposeInfo, InferDevMatrixShape1) {
  std::vector<Dimensions> inputs = {{4, 8}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  transpose->Init(strategy);
  std::vector<int32_t> dev_matrix_shape = transpose->dev_matrix_shape();

  std::vector<int32_t> expect = {4, 8};
  ASSERT_EQ(dev_matrix_shape, expect);
}

TEST_F(TestTransposeInfo, InferDevMatrixShape2) {
  std::vector<Dimensions> inputs = {{4, 1}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  transpose->Init(strategy);
  std::vector<int32_t> dev_matrix_shape = transpose->dev_matrix_shape();

  std::vector<int32_t> expect = {8, 4, 1};
  ASSERT_EQ(dev_matrix_shape, expect);
}

TEST_F(TestTransposeInfo, InferSliceShape1) {
  std::vector<Dimensions> str = {{4, 8}};
  StrategyPtr strategy = NewStrategy(0, str);

  transpose->Init(strategy);
  std::vector<TensorInfo> inputs = transpose->inputs_tensor_info();
  std::vector<TensorInfo> outputs = transpose->outputs_tensor_info();

  Shape input_slice_shape_expect = {32, 8};
  Shape output_slice_shape_expect = {8, 32};

  TensorInfo input_tensor_info = inputs.at(0);
  TensorInfo output_tensor_info = outputs.at(0);

  Shape input_slice_shape = input_tensor_info.slice_shape();
  Shape output_slice_shape = output_tensor_info.slice_shape();

  ASSERT_EQ(input_slice_shape, input_slice_shape_expect);
  ASSERT_EQ(output_slice_shape, output_slice_shape_expect);
}

TEST_F(TestTransposeInfo, GetTensorLayout1) {
  std::vector<Dimensions> str = {{4, 8}};
  StrategyPtr strategy = NewStrategy(0, str);

  transpose->Init(strategy);
  std::vector<TensorInfo> inputs = transpose->inputs_tensor_info();
  std::vector<TensorInfo> outputs = transpose->outputs_tensor_info();

  TensorMap input_expect = {1, 0};
  TensorMap output_expect = {0, 1};

  TensorInfo input_tensor_info = inputs.at(0);
  TensorInfo output_tensor_info = outputs.at(0);

  Map input_tensor_map = input_tensor_info.tensor_layout().origin_tensor_map();
  Map output_tensor_map = output_tensor_info.tensor_layout().origin_tensor_map();

  ASSERT_EQ(input_tensor_map.array(), input_expect);
  ASSERT_EQ(output_tensor_map.array(), output_expect);
}

TEST_F(TestTransposeInfo, GetForwardOp1) {
  std::vector<Dimensions> inputs = {{4, 8}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  transpose->Init(strategy);
  OperatorVector forward_op = transpose->forward_op();
  size_t size = forward_op.size();

  ASSERT_EQ(size, 0);
}

TEST_F(TestTransposeInfo, GetMirrorOPs1) {
  std::vector<Dimensions> inputs = {{4, 8}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  transpose->Init(strategy);
  MirrorOps mirror_ops = transpose->mirror_ops();

  size_t size = mirror_ops.size();

  ASSERT_EQ(size, 0);
}

TEST_F(TestTransposeInfo, CheckStrategy1) {
  std::vector<Dimensions> inputs = {{1, 4, 8}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  Status ret = transpose->Init(strategy);
  ASSERT_EQ(ret, FAILED);
}

TEST_F(TestTransposeInfo, CheckStrategy2) {
  std::vector<Dimensions> inputs = {{2, 4, 8}, {2, 4, 8}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  Status ret = transpose->Init(strategy);
  ASSERT_EQ(ret, FAILED);
}

TEST_F(TestTransposeInfo, CheckStrategy3) {
  std::vector<Dimensions> inputs = {{4, 8}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  Status ret = transpose->Init(strategy);
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(TestTransposeInfo, AutoStrategy1) {
  ASSERT_EQ(transpose->GenerateStrategies(0), Status::SUCCESS);
  std::vector<std::shared_ptr<StrategyWithCost>> sc = transpose->GetStrategyCost();

  Shapes splittable_inputs = {{1, 1}};
  std::vector<StrategyPtr> sp_vector;
  Shapes inputs_shape = {{128, 64}};
  GenerateStrategiesForIndependentInputs(0, inputs_shape, splittable_inputs, &sp_vector);
  ASSERT_EQ(sc.size(), sp_vector.size());
}

}  // namespace parallel
}  // namespace mindspore
