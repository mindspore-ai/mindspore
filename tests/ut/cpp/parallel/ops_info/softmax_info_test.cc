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

class SoftmaxInfo;
using SoftmaxInfoPtr = std::shared_ptr<SoftmaxInfo>;
SoftmaxInfoPtr softmax;
SoftmaxInfoPtr softmax2;

class TestSoftmaxInfo : public UT::Common {
 public:
  TestSoftmaxInfo() {}
  void SetUp();
  void TearDown() {}
};

void TestSoftmaxInfo::SetUp() {
  RankList dev_list;

  for (int32_t i = 0; i < 130; i++) {
    dev_list.push_back(i);
  }

  RankList stage_map;
  stage_map.push_back(128);
  stage_map.push_back(2);

  int32_t local_dev = 0;

  g_device_manager = std::make_shared<DeviceManager>();
  g_device_manager->Init(dev_list, local_dev, stage_map, "hccl");

  ValuePtr axis1 = MakeValue(static_cast<int64_t>(-2));
  mindspore::HashMap<std::string, ValuePtr> attr1 = {{"axis", axis1}};

  ValuePtr axis2 = MakeValue(static_cast<int64_t>(4));
  mindspore::HashMap<std::string, ValuePtr> attr2 = {{"axis", axis2}};

  Shapes inputs_shape = {{2, 4, 8, 16}};
  Shapes outputs_shape = {{2, 4, 8, 16}};

  softmax = std::make_shared<SoftmaxInfo>("softmax_info1", inputs_shape, outputs_shape, attr1);
  softmax2 = std::make_shared<SoftmaxInfo>("softmax_info2", inputs_shape, outputs_shape, attr2);
}

TEST_F(TestSoftmaxInfo, DISABLED_InferDevMatrixShape1) {
  Strategies inputs = {{2, 4, 1, 16}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  softmax->Init(strategy, nullptr);
  Shape dev_matrix_shape = softmax->dev_matrix_shape();

  Shape expect = {2, 4, 1, 16};
  ASSERT_EQ(dev_matrix_shape, expect);
}

TEST_F(TestSoftmaxInfo, DISABLED_InferSliceShape1) {
  Strategies str = {{2, 4, 1, 16}};
  StrategyPtr strategy = NewStrategy(0, str);

  softmax->Init(strategy, nullptr);
  std::vector<TensorInfo> inputs = softmax->inputs_tensor_info();
  std::vector<TensorInfo> outputs = softmax->outputs_tensor_info();

  Shape input_slice_shape_expect = {1, 1, 8, 1};
  Shape output_slice_shape_expect = {1, 1, 8, 1};

  TensorInfo input_tensor_info = inputs.at(0);
  TensorInfo output_tensor_info = outputs.at(0);

  Shape input_slice_shape = input_tensor_info.slice_shape();
  Shape output_slice_shape = output_tensor_info.slice_shape();

  ASSERT_EQ(input_slice_shape, input_slice_shape_expect);
  ASSERT_EQ(output_slice_shape, output_slice_shape_expect);
}

TEST_F(TestSoftmaxInfo, DISABLED_GetTensorLayout1) {
  Strategies str = {{2, 4, 1, 16}};
  StrategyPtr strategy = NewStrategy(0, str);

  softmax->Init(strategy, nullptr);
  std::vector<TensorInfo> inputs = softmax->inputs_tensor_info();
  std::vector<TensorInfo> outputs = softmax->outputs_tensor_info();

  TensorMap input_expect = {3, 2, 1, 0};
  TensorMap output_expect = {3, 2, 1, 0};

  TensorInfo input_tensor_info = inputs.at(0);
  TensorInfo output_tensor_info = outputs.at(0);

  Map input_tensor_map = input_tensor_info.tensor_layout().origin_tensor_map();
  Map output_tensor_map = output_tensor_info.tensor_layout().origin_tensor_map();

  ASSERT_EQ(input_tensor_map.array(), input_expect);
  ASSERT_EQ(output_tensor_map.array(), output_expect);
}

TEST_F(TestSoftmaxInfo, DISABLED_GetForwardOp1) {
  Strategies inputs = {{2, 4, 1, 16}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  softmax->Init(strategy, nullptr);
  OperatorVector forward_op = softmax->forward_op();
  size_t size = forward_op.size();

  ASSERT_EQ(size, 0);
}

TEST_F(TestSoftmaxInfo, DISABLED_GetMirrorOPs1) {
  Strategies inputs = {{2, 4, 1, 16}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  softmax->Init(strategy, nullptr);
  MirrorOps mirror_ops = softmax->mirror_ops();

  size_t size = mirror_ops.size();

  ASSERT_EQ(size, 0);
}

TEST_F(TestSoftmaxInfo, DISABLED_CheckStrategy1) {
  // Success: {{2,4,1,16}}
  Strategies inputs = {{2, 2, 8, 16}, {2, 4, 16, 1}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  Status ret = softmax->Init(strategy, nullptr);
  ASSERT_EQ(ret, FAILED);
}

TEST_F(TestSoftmaxInfo, DISABLED_CheckStrategy2) {
  // Success: {{2,4,1,16}}
  Strategies inputs = {{2, 4, 8}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  Status ret = softmax->Init(strategy, nullptr);
  ASSERT_EQ(ret, FAILED);
}

TEST_F(TestSoftmaxInfo, DISABLED_CheckStrategy3) {
  // Success: {{2,4,1,16}}
  Strategies inputs = {{2, 4, 8, 16}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  Status ret = softmax->Init(strategy, nullptr);
  ASSERT_EQ(ret, FAILED);
}

TEST_F(TestSoftmaxInfo, DISABLED_InitFailed1) {
  // softmax2's axis is wrong
  Strategies inputs = {{2, 4, 1, 16}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  Status ret = softmax2->Init(strategy, nullptr);
  ASSERT_EQ(ret, FAILED);
}

TEST_F(TestSoftmaxInfo, DISABLED_InitFailed2) {
  // dev num is wrong
  Strategies inputs = {{2, 4, 1, 100}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  Status ret = softmax2->Init(strategy, nullptr);
  ASSERT_EQ(ret, FAILED);
}

}  // namespace parallel
}  // namespace mindspore
