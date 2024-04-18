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
#include "frontend/parallel/ops_info/onehot_info.h"
#include "frontend/parallel/device_manager.h"
#include "frontend/parallel/tensor_layout/tensor_redistribution.h"

namespace mindspore {
namespace parallel {

class OneHotInfo;
using OneHotInfoPtr = std::shared_ptr<OneHotInfo>;
OneHotInfoPtr onehot_info2;

class TestOneHotInfo2 : public UT::Common {
 public:
  TestOneHotInfo2() {}
  void SetUp();
  void TearDown() {}
};

void TestOneHotInfo2::SetUp() {
  RankList dev_list;

  for (int32_t i = 0; i < 10; i++) {
    dev_list.push_back(i);
  }

  RankList stage_map;
  stage_map.push_back(8);
  stage_map.push_back(2);

  int32_t local_dev = 0;

  // create a new g_device_manager
  g_device_manager = std::make_shared<DeviceManager>();
  g_device_manager->Init(dev_list, local_dev, stage_map, "hccl");

  ValuePtr axis = MakeValue(std::int64_t(0));
  mindspore::HashMap<std::string, ValuePtr> attr = {{"axis", axis}};

  Shapes inputs_shape = {{64}, {}, {}};
  Shapes outputs_shape = {{10, 64}};

  onehot_info2 = std::make_shared<OneHotInfo>("onehot_info", inputs_shape, outputs_shape, attr);
}

TEST_F(TestOneHotInfo2, DISABLED_InferDevMatrixShape1) {
  Strategies inputs = {{1, 8}, {}, {}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  Status status = onehot_info2->Init(strategy, nullptr);
  ASSERT_EQ(status, SUCCESS);
  Shape dev_matrix_shape = onehot_info2->dev_matrix_shape();

  Shape expect = {8, 1};
  ASSERT_EQ(dev_matrix_shape, expect);
}

TEST_F(TestOneHotInfo2, DISABLED_InferDevMatrixShape2) {
  Strategies inputs = {{1, 4}, {}, {}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  Status status = onehot_info2->Init(strategy, nullptr);
  ASSERT_EQ(status, SUCCESS);
  Shape dev_matrix_shape = onehot_info2->dev_matrix_shape();

  Shape expect = {4, 1, 2};
  ASSERT_EQ(dev_matrix_shape, expect);
}

TEST_F(TestOneHotInfo2, DISABLED_InferDevMatrixShape3) {
  Strategies inputs = {{2, 4}, {}, {}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  Status status = onehot_info2->Init(strategy, nullptr);
  ASSERT_EQ(status, SUCCESS);
  Shape dev_matrix_shape = onehot_info2->dev_matrix_shape();

  Shape expect = {4, 2};
  ASSERT_EQ(dev_matrix_shape, expect);
}

TEST_F(TestOneHotInfo2, DISABLED_InferTensorMap2) {
  Strategies str = {{1, 8}, {}, {}};
  StrategyPtr strategy = NewStrategy(0, str);

  Status status = onehot_info2->Init(strategy, nullptr);
  ASSERT_EQ(status, SUCCESS);
  std::vector<TensorInfo> inputs = onehot_info2->inputs_tensor_info();
  std::vector<TensorInfo> outputs = onehot_info2->outputs_tensor_info();

  TensorMap input_expect = {1};
  TensorMap output_expect = {0, 1};

  TensorInfo input_tensor_info = inputs.at(0);
  TensorInfo output_tensor_info = outputs.at(0);

  Map input_tensor_map = input_tensor_info.tensor_layout().origin_tensor_map();
  Map output_tensor_map = output_tensor_info.tensor_layout().origin_tensor_map();

  ASSERT_EQ(input_tensor_map.array(), input_expect);
  ASSERT_EQ(output_tensor_map.array(), output_expect);
}

TEST_F(TestOneHotInfo2, DISABLED_InferSliceShape1) {
  Strategies str = {{1, 8}, {}, {}};
  StrategyPtr strategy = NewStrategy(0, str);

  Status status = onehot_info2->Init(strategy, nullptr);
  ASSERT_EQ(status, SUCCESS);
  std::vector<TensorInfo> inputs = onehot_info2->inputs_tensor_info();
  std::vector<TensorInfo> outputs = onehot_info2->outputs_tensor_info();

  Shape input_slice_shape_expect = {8};
  Shape output_slice_shape_expect = {10, 8};

  TensorInfo input_tensor_info = inputs.at(0);
  TensorInfo output_tensor_info = outputs.at(0);

  Shape input_slice_shape = input_tensor_info.slice_shape();
  Shape output_slice_shape = output_tensor_info.slice_shape();

  ASSERT_EQ(input_slice_shape, input_slice_shape_expect);
  ASSERT_EQ(output_slice_shape, output_slice_shape_expect);
}

TEST_F(TestOneHotInfo2, DISABLED_InferSliceShape2) {
  Strategies str = {{2, 4}, {}, {}};
  StrategyPtr strategy = NewStrategy(0, str);

  Status status = onehot_info2->Init(strategy, nullptr);
  ASSERT_EQ(status, SUCCESS);
  std::vector<TensorInfo> inputs = onehot_info2->inputs_tensor_info();
  std::vector<TensorInfo> outputs = onehot_info2->outputs_tensor_info();

  Shape input_slice_shape_expect = {16};
  Shape output_slice_shape_expect = {5, 16};

  TensorInfo input_tensor_info = inputs.at(0);
  TensorInfo output_tensor_info = outputs.at(0);

  Shape input_slice_shape = input_tensor_info.slice_shape();
  Shape output_slice_shape = output_tensor_info.slice_shape();

  ASSERT_EQ(input_slice_shape, input_slice_shape_expect);
  ASSERT_EQ(output_slice_shape, output_slice_shape_expect);
}

TEST_F(TestOneHotInfo2, DISABLED_InferSliceShape3) {
  Strategies str = {{2, 2}, {}, {}};
  StrategyPtr strategy = NewStrategy(0, str);

  Status status = onehot_info2->Init(strategy, nullptr);
  ASSERT_EQ(status, SUCCESS);
  std::vector<TensorInfo> inputs = onehot_info2->inputs_tensor_info();
  std::vector<TensorInfo> outputs = onehot_info2->outputs_tensor_info();

  Shape input_slice_shape_expect = {32};
  Shape output_slice_shape_expect = {5, 32};

  TensorInfo input_tensor_info = inputs.at(0);
  TensorInfo output_tensor_info = outputs.at(0);

  Shape input_slice_shape = input_tensor_info.slice_shape();
  Shape output_slice_shape = output_tensor_info.slice_shape();

  ASSERT_EQ(input_slice_shape, input_slice_shape_expect);
  ASSERT_EQ(output_slice_shape, output_slice_shape_expect);
}
}  // namespace parallel
}  // namespace mindspore
