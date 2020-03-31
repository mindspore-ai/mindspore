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
#include "parallel/ops_info/onehot_info.h"
#include "parallel/device_manager.h"
#include "parallel/tensor_layout/tensor_redistribution.h"

namespace mindspore {
namespace parallel {

class OneHotInfo;
using OneHotInfoPtr = std::shared_ptr<OneHotInfo>;
OneHotInfoPtr onehot_info;

class TestOneHotInfo : public UT::Common {
 public:
  TestOneHotInfo() {}
  void SetUp();
  void TearDown() {}
};

void TestOneHotInfo::SetUp() {
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

  ValuePtr axis = MakeValue(std::int32_t(-1));
  std::unordered_map<std::string, ValuePtr> attr = {{"axis", axis}};

  Shapes inputs_shape = {{64}, {}, {}};
  Shapes outputs_shape = {{64, 10}};

  onehot_info = std::make_shared<OneHotInfo>("onehot_info", inputs_shape, outputs_shape, attr);
}

TEST_F(TestOneHotInfo, InferDevMatrixShape1) {
  std::vector<Dimensions> inputs = {{8, 1}, {}, {}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  Status status = onehot_info->Init(strategy);
  ASSERT_EQ(status, SUCCESS);
  std::vector<int32_t> dev_matrix_shape = onehot_info->dev_matrix_shape();

  std::vector<int32_t> expect = {8, 1};
  ASSERT_EQ(dev_matrix_shape, expect);
}

TEST_F(TestOneHotInfo, InferDevMatrixShape2) {
  std::vector<Dimensions> inputs = {{4, 1}, {}, {}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  Status status = onehot_info->Init(strategy);
  ASSERT_EQ(status, SUCCESS);
  std::vector<int32_t> dev_matrix_shape = onehot_info->dev_matrix_shape();

  std::vector<int32_t> expect = {2, 4, 1};
  ASSERT_EQ(dev_matrix_shape, expect);
}

TEST_F(TestOneHotInfo, InferDevMatrixShape3) {
  std::vector<Dimensions> inputs = {{4, 2}, {}, {}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  Status status = onehot_info->Init(strategy);
  ASSERT_EQ(status, FAILED);
  std::vector<int32_t> dev_matrix_shape = onehot_info->dev_matrix_shape();

  std::vector<int32_t> expect = {4, 2};
  ASSERT_EQ(dev_matrix_shape, expect);
}

TEST_F(TestOneHotInfo, InferTensorMap2) {
  std::vector<Dimensions> str = {{8, 1}, {}, {}};
  StrategyPtr strategy = NewStrategy(0, str);

  Status status = onehot_info->Init(strategy);
  ASSERT_EQ(status, SUCCESS);
  std::vector<TensorInfo> inputs = onehot_info->inputs_tensor_info();
  std::vector<TensorInfo> outputs = onehot_info->outputs_tensor_info();

  TensorMap input_expect = {1};
  TensorMap output_expect = {1, 0};

  TensorInfo input_tensor_info = inputs.at(0);
  TensorInfo output_tensor_info = outputs.at(0);

  Map input_tensor_map = input_tensor_info.tensor_layout().origin_tensor_map();
  Map output_tensor_map = output_tensor_info.tensor_layout().origin_tensor_map();

  ASSERT_EQ(input_tensor_map.array(), input_expect);
  ASSERT_EQ(output_tensor_map.array(), output_expect);
}

TEST_F(TestOneHotInfo, InferSliceShape1) {
  std::vector<Dimensions> str = {{8, 1}, {}, {}};
  StrategyPtr strategy = NewStrategy(0, str);

  Status status = onehot_info->Init(strategy);
  ASSERT_EQ(status, SUCCESS);
  std::vector<TensorInfo> inputs = onehot_info->inputs_tensor_info();
  std::vector<TensorInfo> outputs = onehot_info->outputs_tensor_info();

  Shape input_slice_shape_expect = {8};
  Shape output_slice_shape_expect = {8, 10};

  TensorInfo input_tensor_info = inputs.at(0);
  TensorInfo output_tensor_info = outputs.at(0);

  Shape input_slice_shape = input_tensor_info.slice_shape();
  Shape output_slice_shape = output_tensor_info.slice_shape();

  ASSERT_EQ(input_slice_shape, input_slice_shape_expect);
  ASSERT_EQ(output_slice_shape, output_slice_shape_expect);
}

TEST_F(TestOneHotInfo, InferSliceShape2) {
  std::vector<Dimensions> str = {{4, 2}, {}, {}};
  StrategyPtr strategy = NewStrategy(0, str);

  Status status = onehot_info->Init(strategy);
  ASSERT_EQ(status, FAILED);
  std::vector<TensorInfo> inputs = onehot_info->inputs_tensor_info();
  std::vector<TensorInfo> outputs = onehot_info->outputs_tensor_info();

  Shape input_slice_shape_expect = {16};
  Shape output_slice_shape_expect = {16, 5};

  TensorInfo input_tensor_info = inputs.at(0);
  TensorInfo output_tensor_info = outputs.at(0);

  Shape input_slice_shape = input_tensor_info.slice_shape();
  Shape output_slice_shape = output_tensor_info.slice_shape();

  ASSERT_EQ(input_slice_shape, input_slice_shape_expect);
  ASSERT_EQ(output_slice_shape, output_slice_shape_expect);
}

TEST_F(TestOneHotInfo, InferSliceShape3) {
  std::vector<Dimensions> str = {{2, 2}, {}, {}};
  StrategyPtr strategy = NewStrategy(0, str);

  Status status = onehot_info->Init(strategy);
  ASSERT_EQ(status, FAILED);
  std::vector<TensorInfo> inputs = onehot_info->inputs_tensor_info();
  std::vector<TensorInfo> outputs = onehot_info->outputs_tensor_info();

  Shape input_slice_shape_expect = {32};
  Shape output_slice_shape_expect = {32, 5};

  TensorInfo input_tensor_info = inputs.at(0);
  TensorInfo output_tensor_info = outputs.at(0);

  Shape input_slice_shape = input_tensor_info.slice_shape();
  Shape output_slice_shape = output_tensor_info.slice_shape();

  ASSERT_EQ(input_slice_shape, input_slice_shape_expect);
  ASSERT_EQ(output_slice_shape, output_slice_shape_expect);
}

TEST_F(TestOneHotInfo, GetMirrorOPs1) {
  std::vector<Dimensions> inputs = {{8, 1}, {}, {}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  Status status = onehot_info->Init(strategy);
  ASSERT_EQ(status, SUCCESS);
  MirrorOps mirror_ops = onehot_info->mirror_ops();

  ASSERT_EQ(mirror_ops.size(), 0);
}

TEST_F(TestOneHotInfo, CheckStrategy1) {
  std::vector<Dimensions> inputs = {{16}, {}, {}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  Status ret = onehot_info->Init(strategy);
  ASSERT_EQ(ret, FAILED);
}
}  // namespace parallel
}  // namespace mindspore
