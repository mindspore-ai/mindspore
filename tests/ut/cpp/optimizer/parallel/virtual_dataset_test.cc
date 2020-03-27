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
#include "optimizer/parallel/ops_info/virtual_dataset_info.h"
#include "optimizer/parallel/device_manager.h"
#include "optimizer/parallel/step_parallel.h"

namespace mindspore {
namespace parallel {

class VirtualDatasetInfo;
using VirtualDatasetInfoPtr = std::shared_ptr<VirtualDatasetInfo>;
VirtualDatasetInfoPtr virtual_dataset;

class TestVirtualDatasetInfo : public UT::Common {
 public:
  TestVirtualDatasetInfo() {}
  void SetUp();
  void TearDown() {}
};

void TestVirtualDatasetInfo::SetUp() {
  std::list<int32_t> dev_list;

  for (int32_t i = 0; i < 130; i++) {
    dev_list.push_back(i);
  }

  std::list<int32_t> stage_map;
  stage_map.push_back(16);
  stage_map.push_back(114);

  int32_t local_dev = 0;

  // create a new g_device_manager
  g_device_manager = std::make_shared<DeviceManager>();
  g_device_manager->Init(dev_list, local_dev, stage_map, "hccl");

  std::unordered_map<std::string, ValuePtr> attr;

  Shapes inputs_shape = {{128, 32}, {1280, 320}, {12800, 3200}};
  Shapes outputs_shape = {{128, 32}, {1280, 320}, {12800, 3200}};

  virtual_dataset = std::make_shared<VirtualDatasetInfo>("virtual_dataset_info", inputs_shape, outputs_shape, attr);
}

TEST_F(TestVirtualDatasetInfo, InferDevMatrixShape1) {
  std::vector<Dimensions> inputs = {{16, 1}, {16, 1}, {16, 1}};
  StrategyPtr strategy = NewStrategy(0, inputs);
  virtual_dataset->Init(strategy);
  std::vector<int32_t> dev_matrix_shape = virtual_dataset->dev_matrix_shape();

  std::vector<int32_t> expect = {16};
  ASSERT_EQ(dev_matrix_shape, expect);
}

TEST_F(TestVirtualDatasetInfo, InferDevMatrixShape2) {
  std::vector<Dimensions> inputs = {{8, 1}, {8, 1}, {8, 1}};
  StrategyPtr strategy = NewStrategy(0, inputs);
  virtual_dataset->Init(strategy);
  std::vector<int32_t> dev_matrix_shape = virtual_dataset->dev_matrix_shape();

  std::vector<int32_t> expect = {8, 2};
  ASSERT_EQ(dev_matrix_shape, expect);
}

TEST_F(TestVirtualDatasetInfo, InferSliceShape1) {
  std::vector<Dimensions> str = {{8, 1}, {8, 1}, {8, 1}};
  StrategyPtr strategy = NewStrategy(0, str);

  virtual_dataset->Init(strategy);
  std::vector<TensorInfo> inputs = virtual_dataset->inputs_tensor_info();
  std::vector<TensorInfo> outputs = virtual_dataset->outputs_tensor_info();

  Shape input_slice_shape_expect = {16, 32};
  Shape output_slice_shape_expect = {16, 32};

  TensorInfo input_tensor_info = inputs.at(0);
  TensorInfo output_tensor_info = outputs.at(0);

  Shape input_slice_shape = input_tensor_info.slice_shape();
  Shape output_slice_shape = output_tensor_info.slice_shape();

  ASSERT_EQ(input_slice_shape, input_slice_shape_expect);
  ASSERT_EQ(output_slice_shape, output_slice_shape_expect);

  Shape input_slice_shape_expect1 = {160, 320};
  Shape output_slice_shape_expect1 = {160, 320};

  TensorInfo input_tensor_info1 = inputs.at(1);
  TensorInfo output_tensor_info1 = outputs.at(1);

  Shape input_slice_shape1 = input_tensor_info1.slice_shape();
  Shape output_slice_shape1 = output_tensor_info1.slice_shape();

  ASSERT_EQ(input_slice_shape1, input_slice_shape_expect1);
  ASSERT_EQ(output_slice_shape1, output_slice_shape_expect1);

  Shape input_slice_shape_expect2 = {1600, 3200};
  Shape output_slice_shape_expect2 = {1600, 3200};

  TensorInfo input_tensor_info2 = inputs.at(2);
  TensorInfo output_tensor_info2 = outputs.at(2);

  Shape input_slice_shape2 = input_tensor_info2.slice_shape();
  Shape output_slice_shape2 = output_tensor_info2.slice_shape();

  ASSERT_EQ(input_slice_shape2, input_slice_shape_expect2);
  ASSERT_EQ(output_slice_shape2, output_slice_shape_expect2);
}

TEST_F(TestVirtualDatasetInfo, GetTensorLayout1) {
  std::vector<Dimensions> str = {{8, 1}, {8, 1}, {8, 1}};
  StrategyPtr strategy = NewStrategy(0, str);

  virtual_dataset->Init(strategy);
  std::vector<TensorInfo> inputs = virtual_dataset->inputs_tensor_info();
  std::vector<TensorInfo> outputs = virtual_dataset->outputs_tensor_info();

  TensorMap input_expect = {1, -1};
  TensorMap output_expect = {1, -1};

  TensorInfo input_tensor_info = inputs.at(0);
  TensorInfo output_tensor_info = outputs.at(0);

  Map input_tensor_map = input_tensor_info.tensor_layout().origin_tensor_map();
  Map output_tensor_map = output_tensor_info.tensor_layout().origin_tensor_map();

  ASSERT_EQ(input_tensor_map.array(), input_expect);
  ASSERT_EQ(output_tensor_map.array(), output_expect);
}

TEST_F(TestVirtualDatasetInfo, GetForwardOp1) {
  std::vector<Dimensions> inputs = {{8, 1}, {8, 1}, {8, 1}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  virtual_dataset->Init(strategy);
  OperatorVector forward_op = virtual_dataset->forward_op();
  size_t size = forward_op.size();

  ASSERT_EQ(size, 0);
}

TEST_F(TestVirtualDatasetInfo, GetMirrorOPs1) {
  std::vector<Dimensions> inputs = {{8, 1}, {8, 1}, {8, 1}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  virtual_dataset->Init(strategy);
  MirrorOps mirror_ops = virtual_dataset->mirror_ops();

  size_t size = mirror_ops.size();
  // no broadcast
  ASSERT_EQ(size, 0);
  // ASSERT_EQ(size, 3);
}

}  // namespace parallel
}  // namespace mindspore
