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

#include <common/common_test.h>
#include "parallel/tensor_layout/tensor_layout.h"
#include "parallel/tensor_layout/tensor_info.h"
#include "parallel/auto_parallel/operator_costmodel.h"
#include "parallel/device_manager.h"

namespace mindspore {
namespace parallel {

class TestMatMulCost : public UT::Common {
 public:
  TestMatMulCost() {}
  void SetUp();
  void TearDown();
  MatMulCost mmcost_;
};

void TestMatMulCost::SetUp() {
  mmcost_ = MatMulCost();
  std::vector<int32_t> dev_list;

  for (int32_t i = 0; i < 1050; i++) {
    dev_list.push_back(i);
  }

  std::vector<int32_t> stage_map;
  stage_map.push_back(1024);
  stage_map.push_back(26);

  int32_t local_dev = 0;

  // create a new g_device_manager
  g_device_manager = std::make_shared<DeviceManager>();
  g_device_manager->Init(dev_list, local_dev, stage_map, "hccl");
}

void TestMatMulCost::TearDown() {
  // destroy resources
}

TEST_F(TestMatMulCost, test_CostGeneration) {
  // Currently, the implementation of GetForwardCommCost() method
  // does not check the tensor layouts information, instead, it checks the
  // tensor shape and the slice shape.
  TensorLayout input0_layout, input1_layout, output0_layout;
  Shape input0_shape{200, 300}, input1_shape{300, 500}, output0_shape{200, 500};
  Shape input0_slice_shape{20, 50}, input1_slice_shape{50, 25}, output0_slice_shape{20, 25};
  TensorInfo input0(input0_layout, input0_shape, input0_slice_shape),
    input1(input1_layout, input1_shape, input1_slice_shape),
    output0(output0_layout, output0_shape, output0_slice_shape);

  std::vector<TensorInfo> inputs, outputs;
  inputs.push_back(input0);
  inputs.push_back(input1);
  outputs.push_back(output0);

  mmcost_.set_is_parameter({false, false});
  std::vector<size_t> inputs_length = {4, 4};
  std::vector<size_t> outputs_length = {4};
  mmcost_.SetInputAndOutputTypeLength(inputs_length, outputs_length);
  mmcost_.GetForwardCommCost(inputs, outputs, 0);
  mmcost_.GetBackwardCommCost(inputs, outputs, 0);
  mmcost_.GetForwardComputationCost(inputs, outputs, 0);
  mmcost_.GetForwardComputationCost(inputs, outputs, 0);
}

class TestActivationCost : public UT::Common {
 public:
  TestActivationCost() {}
  void SetUp();
  void TearDown();
  ActivationCost ac_cost_;
};

void TestActivationCost::SetUp() {
  ac_cost_ = ActivationCost();
  std::vector<int32_t> dev_list;

  for (int32_t i = 0; i < 1050; i++) {
    dev_list.push_back(i);
  }

  std::vector<int32_t> stage_map;
  stage_map.push_back(1024);
  stage_map.push_back(26);

  int32_t local_dev = 0;

  // create a new g_device_manager
  g_device_manager = std::make_shared<DeviceManager>();
  g_device_manager->Init(dev_list, local_dev, stage_map, "hccl");
}

void TestActivationCost::TearDown() {
  // destroy resources
}

TEST_F(TestActivationCost, test_CostGeneration) {
  // Currently, the implementation of GetForwardCommCost() method
  // does not check the tensor layouts information, instead, it checks the
  // tensor shape and the slice shape.
  TensorLayout input0_layout, output0_layout;
  Shape input0_shape{200, 300}, output0_shape{200, 300};
  Shape input0_slice_shape{20, 30}, output0_slice_shape{20, 30};
  TensorInfo input0_info(input0_layout, input0_shape, input0_slice_shape),
    output0_info(output0_layout, output0_shape, output0_slice_shape);
  std::vector<TensorInfo> inputs, outputs;
  inputs.push_back(input0_info);
  outputs.push_back(output0_info);

  ac_cost_.set_is_parameter({false, false});
  std::vector<size_t> inputs_length = {4, 4};
  std::vector<size_t> outputs_length = {4};
  ac_cost_.SetInputAndOutputTypeLength(inputs_length, outputs_length);
  ac_cost_.GetForwardComputationCost(inputs, outputs, 0);
  ac_cost_.GetBackwardComputationCost(inputs, outputs, 0);
}

class TestPReLUCost : public UT::Common {
 public:
  TestPReLUCost() {}
  void SetUp();
  void TearDown();
  PReLUCost prelu_cost_;
};

void TestPReLUCost::SetUp() {
  prelu_cost_ = PReLUCost();
  std::vector<int32_t> dev_list;

  for (int32_t i = 0; i < 1050; i++) {
    dev_list.push_back(i);
  }

  std::vector<int32_t> stage_map;
  stage_map.push_back(1024);
  stage_map.push_back(26);

  int32_t local_dev = 0;

  // create a new g_device_manager
  g_device_manager = std::make_shared<DeviceManager>();
  g_device_manager->Init(dev_list, local_dev, stage_map, "hccl");
}

void TestPReLUCost::TearDown() {
  // destroy resources
}

TEST_F(TestPReLUCost, test_CostGeneration) {
  TensorLayout input_layout, param_layout, output_layout;
  Shape input_shape = {32, 32, 32, 32};
  Shape param_shape = {32};
  Shape output_shape = {32, 32, 32, 32};
  Shape input_slice_shape = {8, 32, 8, 8};
  Shape param_slice_shape = {32};
  Shape output_slice_shape = {8, 32, 8, 8};
  TensorInfo input_info(input_layout, input_shape, input_slice_shape);
  TensorInfo param_info(param_layout, param_shape, param_slice_shape);
  TensorInfo output_info(output_layout, output_shape, output_slice_shape);
  std::vector<TensorInfo> inputs, outputs;
  inputs.push_back(input_info);
  inputs.push_back(param_info);
  outputs.push_back(output_info);
  prelu_cost_.set_is_parameter({false, true});
  std::vector<size_t> inputs_length = {4, 4};
  std::vector<size_t> outputs_length = {4};
  prelu_cost_.SetInputAndOutputTypeLength(inputs_length, outputs_length);
  double BCC, FMC, GMC;
  BCC = prelu_cost_.GetBackwardCommCost(inputs, outputs, 0);
  FMC = prelu_cost_.GetForwardComputationCost(inputs, outputs, 0);
  GMC = prelu_cost_.GetBackwardComputationCost(inputs, outputs, 0);
  ASSERT_EQ(BCC, 32 * 4);
  ASSERT_EQ(FMC, 8 * 32 * 8 * 8 * 4 + 32 * 4);
  ASSERT_EQ(GMC, 128);
}
}  // namespace parallel
}  // namespace mindspore
