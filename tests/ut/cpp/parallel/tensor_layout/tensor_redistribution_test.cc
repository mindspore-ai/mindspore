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

#include <vector>
#include "common/common_test.h"
#include "common/py_func_graph_fetcher.h"
#include "frontend/parallel/tensor_layout/tensor_redistribution.h"
#include "frontend/parallel/step_parallel.h"

namespace mindspore {
namespace parallel {

class TestTensorRedistribution : public UT::Common {
 public:
  TestTensorRedistribution() {}

  void SetUp() {
    RankList dev_list;

    for (int32_t i = 0; i < 20; i++) {
      dev_list.push_back(i);
    }

    RankList stage_map;
    stage_map.push_back(16);
    stage_map.push_back(4);

    int32_t local_dev = 0;

    // create a new g_device_manager
    g_device_manager = std::make_shared<DeviceManager>();
    g_device_manager->Init(dev_list, local_dev, stage_map, "hccl");
  }

  virtual void TearDown() {}
};

// Redistribution: Reshape -> SplitByAxis -> ConcatByAxis -> SplitByAxis -> Reshape
TEST_F(TestTensorRedistribution, TestInferRedistribution1) {
  DeviceArrangement device_arrangement = {2, 4, 2};
  TensorMap tensor_map = {2, 0};
  TensorShape tensor_shape = {512, 1024};

  Arrangement in_device_arrangement;
  Status status = in_device_arrangement.Init(device_arrangement);
  ASSERT_EQ(Status::SUCCESS, status);
  Map in_tensor_map;
  status = in_tensor_map.Init(tensor_map);
  ASSERT_EQ(Status::SUCCESS, status);
  Arrangement in_tensor_shape;
  status = in_tensor_shape.Init(tensor_shape);
  ASSERT_EQ(Status::SUCCESS, status);
  TensorLayout from_layout;
  status = from_layout.Init(in_device_arrangement, in_tensor_map, in_tensor_shape);
  ASSERT_EQ(Status::SUCCESS, status);

  device_arrangement = {4, 2, 2};
  tensor_map = {2, 1};

  Arrangement out_device_arrangement;
  status = out_device_arrangement.Init(device_arrangement);
  ASSERT_EQ(Status::SUCCESS, status);
  Map out_tensor_map;
  status = out_tensor_map.Init(tensor_map);
  ASSERT_EQ(Status::SUCCESS, status);
  Arrangement out_tensor_shape;
  status = out_tensor_shape.Init(tensor_shape);
  ASSERT_EQ(Status::SUCCESS, status);
  TensorLayout to_layout;
  status = to_layout.Init(out_device_arrangement, out_tensor_map, out_tensor_shape);
  ASSERT_EQ(Status::SUCCESS, status);

  TensorRedistribution tensor_redistribution;
  RankList dev_list = g_device_manager->GetDeviceListByStageId(0);
  tensor_redistribution.Init(from_layout, to_layout, dev_list);
  std::shared_ptr<std::pair<OperatorVector, OutPutInfoVector>> op_ptr;
  op_ptr = tensor_redistribution.InferTensorRedistributionOperatorList();
  ASSERT_TRUE(op_ptr != nullptr);
  ASSERT_EQ(op_ptr->first.size(), 7);
  ASSERT_EQ(op_ptr->second.size(), 7);

  std::vector<OperatorName> op_names;
  for (auto iter : op_ptr->first) {
    op_names.push_back(iter.first);
  }
  std::vector<OperatorName> expected_op_names = {"Reshape", "StridedSlice", "AllGather", "Split",
                                                 "Concat",  "StridedSlice", "Reshape"};
  ASSERT_EQ(op_names, expected_op_names);
}

// Redistribution: AlltoAll
TEST_F(TestTensorRedistribution, TestInferRedistribution2) {
  DeviceArrangement device_arrangement = {16, 1, 1};
  TensorMap tensor_map = {2, 0};
  TensorShape tensor_shape = {512, 1024};

  Arrangement in_device_arrangement;
  Status status = in_device_arrangement.Init(device_arrangement);
  ASSERT_EQ(Status::SUCCESS, status);
  Map in_tensor_map;
  status = in_tensor_map.Init(tensor_map);
  ASSERT_EQ(Status::SUCCESS, status);
  Arrangement in_tensor_shape;
  status = in_tensor_shape.Init(tensor_shape);
  ASSERT_EQ(Status::SUCCESS, status);
  TensorLayout from_layout;
  status = from_layout.Init(in_device_arrangement, in_tensor_map, in_tensor_shape);
  ASSERT_EQ(Status::SUCCESS, status);

  device_arrangement = {1, 16, 1};
  tensor_map = {2, 1};

  Arrangement out_device_arrangement;
  status = out_device_arrangement.Init(device_arrangement);
  ASSERT_EQ(Status::SUCCESS, status);
  Map out_tensor_map;
  status = out_tensor_map.Init(tensor_map);
  ASSERT_EQ(Status::SUCCESS, status);
  Arrangement out_tensor_shape;
  status = out_tensor_shape.Init(tensor_shape);
  ASSERT_EQ(Status::SUCCESS, status);
  TensorLayout to_layout;
  status = to_layout.Init(out_device_arrangement, out_tensor_map, out_tensor_shape);
  ASSERT_EQ(Status::SUCCESS, status);

  TensorRedistribution tensor_redistribution;
  RankList dev_list = g_device_manager->GetDeviceListByStageId(0);
  tensor_redistribution.Init(from_layout, to_layout, dev_list);
  std::shared_ptr<std::pair<OperatorVector, OutPutInfoVector>> op_ptr;
  op_ptr = tensor_redistribution.InferTensorRedistributionOperatorList();
  ASSERT_TRUE(op_ptr != nullptr);
  ASSERT_EQ(op_ptr->first.size(), 2);
  ASSERT_EQ(op_ptr->second.size(), 2);

  std::vector<OperatorName> op_names;
  for (auto iter : op_ptr->first) {
    op_names.push_back(iter.first);
  }
  std::vector<OperatorName> expected_op_names = {"AllGather", "StridedSlice"};
  ASSERT_EQ(op_names, expected_op_names);
}

// Redistribution: Reshape
TEST_F(TestTensorRedistribution, TestInferRedistribution3) {
  DeviceArrangement device_arrangement = {8};
  TensorMap tensor_map = {0, -1, -1, -1};
  TensorShape tensor_shape = {128, 64, 1, 1};

  Arrangement in_device_arrangement;
  Status status = in_device_arrangement.Init(device_arrangement);
  ASSERT_EQ(Status::SUCCESS, status);
  Map in_tensor_map;
  status = in_tensor_map.Init(tensor_map);
  ASSERT_EQ(Status::SUCCESS, status);
  Arrangement in_tensor_shape;
  status = in_tensor_shape.Init(tensor_shape);
  ASSERT_EQ(Status::SUCCESS, status);
  TensorLayout from_layout;
  status = from_layout.Init(in_device_arrangement, in_tensor_map, in_tensor_shape);
  ASSERT_EQ(Status::SUCCESS, status);

  device_arrangement = {8};
  tensor_map = {0, -1};
  tensor_shape = {128, 64};

  Arrangement out_device_arrangement;
  status = out_device_arrangement.Init(device_arrangement);
  ASSERT_EQ(Status::SUCCESS, status);
  Map out_tensor_map;
  status = out_tensor_map.Init(tensor_map);
  ASSERT_EQ(Status::SUCCESS, status);
  Arrangement out_tensor_shape;
  status = out_tensor_shape.Init(tensor_shape);
  ASSERT_EQ(Status::SUCCESS, status);
  TensorLayout to_layout;
  status = to_layout.Init(out_device_arrangement, out_tensor_map, out_tensor_shape);
  ASSERT_EQ(Status::SUCCESS, status);

  TensorRedistribution tensor_redistribution;
  RankList dev_list = g_device_manager->GetDeviceListByStageId(0);
  tensor_redistribution.Init(from_layout, to_layout, dev_list);
  std::shared_ptr<std::pair<OperatorVector, OutPutInfoVector>> op_ptr;
  op_ptr = tensor_redistribution.InferTensorRedistributionOperatorList();
  ASSERT_TRUE(op_ptr != nullptr);
  ASSERT_EQ(op_ptr->first.size(), 1);
  ASSERT_EQ(op_ptr->second.size(), 1);

  std::vector<OperatorName> op_names;
  for (auto iter : op_ptr->first) {
    op_names.push_back(iter.first);
  }
  std::vector<OperatorName> expected_op_names = {"Reshape"};
  ASSERT_EQ(op_names, expected_op_names);
}

}  // namespace parallel
}  // namespace mindspore
