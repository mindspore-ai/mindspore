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
#include "frontend/parallel/ops_info/virtual_dataset_info.h"
#include "frontend/parallel/device_manager.h"
#include "frontend/parallel/step_parallel.h"

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
  RankList dev_list;

  for (int32_t i = 0; i < 130; i++) {
    dev_list.push_back(i);
  }

  RankList stage_map;
  stage_map.push_back(16);
  stage_map.push_back(114);

  int32_t local_dev = 0;

  // create a new g_device_manager
  g_device_manager = std::make_shared<DeviceManager>();
  g_device_manager->Init(dev_list, local_dev, stage_map, "hccl");

  mindspore::HashMap<std::string, ValuePtr> attr;

  Shapes inputs_shape = {{128, 32}, {1280, 320}, {12800, 3200}};
  Shapes outputs_shape = {{128, 32}, {1280, 320}, {12800, 3200}};

  virtual_dataset = std::make_shared<VirtualDatasetInfo>("virtual_dataset_info", inputs_shape, outputs_shape, attr);
}

TEST_F(TestVirtualDatasetInfo, InferDevMatrixShape1) {
  Strategies inputs = {{16, 1}, {16, 1}, {16, 1}};
  StrategyPtr strategy = NewStrategy(0, inputs);
  virtual_dataset->Init(strategy, nullptr);
  Shape dev_matrix_shape = virtual_dataset->dev_matrix_shape();

  Shape expect = {16, 1};
  ASSERT_EQ(dev_matrix_shape, expect);
}

TEST_F(TestVirtualDatasetInfo, GetForwardOp1) {
  Strategies inputs = {{8, 1}, {8, 1}, {8, 1}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  virtual_dataset->Init(strategy, nullptr);
  OperatorVector forward_op = virtual_dataset->forward_op();
  size_t size = forward_op.size();

  ASSERT_EQ(size, 0);
}

TEST_F(TestVirtualDatasetInfo, GetMirrorOPs1) {
  Strategies inputs = {{8, 1}, {8, 1}, {8, 1}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  virtual_dataset->Init(strategy, nullptr);
  MirrorOps mirror_ops = virtual_dataset->mirror_ops();

  size_t size = mirror_ops.size();
  // no broadcast
  ASSERT_EQ(size, 0);
  // ASSERT_EQ(size, 3);
}

}  // namespace parallel
}  // namespace mindspore
