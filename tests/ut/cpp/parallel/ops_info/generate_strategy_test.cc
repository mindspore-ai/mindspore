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
#include "frontend/parallel/strategy.h"
#include "frontend/parallel/ops_info/arithmetic_info.h"
#include "frontend/parallel/device_manager.h"
#include "frontend/parallel/step_parallel.h"

namespace mindspore {
namespace parallel {

class TestGenerateStrategy : public UT::Common {
 public:
  TestGenerateStrategy() {}
  void SetUp();
  void TearDown() {}
};

void TestGenerateStrategy::SetUp() {
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
}

TEST_F(TestGenerateStrategy, AutoStrategy1) {
  std::vector<StrategyPtr> sp_vector;
  Shapes inputs_shape = {{64, 32}, {64, 32}};
  Shapes splittable_inputs = {{1, 1}, {1, 1}};
  GenerateStrategiesWithBroadcast(0, inputs_shape, splittable_inputs, &sp_vector);
  ASSERT_EQ(sp_vector.size(), 5);
  for (auto& sp : sp_vector) {
    Dimensions input0_strategy = sp->GetInputDim()[0];
    Dimensions input1_strategy = sp->GetInputDim()[1];
    ASSERT_EQ(input0_strategy, input1_strategy);
  }
}

TEST_F(TestGenerateStrategy, AutoStrategy2) {
  std::vector<StrategyPtr> sp_vector;
  Shapes inputs_shape = {{32}, {64, 32}};
  Shapes splittable_inputs = {{1}, {1, 1}};
  GenerateStrategiesWithBroadcast(0, inputs_shape, splittable_inputs, &sp_vector);
  ASSERT_EQ(sp_vector.size(), 5);
  for (auto& sp : sp_vector) {
    Dimensions input0_strategy = sp->GetInputDim()[0];
    Dimensions input1_strategy = sp->GetInputDim()[1];
    ASSERT_EQ(input0_strategy[0], input1_strategy[1]);
    ASSERT_EQ(input0_strategy.size(), 1);
    ASSERT_EQ(input1_strategy.size(), 2);
  }
}

TEST_F(TestGenerateStrategy, AutoStrategy3) {
  std::vector<StrategyPtr> sp_vector;
  Shapes inputs_shape = {{64, 32}, {32}};
  Shapes splittable_inputs = {{1, 1}, {1}};
  GenerateStrategiesWithBroadcast(0, inputs_shape, splittable_inputs, &sp_vector);
  ASSERT_EQ(sp_vector.size(), 5);
  for (auto& sp : sp_vector) {
    Dimensions input0_strategy = sp->GetInputDim()[0];
    Dimensions input1_strategy = sp->GetInputDim()[1];
    ASSERT_EQ(input0_strategy[1], input1_strategy[0]);
    ASSERT_EQ(input0_strategy.size(), 2);
    ASSERT_EQ(input1_strategy.size(), 1);
  }
}

TEST_F(TestGenerateStrategy, AutoStrategy4) {
  std::vector<StrategyPtr> sp_vector;
  Shapes inputs_shape = {{64, 1}, {1, 32}};
  Shapes splittable_inputs = {{1, 1}, {1, 1}};
  GenerateStrategiesWithBroadcast(0, inputs_shape, splittable_inputs, &sp_vector);
  ASSERT_EQ(sp_vector.size(), 5);
  for (auto& sp : sp_vector) {
    Dimensions input0_strategy = sp->GetInputDim()[0];
    Dimensions input1_strategy = sp->GetInputDim()[1];
    ASSERT_EQ(input0_strategy[1], 1);
    ASSERT_EQ(input1_strategy[0], 1);
  }
}

TEST_F(TestGenerateStrategy, AutoStrategy5) {
  std::vector<StrategyPtr> sp_vector;
  Shapes inputs_shape = {{64, 8, 1}, {1, 8, 32}};
  Shapes splittable_inputs = {{1, 1, 1}, {1, 1, 1}};
  GenerateStrategiesWithBroadcast(0, inputs_shape, splittable_inputs, &sp_vector);
  ASSERT_EQ(sp_vector.size(), 11);
  for (auto& sp : sp_vector) {
    Dimensions input0_strategy = sp->GetInputDim()[0];
    Dimensions input1_strategy = sp->GetInputDim()[1];
    ASSERT_EQ(input0_strategy[2], 1);
    ASSERT_EQ(input1_strategy[0], 1);
    ASSERT_EQ(input0_strategy[1], input1_strategy[1]);
  }
}

TEST_F(TestGenerateStrategy, AutoStrategy6) {
  std::vector<StrategyPtr> sp_vector;
  Shapes inputs_shape = {{64, 32}};
  Shapes splittable_inputs = {{1, 1}};
  GenerateStrategiesForIndependentInputs(0, inputs_shape, splittable_inputs, &sp_vector);
  ASSERT_EQ(sp_vector.size(), 5);
}

TEST_F(TestGenerateStrategy, AutoStrategy7) {
  std::vector<StrategyPtr> sp_vector;
  Shapes inputs_shape = {{1, 32}, {64, 8, 32}};
  Shapes splittable_inputs = {{1, 1}, {1, 1, 1}};
  GenerateStrategiesWithBroadcast(0, inputs_shape, splittable_inputs, &sp_vector);
  ASSERT_EQ(sp_vector.size() > 0, true);
  for (auto& sp : sp_vector) {
    Dimensions input0_strategy = sp->GetInputDim()[0];
    Dimensions input1_strategy = sp->GetInputDim()[1];
    ASSERT_EQ(input0_strategy[0], 1);
    ASSERT_EQ(input0_strategy.size(), 2);
    ASSERT_EQ(input1_strategy.size(), 3);
    ASSERT_EQ(input0_strategy[1], input1_strategy[2]);
  }
}

TEST_F(TestGenerateStrategy, AutoStrategy8) {
  std::vector<StrategyPtr> sp_vector;
  Shapes inputs_shape = {{64, 8, 32}, {1, 32}};
  Shapes splittable_inputs = {{1, 1, 1}, {1, 1}};
  GenerateStrategiesWithBroadcast(0, inputs_shape, splittable_inputs, &sp_vector);
  ASSERT_EQ(sp_vector.size() > 0, true);
  for (auto& sp : sp_vector) {
    Dimensions input0_strategy = sp->GetInputDim()[0];
    Dimensions input1_strategy = sp->GetInputDim()[1];
    ASSERT_EQ(input1_strategy[0], 1);
    ASSERT_EQ(input0_strategy.size(), 3);
    ASSERT_EQ(input1_strategy.size(), 2);
    ASSERT_EQ(input0_strategy[2], input1_strategy[1]);
  }
}

}  // namespace parallel
}  // namespace mindspore
