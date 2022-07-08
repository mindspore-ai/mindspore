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

namespace mindspore {
namespace parallel {

class TestStrategy : public UT::Common {
 public:
  TestStrategy() {}

  void SetUp() {}
  void TearDown() {}
};

TEST_F(TestStrategy, GetInputNumber) {
  int32_t number = 2;
  int32_t stage = 1;
  Dimensions dimension1 = {2, 4};
  Dimensions dimension2 = {2, 2};
  Strategies inputs = {dimension1, dimension2};

  Strategy strategy(stage, inputs);
  int32_t number_test = strategy.GetInputNumber();
  ASSERT_EQ(number, number_test);
}

TEST_F(TestStrategy, GetInputStage) {
  int32_t stage = 1;
  Dimensions dimension1 = {2, 4};
  Dimensions dimension2 = {2, 2};
  Strategies inputs = {dimension1, dimension2};

  Strategy strategy(stage, inputs);
  int32_t stage_test = strategy.GetInputStage();
  ASSERT_EQ(stage, stage_test);
}

TEST_F(TestStrategy, GetInputDim) {
  int32_t stage = 1;
  Dimensions dimension1 = {2, 4};
  Dimensions dimension2 = {2, 2};
  Strategies inputs = {dimension1, dimension2};

  Strategy strategy(stage, inputs);
  Strategies inputs_test = strategy.GetInputDim();
  ASSERT_EQ(inputs, inputs_test);
}

TEST_F(TestStrategy, IsEqual) {
  int32_t stage1 = 0, stage2 = 0, stage3 = 1, stage4 = 0;
  Dimensions dimension1 = {8, 1};
  Dimensions dimension2 = {1, 8};
  Strategies inputs1 = {dimension1};
  Strategies inputs2 = {dimension1};
  Strategies inputs3 = {dimension2};
  Strategies inputs4 = {dimension1, dimension2};

  StrategyPtr stra1 = std::make_shared<Strategy>(stage1, inputs1);
  StrategyPtr stra2 = std::make_shared<Strategy>(stage2, inputs2);
  StrategyPtr stra3 = std::make_shared<Strategy>(stage3, inputs3);
  StrategyPtr stra4 = std::make_shared<Strategy>(stage4, inputs4);

  ASSERT_EQ(stra1->IsEqual(stra2), true);
  ASSERT_EQ(stra1->IsEqual(stra3), false);
  ASSERT_EQ(stra1->IsEqual(stra4), false);
}
}  // namespace parallel
}  // namespace mindspore
