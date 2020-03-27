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
  std::vector<int32_t> dimension1 = {2, 4};
  std::vector<int32_t> dimension2 = {2, 2};
  std::vector<std::vector<int32_t>> inputs = {dimension1, dimension2};

  Strategy strategy(stage, inputs);
  int32_t number_test = strategy.GetInputNumber();
  ASSERT_EQ(number, number_test);
}

TEST_F(TestStrategy, GetInputStage) {
  int32_t stage = 1;
  std::vector<int32_t> dimension1 = {2, 4};
  std::vector<int32_t> dimension2 = {2, 2};
  std::vector<std::vector<int32_t>> inputs = {dimension1, dimension2};

  Strategy strategy(stage, inputs);
  int32_t stage_test = strategy.GetInputStage();
  ASSERT_EQ(stage, stage_test);
}

TEST_F(TestStrategy, GetInputDim) {
  int32_t stage = 1;
  std::vector<int32_t> dimension1 = {2, 4};
  std::vector<int32_t> dimension2 = {2, 2};
  std::vector<std::vector<int32_t>> inputs = {dimension1, dimension2};

  Strategy strategy(stage, inputs);
  std::vector<std::vector<int32_t>> inputs_test = strategy.GetInputDim();
  ASSERT_EQ(inputs, inputs_test);
}

}  // namespace parallel
}  // namespace mindspore
