/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "frontend/parallel/ops_info/addn_info.h"
#include "frontend/parallel/device_manager.h"
#include "frontend/parallel/step_parallel.h"

namespace mindspore {
namespace parallel {

class AddNInfo;
using AddNInfoPtr = std::shared_ptr<AddNInfo>;
AddNInfoPtr addn;

class TestInferStrategySameMode : public UT::Common {
 public:
  TestInferStrategySameMode() {}
  void SetUp();
  void TearDown() {}
};

void TestInferStrategySameMode::SetUp() {
  RankList dev_list;

  for (int32_t i = 0; i < 34; i++) {
    dev_list.push_back(i);
  }

  RankList stage_map;
  stage_map.push_back(32);
  stage_map.push_back(2);

  int32_t local_dev = 0;

  // create a new g_device_manager
  g_device_manager = std::make_shared<DeviceManager>();
  g_device_manager->Init(dev_list, local_dev, stage_map, "hccl");

  mindspore::HashMap<std::string, ValuePtr> attr;

  Shapes inputs_shape = {{32, 64, 96}, {32, 64, 96}};
  Shapes outputs_shape = {{32, 64, 96}};
  addn = std::make_shared<AddNInfo>("addn_info", inputs_shape, outputs_shape, attr);
}

/// Feature: infer strategy for same mode
/// Description: the in strategy is {{2, 4, 4}, {}}
/// Expectation: the return strategy is {{2, 4, 4}, {2, 4, 4}}
TEST_F(TestInferStrategySameMode, GenerateFullStrategy1) {
  Strategies in_strategy = {{2, 4, 4}, {}};
  Strategies ret = addn->GenerateFullStrategy(in_strategy);

  Strategies expect = {{2, 4, 4}, {2, 4, 4}};
  ASSERT_EQ(ret, expect);
}

/// Feature: infer strategy for same mode
/// Description: the in strategy is {{}, {2, 4, 4}}
/// Expectation: the return strategy is {{2, 4, 4}, {2, 4, 4}}
TEST_F(TestInferStrategySameMode, GenerateFullStrategy2) {
  Strategies in_strategy = {{}, {2, 4, 4}};
  Strategies ret = addn->GenerateFullStrategy(in_strategy);

  Strategies expect = {{2, 4, 4}, {2, 4, 4}};
  ASSERT_EQ(ret, expect);
}
}  // namespace parallel
}  // namespace mindspore
