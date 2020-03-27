/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "common/common_test.h"
#include "utils/config_manager.h"

namespace mindspore {
class TestConfigManager : public UT::Common {
 public:
  TestConfigManager() {}
};

TEST_F(TestConfigManager, TestAPI) {
  ASSERT_TRUE(ConfigManager::GetInstance().parallel_strategy() == ParallelStrategy::ONE_DEVICE);

  ConfigManager::GetInstance().set_parallel_strategy(ParallelStrategy::DISTRIBUTION);
  ASSERT_TRUE(ConfigManager::GetInstance().parallel_strategy() == ParallelStrategy::DISTRIBUTION);
}

}  // namespace mindspore
