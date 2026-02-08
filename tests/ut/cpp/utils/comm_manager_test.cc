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
#include "include/utils/comm_manager.h"
#include "common/common_test.h"

namespace mindspore {
class TestCommManager : public UT::Common {
 public:
  TestCommManager() {}
};

TEST_F(TestCommManager, TestCreate) {
  std::vector<unsigned int> devices{0, 1, 2};
  ASSERT_TRUE(CommManager::GetInstance().CreateGroupSync(std::string("1-2-3"), devices));
  ASSERT_TRUE(CommManager::GetInstance().CreateGroupSync(std::string("hccl_world_group"), devices));
}

TEST_F(TestCommManager, TestGetSize) {
  unsigned int rank_size = 0;
  ASSERT_TRUE(CommManager::GetInstance().GetRankSize(std::string("1-2-3"), &rank_size));
  ASSERT_TRUE(CommManager::GetInstance().GetRankSize(std::string("hccl_world_group"), &rank_size));
}

}  // namespace mindspore
