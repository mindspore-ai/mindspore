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

#include <memory>
#include <vector>
#include <string>
#include "operator/ops.h"
#include "pre_activate/mem_reuse/stream_reuse.h"
#include "common/common_test.h"
#include "common/py_func_graph_fetcher.h"

using mindspore::memreuse::StreamReuse;

namespace mindspore {
class TestStreamMemReuse : public UT::Common {
 public:
  TestStreamMemReuse() : getPyFun_("gtest_input.mem_reuse.TestMemReuseAllocator", true) {}
  void SetUp() {}

 public:
  UT::PyFuncGraphFetcher getPyFun_;
};

TEST_F(TestStreamMemReuse, init_reusable_stream_map_test) {
  std::unordered_map<uint32_t, uint32_t> logic_physic_map{{1, 0}, {2, 8}, {3, 3}};
  std::unordered_map<uint32_t, uint32_t> logic_independent_map{{3, 10}, {2, 11}};
  auto stream_reuse = std::make_shared<StreamReuse>();
  stream_reuse->set_logic_physic_map(logic_physic_map);
  stream_reuse->set_logic_independent_map(logic_independent_map);

  auto logic_phyics_map = stream_reuse->GetLogicPhysicsStreamMap();
  for (const auto &logic_physics : logic_phyics_map) {
    MS_LOG(INFO) << "[logic_id: " << logic_physics.first << "]";
    for (const auto &physic : logic_physics.second) {
      MS_LOG(INFO) << "physic: " << physic;
    }
  }
  MS_LOG(INFO) << "===========UT logic_physic_map size: " << logic_physic_map.size() << "========";
  ASSERT_EQ(logic_physic_map.size(), 3);
  stream_reuse->InitReusableStreamMap();
  auto parallel_streams_map = stream_reuse->parallel_streams_map();
  for (const auto &parallel_streams : parallel_streams_map) {
    MS_LOG(INFO) << "[stream id: " << parallel_streams.first << "]";
    for (const auto &stream : parallel_streams.second) {
      MS_LOG(INFO) << "parallel stream id: " << stream;
    }
  }
  ASSERT_EQ(parallel_streams_map[7].size(), 1);
}
}  // namespace mindspore
