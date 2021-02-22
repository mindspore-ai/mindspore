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
#include <string>

#include "common/common_test.h"
#include "ps/core/cluster_metadata.h"

namespace mindspore {
namespace ps {
namespace core {
class TestClusterMetadata : public UT::Common {
 public:
  TestClusterMetadata() = default;
  virtual ~TestClusterMetadata() = default;

  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(TestClusterMetadata, HeartbeatInterval) {
  ClusterMetadata::instance()->Init(2, 2, "127.0.0.1", 8080);
  EXPECT_TRUE(ClusterMetadata::instance()->heartbeat_interval() == 3);
  ClusterMetadata::instance()->set_heartbeat_interval(100);
  EXPECT_TRUE(ClusterMetadata::instance()->heartbeat_interval() == 100);
  EXPECT_STREQ(ClusterMetadata::instance()->scheduler_host().c_str(), "127.0.0.1");
  EXPECT_TRUE(ClusterMetadata::instance()->scheduler_port() == 8080);
}
}  // namespace core
}  // namespace ps
}  // namespace mindspore