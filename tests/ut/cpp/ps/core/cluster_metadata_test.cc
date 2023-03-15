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
#include "ps/core/cluster_config.h"
#include "include/backend/distributed/ps/ps_context.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace ps {
namespace core {
class TestClusterConfig : public UT::Common {
 public:
  TestClusterConfig() = default;
  virtual ~TestClusterConfig() = default;
  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(TestClusterConfig, HeartbeatInterval) {
  std::string worker_num = "1";
  std::string server_num = "1";
  std::string host = "127.0.0.1";
  std::string port = "9999";
  common::SetEnv(kEnvWorkerNum, worker_num.c_str());
  common::SetEnv(kEnvPServerNum, server_num.c_str());
  common::SetEnv(kEnvSchedulerHost, host.c_str());
  common::SetEnv(kEnvSchedulerPort, port.c_str());
  PSContext::instance()->SetPSEnable(true);
  EXPECT_EQ(900, PSContext::instance()->cluster_config().cluster_available_timeout);
}
}  // namespace core
}  // namespace ps
}  // namespace mindspore