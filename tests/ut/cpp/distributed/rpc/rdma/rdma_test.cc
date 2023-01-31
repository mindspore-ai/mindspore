/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include <sys/resource.h>
#include <sys/types.h>
#include <dirent.h>
#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <csignal>

#include <gtest/gtest.h>
#define private public
#include "distributed/rpc/rdma/rdma_server.h"
#include "distributed/rpc/rdma/rdma_client.h"
#include "distributed/rpc/rdma/constants.h"
#include "common/common_test.h"
#undef private

namespace mindspore {
namespace distributed {
namespace rpc {
class RDMATest : public UT::Common {
 public:
  RDMATest() = default;
  ~RDMATest() = default;
};

/// Feature: RDMA communication.
/// Description: test basic connection function between RDMA client and server.
/// Expectation: RDMA client successfully connects to RDMA server and sends a simple message.
TEST_F(RDMATest, TestRDMAConnection) {
  size_t server_pid = fork();
  if (server_pid == 0) {
    std::shared_ptr<RDMAServer> rdma_server = std::make_shared<RDMAServer>();
    (void)rdma_server->Initialize(kLocalHost);
    return;
  }
  sleep(2);
  size_t client_pid = fork();
  if (client_pid == 0) {
    std::shared_ptr<RDMAClient> rdma_client = std::make_shared<RDMAClient>();
    (void)rdma_client->Initialize();
    return;
  }

  int wstatus;
  (void)waitpid(client_pid, &wstatus, WUNTRACED | WCONTINUED);
  (void)waitpid(server_pid, &wstatus, WUNTRACED | WCONTINUED);
}
}  // namespace rpc
}  // namespace distributed
}  // namespace mindspore
