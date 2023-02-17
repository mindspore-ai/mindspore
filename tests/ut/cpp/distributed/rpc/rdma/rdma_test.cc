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

  std::unique_ptr<MessageBase> CreateMessage(const std::string &msg);
};

std::unique_ptr<MessageBase> RDMATest::CreateMessage(const std::string &msg) {
  std::unique_ptr<MessageBase> message = std::make_unique<MessageBase>();
  size_t msg_size = msg.size();
  if (msg_size == 0) {
    MS_LOG(EXCEPTION) << "msg_size is 0!";
  }
  void *data = malloc(msg_size + 1);
  (void)memcpy_s(data, msg_size, msg.c_str(), msg_size);
  message->data = data;
  message->size = msg_size;
  return message;
}

/// Feature: RDMA communication.
/// Description: test basic connection function between RDMA client and server.
/// Expectation: RDMA client successfully connects to RDMA server and sends a simple message.
TEST_F(RDMATest, TestRDMAConnection) {
  std::string url = "127.0.0.1:10969";
  size_t server_pid = fork();
  if (server_pid == 0) {
    std::shared_ptr<RDMAServer> rdma_server = std::make_shared<RDMAServer>();
    MS_EXCEPTION_IF_NULL(rdma_server);
    ASSERT_TRUE(rdma_server->Initialize(url));
    sleep(3);
    rdma_server->Finalize();
    return;
  }
  sleep(1);
  size_t client_pid = fork();
  if (client_pid == 0) {
    std::shared_ptr<RDMAClient> rdma_client = std::make_shared<RDMAClient>();
    MS_EXCEPTION_IF_NULL(rdma_client);
    ASSERT_TRUE(rdma_client->Initialize());
    ASSERT_TRUE(rdma_client->Connect(url));
    rdma_client->Finalize();
    return;
  }

  int wstatus;
  (void)waitpid(client_pid, &wstatus, WUNTRACED | WCONTINUED);
  (void)waitpid(server_pid, &wstatus, WUNTRACED | WCONTINUED);
}

/// Feature: RDMA communication.
/// Description: test SendSync interface for RDMA client and server.
/// Expectation: RDMA client successfully sends two messages to RDMA server synchronously.
TEST_F(RDMATest, TestRDMASendSync) {
  std::string url = "127.0.0.1:10969";
  size_t server_pid = fork();
  if (server_pid == 0) {
    std::shared_ptr<RDMAServer> rdma_server = std::make_shared<RDMAServer>();
    MS_EXCEPTION_IF_NULL(rdma_server);
    ASSERT_TRUE(rdma_server->Initialize(url));

    auto msg_handler = [](MessageBase *const msg) {
      MS_LOG(INFO) << "Receive message from client: " << static_cast<char *>(msg->data);
      return nullptr;
    };
    rdma_server->SetMessageHandler(msg_handler);
    sleep(3);
    rdma_server->Finalize();
    return;
  }
  sleep(1);
  size_t client_pid = fork();
  if (client_pid == 0) {
    std::shared_ptr<RDMAClient> rdma_client = std::make_shared<RDMAClient>();
    MS_EXCEPTION_IF_NULL(rdma_client);
    ASSERT_TRUE(rdma_client->Initialize());
    ASSERT_TRUE(rdma_client->Connect(url));

    auto message1 = CreateMessage("Hello server sync!");
    ASSERT_TRUE(rdma_client->SendSync(std::move(message1)));
    auto message2 = CreateMessage("Hello server sync!");
    ASSERT_TRUE(rdma_client->SendSync(std::move(message2)));
    rdma_client->Finalize();
    return;
  }

  int wstatus;
  (void)waitpid(client_pid, &wstatus, WUNTRACED | WCONTINUED);
  (void)waitpid(server_pid, &wstatus, WUNTRACED | WCONTINUED);
}

/// Feature: RDMA communication.
/// Description: test SendAsync interface for RDMA client and server.
/// Expectation: RDMA client successfully sends two messages to RDMA server asynchronously.
TEST_F(RDMATest, TestRDMASendAsync) {
  std::string url = "127.0.0.1:10969";
  size_t server_pid = fork();
  if (server_pid == 0) {
    std::shared_ptr<RDMAServer> rdma_server = std::make_shared<RDMAServer>();
    MS_EXCEPTION_IF_NULL(rdma_server);
    ASSERT_TRUE(rdma_server->Initialize(url));

    auto msg_handler = [](MessageBase *const msg) {
      MS_LOG(INFO) << "Receive message from client: " << static_cast<char *>(msg->data);
      return nullptr;
    };
    rdma_server->SetMessageHandler(msg_handler);
    sleep(3);
    rdma_server->Finalize();
    return;
  }
  sleep(1);
  size_t client_pid = fork();
  if (client_pid == 0) {
    std::shared_ptr<RDMAClient> rdma_client = std::make_shared<RDMAClient>();
    MS_EXCEPTION_IF_NULL(rdma_client);
    ASSERT_TRUE(rdma_client->Initialize());
    ASSERT_TRUE(rdma_client->Connect(url));

    auto message1 = CreateMessage("Hello server async!");
    rdma_client->SendAsync(std::move(message1));
    ASSERT_TRUE(rdma_client->Flush(url));
    auto message2 = CreateMessage("Hello server async!");
    rdma_client->SendAsync(std::move(message2));
    ASSERT_TRUE(rdma_client->Flush(url));

    rdma_client->Finalize();
    return;
  }

  int wstatus;
  (void)waitpid(client_pid, &wstatus, WUNTRACED | WCONTINUED);
  (void)waitpid(server_pid, &wstatus, WUNTRACED | WCONTINUED);
}
}  // namespace rpc
}  // namespace distributed
}  // namespace mindspore
