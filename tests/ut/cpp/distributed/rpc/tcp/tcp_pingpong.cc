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

#include <sys/resource.h>
#include <sys/types.h>
#include <dirent.h>
#include <atomic>
#include <string>
#include <thread>
#include <csignal>
#include <chrono>

#include <gtest/gtest.h>
#define private public
#include "include/backend/distributed/rpc/tcp/tcp_server.h"
#include "include/backend/distributed/rpc/tcp/tcp_client.h"
#include "include/backend/distributed/rpc/tcp/constants.h"
#include "include/backend/distributed/constants.h"
#include "common/common_test.h"

namespace mindspore {
namespace distributed {
namespace rpc {
int recv_num = 0;
int exit_msg_num = 0;
int pingpong_count = 20;
std::string server_ip = distributed::kLocalHost;
std::string server_port = "12345";
std::string server_url = server_ip + ":" + server_port;

static size_t data_msg_num = 0;
std::vector<size_t> send_counts = {
  16, 16 << 8, 16 << 12, 16 << 14, 16 << 16, 16 << 18, 16 << 20, 16 << 22, 16 << 24, 16 << 26,
};

#define CURRENT_TIMESTAMP_MICRO \
  (std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()))

static void Init() {
  std::string pingpong_count_env = "20";
  pingpong_count = atoi(pingpong_count_env.c_str());
  ASSERT_TRUE(pingpong_count > 0);

  std::string user_set_server = "127.0.0.1";
  server_ip = user_set_server.empty() ? distributed::kLocalHost : user_set_server;
  server_url = server_ip + ":" + server_port;
  MS_LOG(INFO) << "Server url is " << server_url;
}

static bool WaitForDataMsg(size_t expected_msg_num, int timeout_in_sec) {
  bool rt = false;
  int timeout = timeout_in_sec * 1000 * 1000;
  int usleepCount = 1000;

  while (timeout) {
    if (data_msg_num == expected_msg_num) {
      rt = true;
      break;
    }
    timeout = timeout - usleepCount;
    usleep(usleepCount);
  }
  return rt;
}

static void IncrDataMsgNum(size_t number) { data_msg_num += number; }

class TCPPingPongTest : public UT::Common {
 protected:
  static const size_t pid_num = 100;
  pid_t pid1;
  pid_t pid2;

  pid_t pids[pid_num];

  void SetUp() {}
  void TearDown() {}

  std::unique_ptr<MessageBase> CreateMessage(const std::string &server_url, size_t msg_size = 100);

  bool CheckRecvNum(int expectedRecvNum, int _timeout);
  bool CheckExitNum(int expectedExitNum, int _timeout);
};

std::unique_ptr<MessageBase> TCPPingPongTest::CreateMessage(const std::string &server_url, size_t msg_size) {
  std::unique_ptr<MessageBase> message = std::make_unique<MessageBase>();
  message->name = "testname";
  message->from = AID("client", "");
  message->to = AID("server", server_url);

  void *data = malloc(msg_size);
  (void)memset_s(data, msg_size, 'A', msg_size);
  message->data = data;
  message->size = msg_size;
  return message;
}

bool TCPPingPongTest::CheckRecvNum(int expectedRecvNum, int _timeout) {
  int timeout = _timeout * 1000 * 1000;  // us
  int usleepCount = 100000;

  while (timeout) {
    usleep(usleepCount);
    if (recv_num >= expectedRecvNum) {
      return true;
    }
    timeout = timeout - usleepCount;
  }
  return false;
}

bool TCPPingPongTest::CheckExitNum(int expectedExitNum, int _timeout) {
  int timeout = _timeout * 1000 * 1000;
  int usleepCount = 100000;

  while (timeout) {
    usleep(usleepCount);
    if (exit_msg_num >= expectedExitNum) {
      return true;
    }
    timeout = timeout - usleepCount;
  }

  return false;
}

/// Feature: Test tcp message send and recv multiple times.
/// Description: Start a socket server and start pingpong with client.
/// Expectation: The server stops or crashes.
TEST_F(TCPPingPongTest, DISABLED_PingPongServer) {
  Init();

  // Start the tcp server.
  std::unique_ptr<TCPServer> server = std::make_unique<TCPServer>();
  bool ret = server->Initialize(server_url);
  ASSERT_TRUE(ret);

  server->SetMessageHandler([this](MessageBase *const message) -> MessageBase *const {
    IncrDataMsgNum(1);
    return message;
  });

  // Wait timeout: 5s
  WaitForDataMsg(pingpong_count * send_counts.size(), 5 * 60);

  // Destroy
  server->Finalize();
}

/// Feature: Test tcp message send and recv multiple times.
/// Description: Start a socket client and start pingpong with server.
/// Expectation: The client stops or crashes.
TEST_F(TCPPingPongTest, DISABLED_PingPongClient) {
  Init();

  // Start the tcp client.
  std::unique_ptr<TCPClient> client = std::make_unique<TCPClient>();
  bool ret = client->Initialize();
  ASSERT_TRUE(ret);

  // Send the message.
  if (client->Connect(server_url)) {
    for (size_t msg_size : send_counts) {
      size_t total_time = 0;
      for (size_t i = 0; i < pingpong_count; i++) {
        size_t start_ts = CURRENT_TIMESTAMP_MICRO.count();
        auto message = CreateMessage(server_url, msg_size);
        client->ReceiveSync(std::move(message));
        size_t end_ts = CURRENT_TIMESTAMP_MICRO.count();
        total_time += end_ts - start_ts;
      }
      float avg_time = total_time / pingpong_count;
      MS_LOG(WARNING) << "Avg pingpong time for message " << msg_size << " is " << avg_time << " ms.";
    }
  }

  // Destroy
  client->Disconnect(server_url);
  client->Finalize();
}
}  // namespace rpc
}  // namespace distributed
}  // namespace mindspore
