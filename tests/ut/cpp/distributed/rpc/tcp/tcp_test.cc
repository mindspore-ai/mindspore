/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include <gtest/gtest.h>
#define private public
#include "distributed/rpc/tcp/tcp_server.h"
#include "distributed/rpc/tcp/tcp_client.h"
#include "common/common_test.h"

namespace mindspore {
namespace distributed {
namespace rpc {
int g_recv_num = 0;
int g_exit_msg_num = 0;

static size_t g_data_msg_num = 0;

static void Init() { g_data_msg_num = 0; }

static bool WaitForDataMsg(size_t expected_msg_num, int timeout_in_sec) {
  bool rt = false;
  int timeout = timeout_in_sec * 1000 * 1000;
  int usleepCount = 100000;

  while (timeout) {
    if (g_data_msg_num == expected_msg_num) {
      rt = true;
      break;
    }
    timeout = timeout - usleepCount;
    usleep(usleepCount);
  }
  return rt;
}

static void IncrDataMsgNum(size_t number) { g_data_msg_num += number; }

static size_t GetDataMsgNum() { return g_data_msg_num; }

std::atomic<int> m_sendNum(0);
std::string m_localIP = "127.0.0.1";
bool m_notRemote = false;

class TCPTest : public UT::Common {
 protected:
  char *testServerPath;
  static const size_t pid_num = 100;
  pid_t pid1;
  pid_t pid2;

  pid_t pids[pid_num];

  void SetUp() {}
  void TearDown() {}

  std::unique_ptr<MessageBase> CreateMessage(const std::string &serverUrl, const std::string &client_url);

  bool CheckRecvNum(int expectedRecvNum, int _timeout);
  bool CheckExitNum(int expectedExitNum, int _timeout);
};

std::unique_ptr<MessageBase> TCPTest::CreateMessage(const std::string &serverUrl, const std::string &clientUrl) {
  std::unique_ptr<MessageBase> message = std::make_unique<MessageBase>();
  size_t len = 100;
  std::string data(len, 'A');
  message->name = "testname";
  message->from = AID("client", clientUrl);
  message->to = AID("server", serverUrl);
  message->body = data;
  return message;
}

bool TCPTest::CheckRecvNum(int expectedRecvNum, int _timeout) {
  int timeout = _timeout * 1000 * 1000;  // us
  int usleepCount = 100000;

  while (timeout) {
    usleep(usleepCount);
    if (g_recv_num >= expectedRecvNum) {
      return true;
    }
    timeout = timeout - usleepCount;
  }
  return false;
}

bool TCPTest::CheckExitNum(int expectedExitNum, int _timeout) {
  int timeout = _timeout * 1000 * 1000;
  int usleepCount = 100000;

  while (timeout) {
    usleep(usleepCount);
    if (g_exit_msg_num >= expectedExitNum) {
      return true;
    }
    timeout = timeout - usleepCount;
  }

  return false;
}

/// Feature: test failed to start a socket server.
/// Description: start a socket server with an invalid url.
/// Expectation: failed to start the server with invalid url.
TEST_F(TCPTest, StartServerFail) {
  std::unique_ptr<TCPServer> server = std::make_unique<TCPServer>();
  bool ret = server->Initialize("0:2225");
  ASSERT_FALSE(ret);
  server->Finalize();
}

/// Feature: test start a socket server.
/// Description: start the socket server with a specified socket.
/// Expectation: the socket server is started successfully.
TEST_F(TCPTest, StartServerSucc) {
  std::unique_ptr<TCPServer> server = std::make_unique<TCPServer>();
  bool ret = server->Initialize("127.0.0.1:8081");
  ASSERT_TRUE(ret);
  server->Finalize();
}

/// Feature: test normal tcp message sending.
/// Description: start a socket server and send a normal message to it.
/// Expectation: the server received the message sented from client.
TEST_F(TCPTest, SendOneMessage) {
  Init();

  // Start the tcp server.
  auto server_url = "127.0.0.1:8081";
  std::unique_ptr<TCPServer> server = std::make_unique<TCPServer>();
  bool ret = server->Initialize(server_url);
  ASSERT_TRUE(ret);

  server->SetMessageHandler([](const std::shared_ptr<MessageBase> &message) -> void { IncrDataMsgNum(1); });

  // Start the tcp client.
  auto client_url = "127.0.0.1:1234";
  std::unique_ptr<TCPClient> client = std::make_unique<TCPClient>();
  ret = client->Initialize();
  ASSERT_TRUE(ret);

  // Create the message.
  auto message = CreateMessage(server_url, client_url);

  // Send the message.
  client->Connect(server_url);
  client->Send(std::move(message));

  // Wait timeout: 5s
  WaitForDataMsg(1, 5);

  // Check result
  EXPECT_EQ(1, GetDataMsgNum());

  // Destroy
  client->Disconnect(server_url);
  client->Finalize();
  server->Finalize();
}

/// Feature: test sending two message continuously.
/// Description: start a socket server and send two normal message to it.
/// Expectation: the server received the two messages sented from client.
TEST_F(TCPTest, sendTwoMessages) {
  Init();

  // Start the tcp server.
  auto server_url = "127.0.0.1:8081";
  std::unique_ptr<TCPServer> server = std::make_unique<TCPServer>();
  bool ret = server->Initialize(server_url);
  ASSERT_TRUE(ret);

  server->SetMessageHandler([](const std::shared_ptr<MessageBase> &message) -> void { IncrDataMsgNum(1); });

  // Start the tcp client.
  auto client_url = "127.0.0.1:1234";
  std::unique_ptr<TCPClient> client = std::make_unique<TCPClient>();
  ret = client->Initialize();
  ASSERT_TRUE(ret);

  // Create messages.
  auto message1 = CreateMessage(server_url, client_url);
  auto message2 = CreateMessage(server_url, client_url);

  // Send messages.
  client->Connect(server_url);
  client->Send(std::move(message1));
  client->Send(std::move(message2));

  // Wait timeout: 5s
  WaitForDataMsg(2, 5);

  // Check result
  EXPECT_EQ(2, GetDataMsgNum());
  client->Disconnect(server_url);
  client->Finalize();
  server->Finalize();
}
}  // namespace rpc
}  // namespace distributed
}  // namespace mindspore
