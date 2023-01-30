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
#include "distributed/rpc/tcp/constants.h"
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

  std::unique_ptr<MessageBase> CreateMessage(const std::string &serverUrl, const std::string &client_url,
                                             size_t msg_size = 100);

  bool CheckRecvNum(int expectedRecvNum, int _timeout);
  bool CheckExitNum(int expectedExitNum, int _timeout);
};

std::unique_ptr<MessageBase> TCPTest::CreateMessage(const std::string &serverUrl, const std::string &clientUrl,
                                                    size_t msg_size) {
  std::unique_ptr<MessageBase> message = std::make_unique<MessageBase>();
  message->name = "testname";
  message->from = AID("client", clientUrl);
  message->to = AID("server", serverUrl);

  if (msg_size == 0) {
    MS_LOG(EXCEPTION) << "msg_size should be greater than 0.";
  }
  void *data = malloc(msg_size);
  (void)memset_s(data, msg_size, 'A', msg_size);
  message->data = data;
  message->size = msg_size;
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

  server->SetMessageHandler([](MessageBase *const message) -> MessageBase *const {
    IncrDataMsgNum(1);
    return NULL_MSG;
  });

  // Start the tcp client.
  auto client_url = "127.0.0.1:1234";
  std::unique_ptr<TCPClient> client = std::make_unique<TCPClient>();
  ret = client->Initialize();
  ASSERT_TRUE(ret);

  // Create the message.
  auto message = CreateMessage(server_url, client_url);

  // Send the message.
  client->Connect(server_url);
  client->SendAsync(std::move(message));

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
TEST_F(TCPTest, SendTwoMessages) {
  Init();

  // Start the tcp server.
  auto server_url = "127.0.0.1:8081";
  std::unique_ptr<TCPServer> server = std::make_unique<TCPServer>();
  bool ret = server->Initialize(server_url);
  ASSERT_TRUE(ret);

  server->SetMessageHandler([](MessageBase *const message) -> MessageBase *const {
    IncrDataMsgNum(1);
    return NULL_MSG;
  });

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
  client->SendAsync(std::move(message1));
  client->SendAsync(std::move(message2));

  // Wait timeout: 5s
  WaitForDataMsg(2, 5);

  // Check result
  EXPECT_EQ(2, GetDataMsgNum());
  client->Disconnect(server_url);
  client->Finalize();
  server->Finalize();
}

/// Feature: test start the tcp server with random port.
/// Description: start a socket server without specified fixed port.
/// Expectation: the server started successfully.
TEST_F(TCPTest, StartServerWithRandomPort) {
  std::unique_ptr<TCPServer> server = std::make_unique<TCPServer>();
  bool ret = server->Initialize();
  ASSERT_TRUE(ret);

  auto port = server->GetPort();
  EXPECT_LT(0, port);
  server->Finalize();
}

/// Feature: test send the message synchronously.
/// Description: start a socket server and send the message synchronously.
/// Expectation: the number of bytes sent could be got synchronously.
TEST_F(TCPTest, SendSyncMessage) {
  Init();

  // Start the tcp server.
  auto server_url = "127.0.0.1:8081";
  std::unique_ptr<TCPServer> server = std::make_unique<TCPServer>();
  bool ret = server->Initialize(server_url);
  ASSERT_TRUE(ret);

  server->SetMessageHandler([](MessageBase *const message) -> MessageBase *const {
    IncrDataMsgNum(1);
    return NULL_MSG;
  });

  // Start the tcp client.
  auto client_url = "127.0.0.1:1234";
  std::unique_ptr<TCPClient> client = std::make_unique<TCPClient>();
  ret = client->Initialize();
  ASSERT_TRUE(ret);

  // Create the message.
  auto message = CreateMessage(server_url, client_url);
  auto msg_size = message->size;

  // Send the message.
  client->Connect(server_url);
  size_t bytes_num = 0;
  (void)client->SendSync(std::move(message), &bytes_num);

  EXPECT_EQ(msg_size, bytes_num);

  WaitForDataMsg(1, 5);
  EXPECT_EQ(1, GetDataMsgNum());

  // Destroy
  client->Disconnect(server_url);
  client->Finalize();
  server->Finalize();
}

/// Feature: test sending large messages.
/// Description: start a socket server and send several large messages to it.
/// Expectation: the server received these large messages sented from client.
TEST_F(TCPTest, SendLargeMessages) {
  Init();

  // Start the tcp server.
  std::unique_ptr<TCPServer> server = std::make_unique<TCPServer>();
  bool ret = server->Initialize();
  ASSERT_TRUE(ret);

  server->SetMessageHandler([](MessageBase *const message) -> MessageBase *const {
    IncrDataMsgNum(1);
    return NULL_MSG;
  });

  // Start the tcp client.
  auto client_url = "127.0.0.1:1234";
  std::unique_ptr<TCPClient> client = std::make_unique<TCPClient>();
  ret = client->Initialize();
  ASSERT_TRUE(ret);

  // Send the message.
  auto ip = server->GetIP();
  auto port = server->GetPort();
  auto server_url = ip + ":" + std::to_string(port);
  client->Connect(server_url);

  size_t msg_cnt = 5;
  size_t large_msg_size = 1024000;
  for (int i = 0; i < msg_cnt; ++i) {
    auto message = CreateMessage(server_url, client_url, large_msg_size);
    client->SendAsync(std::move(message));
  }

  // Wait timeout: 15s
  WaitForDataMsg(msg_cnt, 15);

  // Check result
  EXPECT_EQ(msg_cnt, GetDataMsgNum());

  // Destroy
  client->Disconnect(server_url);
  client->Finalize();
  server->Finalize();
}

/// Feature: test delete invalid tcp connection used in connection pool in tcp client when some socket error happened.
/// Description: start a socket server and tcp client pair and stop the tcp server.
/// Expectation: the connection from the tcp client to the tcp server will be deleted automatically.
TEST_F(TCPTest, DeleteInvalidConnectionForTcpClient) {
  pid_t pid = fork();
  EXPECT_LE(0, pid);

  auto server_url = "127.0.0.1:8081";
  if (pid == 0) {
    // Start the tcp server.
    std::unique_ptr<TCPServer> server = std::make_unique<TCPServer>();
    bool ret = server->Initialize(server_url);
    ASSERT_TRUE(ret);

    server->SetMessageHandler([](MessageBase *const message) -> MessageBase *const {
      IncrDataMsgNum(1);
      return NULL_MSG;
    });

    // Wait timeout: 30s
    WaitForDataMsg(1, 30);

    // Check result
    EXPECT_EQ(1, GetDataMsgNum());

    sleep(60 * 60);

  } else {
    // Start the tcp client.
    auto client_url = "127.0.0.1:1234";
    std::unique_ptr<TCPClient> client = std::make_unique<TCPClient>();
    auto ret = client->Initialize();
    ASSERT_TRUE(ret);

    // Create the message.
    auto message = CreateMessage(server_url, client_url);

    // Send the message.
    client->Connect(server_url);
    client->SendAsync(std::move(message));

    // Kill the tcp server process.
    kill(pid, 9);

    bool disconnected = false;
    size_t retry = 20;
    size_t interval = 3;
    while (!disconnected && retry-- > 0) {
      if (!client->IsConnected(server_url)) {
        disconnected = true;
        continue;
      }
      sleep(interval);
    }
    // Destroy the tcp client.
    client->Finalize();

    ASSERT_TRUE(disconnected);
  }
}
}  // namespace rpc
}  // namespace distributed
}  // namespace mindspore
