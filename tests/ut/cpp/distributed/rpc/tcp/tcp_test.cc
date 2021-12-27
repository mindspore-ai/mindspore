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
#include "actor/iomgr.h"
#include "async/async.h"
#include "distributed/rpc/tcp/tcp_comm.h"
#include "common/common_test.h"

namespace mindspore {
namespace distributed {
namespace rpc {
int g_recv_num = 0;
int g_exit_msg_num = 0;

TCPComm *m_io = nullptr;
std::atomic<int> m_sendNum(0);
std::string m_localIP = "127.0.0.1";
bool m_notRemote = false;

void msgHandle(std::unique_ptr<MessageBase> &&msg) {
  if (msg->GetType() == MessageBase::Type::KEXIT) {
    g_exit_msg_num++;
  } else {
    g_recv_num++;
  }
}

class TCPTest : public UT::Common {
 public:
  static void SendMsg(std::string &_localUrl, std::string &_remoteUrl, int msgsize, bool remoteLink = false,
                      std::string body = "");

 protected:
  char *args[4];
  char *testServerPath;
  static const size_t pid_num = 100;
  pid_t pid1;
  pid_t pid2;

  pid_t pids[pid_num];

  void SetUp() {
    char *localpEnv = getenv("LITEBUS_IP");
    if (localpEnv != nullptr) {
      m_localIP = std::string(localpEnv);
    }

    char *locaNotRemoteEnv = getenv("LITEBUS_SEND_ON_REMOTE");
    if (locaNotRemoteEnv != nullptr) {
      m_notRemote = (std::string(locaNotRemoteEnv) == "true") ? true : false;
    }

    pid1 = 0;
    pid2 = 0;
    pids[pid_num] = {0};
    size_t size = pid_num * sizeof(pid_t);
    if (memset_s(&pids, size, 0, size)) {
      MS_LOG(ERROR) << "Failed to init pid array";
    }
    g_recv_num = 0;
    g_exit_msg_num = 0;
    m_sendNum = 0;

    m_io = new TCPComm();
    m_io->Initialize();
    m_io->SetMessageHandler(msgHandle);
    m_io->StartServerSocket("tcp://" + m_localIP + ":2225", "tcp://" + m_localIP + ":2225");
  }

  void TearDown() {
    shutdownTcpServer(pid1);
    shutdownTcpServer(pid2);
    pid1 = 0;
    pid2 = 0;
    int i = 0;
    for (i = 0; i < pid_num; i++) {
      shutdownTcpServer(pids[i]);
      pids[i] = 0;
    }
    g_recv_num = 0;
    g_exit_msg_num = 0;
    m_sendNum = 0;
    m_io->Finalize();
    delete m_io;
    m_io = nullptr;
  }

  bool CheckRecvNum(int expectedRecvNum, int _timeout);
  bool CheckExitNum(int expectedExitNum, int _timeout);
  pid_t startTcpServer(char **args);
  void shutdownTcpServer(pid_t pid);
  void KillTcpServer(pid_t pid);

  void Link(std::string &_localUrl, std::string &_remoteUrl);
  void Reconnect(std::string &_localUrl, std::string &_remoteUrl);
  void Unlink(std::string &_remoteUrl);
};

// listening local url and sending msg to remote url,if start succ.
pid_t TCPTest::startTcpServer(char **args) {
  pid_t pid = fork();
  if (pid == 0) {
    return -1;
  } else {
    return pid;
  }
}
void TCPTest::shutdownTcpServer(pid_t pid) {
  if (pid > 1) {
    kill(pid, SIGALRM);
    int status;
    waitpid(pid, &status, 0);
  }
}

void TCPTest::KillTcpServer(pid_t pid) {
  if (pid > 1) {
    kill(pid, SIGKILL);
    int status;
    waitpid(pid, &status, 0);
  }
}

void TCPTest::SendMsg(std::string &_localUrl, std::string &_remoteUrl, int msgsize, bool remoteLink, std::string body) {
  AID from("testserver", _localUrl);
  AID to("testserver", _remoteUrl);

  std::unique_ptr<MessageBase> message = std::make_unique<MessageBase>();
  std::string data(msgsize, 'A');
  message->name = "testname";
  message->from = from;
  message->to = to;
  message->body = data;
  if (body != "") {
    message->body = body;
  }

  if (m_notRemote) {
    m_io->Send(std::move(message), remoteLink, true);
  } else {
    m_io->Send(std::move(message), remoteLink);
  }
}

void TCPTest::Link(std::string &_localUrl, std::string &_remoteUrl) {
  AID from("testserver", _localUrl);
  AID to("testserver", _remoteUrl);
  m_io->Link(from, to);
}

void TCPTest::Reconnect(std::string &_localUrl, std::string &_remoteUrl) {
  AID from("testserver", _localUrl);
  AID to("testserver", _remoteUrl);
  m_io->Reconnect(from, to);
}

void TCPTest::Unlink(std::string &_remoteUrl) {
  AID to("testserver", _remoteUrl);
  m_io->UnLink(to);
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
  std::unique_ptr<TCPComm> io = std::make_unique<TCPComm>();
  io->Initialize();

  bool ret = io->StartServerSocket("tcp://0:2225", "tcp://0:2225");
  ASSERT_FALSE(ret);
  io->Finalize();
}

/// Feature: test start a socket server.
/// Description: start the socket server with a specified socket.
/// Expectation: the socket server is started successfully.
TEST_F(TCPTest, StartServer2) {
  std::unique_ptr<TCPComm> io = std::make_unique<TCPComm>();
  io->Initialize();
  io->SetMessageHandler(msgHandle);
  bool ret = io->StartServerSocket("tcp://" + m_localIP + ":2225", "tcp://" + m_localIP + ":2225");
  ASSERT_FALSE(ret);
  ret = io->StartServerSocket("tcp://" + m_localIP + ":2224", "tcp://" + m_localIP + ":2224");
  io->Finalize();
  ASSERT_TRUE(ret);
}

/// Feature: test normal tcp message sending.
/// Description: start a socket server and send a normal message to it.
/// Expectation: the server received the message sented from client.
TEST_F(TCPTest, send1Msg) {
  g_recv_num = 0;
  pid1 = startTcpServer(args);
  bool ret = CheckRecvNum(1, 5);
  ASSERT_FALSE(ret);

  std::string from = "tcp://" + m_localIP + ":2223";
  std::string to = "tcp://" + m_localIP + ":2225";
  SendMsg(from, to, pid_num);

  ret = CheckRecvNum(1, 5);
  ASSERT_TRUE(ret);

  Unlink(to);
  shutdownTcpServer(pid1);
  pid1 = 0;
}
}  // namespace rpc
}  // namespace distributed
}  // namespace mindspore
