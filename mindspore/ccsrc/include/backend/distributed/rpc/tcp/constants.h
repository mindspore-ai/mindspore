/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_DISTRIBUTED_RPC_TCP_CONSTANTS_H_
#define MINDSPORE_CCSRC_DISTRIBUTED_RPC_TCP_CONSTANTS_H_

#include <arpa/inet.h>
#include <string>
#include <csignal>
#include <queue>
#include <memory>
#include <functional>

#include "include/backend/distributed/constants.h"

namespace mindspore {
namespace distributed {
namespace rpc {
using DeleteCallBack = void (*)(const std::string &from, const std::string &to);
using ConnectionCallBack = std::function<void(void *connection)>;

constexpr int SEND_MSG_IO_VEC_LEN = 5;
constexpr int RECV_MSG_IO_VEC_LEN = 4;

constexpr unsigned int MAGICID_LEN = 4;
constexpr int SENDMSG_QUEUELEN = 1024;
constexpr int SENDMSG_DROPED = -1;

constexpr size_t MAX_KMSG_FROM_LEN = 1024;
constexpr size_t MAX_KMSG_TO_LEN = 1024;
constexpr size_t MAX_KMSG_NAME_LEN = 1024;
constexpr size_t MAX_KMSG_BODY_LEN = 1073741824;

enum ParseType { kTcpMsg = 1, kHttpReq, kHttpRsp, kUnknown };
enum State { kMsgHeader, kBody };
enum ConnectionState { kInit = 1, kConnecting, kConnected, kDisconnecting, kClose };
enum ConnectionType { kTcp = 1, kSSL };
enum ConnectionPriority { kPriorityLow = 1, kPriorityHigh };

static const int g_httpKmsgEnable = -1;

using IntTypeMetrics = std::queue<int>;
using StringTypeMetrics = std::queue<std::string>;

static MessageBase *const NULL_MSG = nullptr;

// Server socket listen backlog.
static const int SOCKET_LISTEN_BACKLOG = 2048;

static const int SOCKET_KEEPALIVE = 1;

// Send first probe after `interval' seconds.
static const int SOCKET_KEEPIDLE = 600;

// Send next probes after the specified interval.
static const int SOCKET_KEEPINTERVAL = 5;

// Consider the socket in error state after we send three ACK
// probes without getting a reply.
static const int SOCKET_KEEPCOUNT = 3;

static const char RPC_MAGICID[] = "RPC0";
static const char TCP_RECV_EVLOOP_THREADNAME[] = "RECV_EVENT_LOOP";
static const char TCP_SEND_EVLOOP_THREADNAME[] = "SEND_EVENT_LOOP";

constexpr int RPC_OK = 0;
constexpr int RPC_ERROR = -1;

constexpr int IO_RW_OK = 1;
constexpr int IO_RW_ERROR = -1;

constexpr int IP_LEN_MAX = 128;

// Kill the process for safe exiting.
inline void KillProcess(const std::string &ret) {
  MS_LOG(ERROR) << ret;
  (void)raise(SIGKILL);
}

/*
 * The MessageHeader contains the stats info about the message body.
 */
struct MessageHeader {
  MessageHeader() {
    for (unsigned int i = 0; i < MAGICID_LEN; ++i) {
      if (i < sizeof(RPC_MAGICID) - 1) {
        magic[i] = RPC_MAGICID[i];
      } else {
        magic[i] = '\0';
      }
    }
  }

  char magic[MAGICID_LEN];
  uint32_t name_len{0};
  uint32_t to_len{0};
  uint32_t from_len{0};
  uint32_t body_len{0};
};

// Fill the message header using the given message.
__attribute__((unused)) static void FillMessageHeader(const MessageBase &message, MessageHeader *header) {
  std::string send_to = message.to;
  std::string send_from = message.from;
  header->name_len = htonl(static_cast<uint32_t>(message.name.size()));
  header->to_len = htonl(static_cast<uint32_t>(send_to.size()));
  header->from_len = htonl(static_cast<uint32_t>(send_from.size()));
  if (message.data != nullptr) {
    header->body_len = htonl(static_cast<uint32_t>(message.size));
  } else {
    header->body_len = htonl(static_cast<uint32_t>(message.body.size()));
  }
}

// Compute and return the byte size of the whole message.
__attribute__((unused)) static size_t GetMessageSize(const MessageBase &message) {
  std::string send_to = message.to;
  std::string send_from = message.from;
  size_t size = message.name.size() + send_to.size() + send_from.size() + message.body.size() + sizeof(MessageHeader);
  return size;
}

#define RPC_ASSERT(expression)                                                                       \
  do {                                                                                               \
    if (!(expression)) {                                                                             \
      std::stringstream ss;                                                                          \
      ss << "Assertion failed: " << #expression << ", file: " << __FILE__ << ", line: " << __LINE__; \
      KillProcess(ss.str());                                                                         \
    }                                                                                                \
  } while (0)

#define RPC_EXIT(ret)                                                           \
  do {                                                                          \
    std::stringstream ss;                                                       \
    ss << (ret) << "  ( file: " << __FILE__ << ", line: " << __LINE__ << " )."; \
    KillProcess(ss.str());                                                      \
  } while (0)
}  // namespace rpc
}  // namespace distributed
}  // namespace mindspore
#endif
