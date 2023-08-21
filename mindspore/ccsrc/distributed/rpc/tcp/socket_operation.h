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

#ifndef MINDSPORE_CCSRC_DISTRIBUTED_RPC_TCP_SOCKET_OPERATION_H_
#define MINDSPORE_CCSRC_DISTRIBUTED_RPC_TCP_SOCKET_OPERATION_H_

#include <netinet/in.h>
#include <string>
#include "utils/convert_utils_base.h"

namespace mindspore {
namespace distributed {
namespace rpc {
// This is the return value that represents the address is in use errno.
constexpr int kAddressInUseError = -2;

class Connection;

union SocketAddress {
  struct sockaddr sa;
  struct sockaddr_in saIn;
  struct sockaddr_in6 saIn6;
  struct sockaddr_storage saStorage;
};

class SocketOperation {
 public:
  SocketOperation() = default;
  virtual ~SocketOperation() {}

  virtual bool Initialize() { return true; }

  // Lookup the local IP address of the first available network interface.
  static std::string GetLocalIP();

  static std::string GetIP(const std::string &url);

  // Get ip and port of the specified socket fd.
  static std::string GetIP(int sock_fd);
  static uint16_t GetPort(int sock_fd);

  // Get the address(ip:port) of the other end of the connection.
  static std::string GetPeer(int sock_fd);

  // Get socket address of the url.
  static bool GetSockAddr(const std::string &url, SocketAddress *addr);

  // Create a socket.
  static int CreateSocket(sa_family_t family);

  // Set socket options.
  static int SetSocketOptions(int sock_fd);
  static int SetSocketKeepAlive(int fd, int keepalive, int keepidle, int keepinterval, int keepcount);

  // Connect to the Socket sock_fd.
  static int Connect(int sock_fd, const struct sockaddr *sa, socklen_t saLen, uint16_t *boundPort);

  // Get interface name for specified socket address.
  static std::string GetInterfaceName(SocketAddress *const addr);

  // Close the given connection.
  virtual void Close(Connection *connection) = 0;

  // Start and listen on the socket represented by the given url.
  static int Listen(const std::string &url);

  // Accept connection on the server socket.
  static int Accept(int sock_fd);

  // Call recv with flag MSG_PEEK which means do not delete data in buffer after reading.
  virtual ssize_t ReceivePeek(Connection *connection, char *recvBuf, uint32_t recvLen) = 0;

  // Try to receive messages up to totalRecvLen (for message header).
  virtual int Receive(Connection *connection, char *recvBuf, size_t totalRecvLen, size_t *recvLen) = 0;

  // Receive message (for message body).
  virtual int ReceiveMessage(Connection *connection, struct msghdr *recvMsg, size_t totalRecvLen, size_t *recvLen) = 0;

  virtual int SendMessage(Connection *connection, struct msghdr *sendMsg, size_t totalSendLen, size_t *sendLen) = 0;

  // Handle connect and connected events.
  virtual void NewConnEventHandler(int fd, uint32_t events, void *context) = 0;
  virtual void ConnEstablishedEventHandler(int fd, uint32_t events, void *context) = 0;
};
}  // namespace rpc
}  // namespace distributed
}  // namespace mindspore

#endif
