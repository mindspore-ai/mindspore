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

#ifndef MINDSPORE_CCSRC_DISTRIBUTED_RPC_TCP_TCP_COMM_H_
#define MINDSPORE_CCSRC_DISTRIBUTED_RPC_TCP_TCP_COMM_H_

#include <string>
#include <memory>
#include <vector>

#include "actor/iomgr.h"
#include "distributed/rpc/tcp/connection.h"
#include "distributed/rpc/tcp/event_loop.h"

namespace mindspore {
namespace distributed {
namespace rpc {
// Event handler for new connecting request arrived.
void OnAccept(int server, uint32_t events, void *arg);

// Send messages buffered in the connection.
void DoSend(Connection *conn);

// Create a server socket and connect to it, this is a local connection..
int DoConnect(const std::string &to, Connection *conn, ConnectionCallBack eventCallBack,
              ConnectionCallBack writeCallBack, ConnectionCallBack readCallBack);

void DoDisconnect(int fd, Connection *conn, uint32_t error, int soError);

void ConnectedEventHandler(int fd, uint32_t events, void *context);

class TCPComm : public IOMgr {
 public:
  TCPComm() : server_fd_(-1), recv_event_loop_(nullptr), send_event_loop_(nullptr) {}
  TCPComm(const TCPComm &) = delete;
  TCPComm &operator=(const TCPComm &) = delete;
  ~TCPComm();

  // Init the event loop for reading and writing.
  bool Initialize() override;

  // Destroy all the resources.
  void Finalize() override;

  // Create the server socket represented by url.
  bool StartServerSocket(const std::string &url, const std::string &aAdvertiseUrl) override;

  // Build a connection between the source and destination.
  void Link(const AID &source, const AID &destination) override;
  void UnLink(const AID &destination) override;

  // Send the message from the source to the destination.
  int Send(std::unique_ptr<MessageBase> &&msg, bool remoteLink = false, bool isExactNotRemote = false) override;

  uint64_t GetInBufSize() override;
  uint64_t GetOutBufSize() override;
  void CollectMetrics() override;

 private:
  // Build the connection.
  Connection *CreateDefaultConn(std::string to);
  void Reconnect(const AID &source, const AID &destination);
  void DoReConnectConn(Connection *conn, std::string to, const AID &source, const AID &destination, int *oldFd);

  // Send a message.
  int Send(MessageBase *msg, bool remoteLink = false, bool isExactNotRemote = false);
  static void Send(MessageBase *msg, const TCPComm *tcpmgr, bool remoteLink, bool isExactNotRemote);
  void SendByRecvLoop(MessageBase *msg, const TCPComm *tcpmgr, bool remoteLink, bool isExactNotRemote);
  static void SendExitMsg(const std::string &from, const std::string &to);

  // Called by ReadCallBack when new message arrived.
  static int ReceiveMessage(Connection *conn);

  void SetMessageHandler(IOMgr::MessageHandler handler);
  static int SetConnectedHandler(Connection *conn);

  static int Connect(Connection *conn, const struct sockaddr *sa, socklen_t saLen);

  static bool IsHttpMsg();

  // Read and write events.
  static void ReadCallBack(void *context);
  static void WriteCallBack(void *context);
  // Connected and Disconnected events.
  static void EventCallBack(void *context);

  // The server url.
  std::string url_;

  // The socket of server.
  int server_fd_;

  // The message size waiting to be sent.
  static uint64_t output_buf_size_;

  // User defined handler for Handling received messages.
  static MessageHandler message_handler_;

  // The source url of a message.
  static std::vector<char> advertise_url_;

  static bool is_http_msg_;

  // All the connections share the same read and write event loop objects.
  EventLoop *recv_event_loop_;
  EventLoop *send_event_loop_;

  friend void OnAccept(int server, uint32_t events, void *arg);
  friend void DoSend(Connection *conn);
  friend int DoConnect(const std::string &to, Connection *conn, ConnectionCallBack event_callback,
                       ConnectionCallBack write_callback, ConnectionCallBack read_callback);
};
}  // namespace rpc
}  // namespace distributed
}  // namespace mindspore

#endif
