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
#include <mutex>

#include "actor/msg.h"
#include "distributed/rpc/tcp/connection.h"
#include "distributed/rpc/tcp/connection_pool.h"
#include "distributed/rpc/tcp/event_loop.h"

namespace mindspore {
namespace distributed {
namespace rpc {
// Event handler for new connecting request arrived.
void OnAccept(int server, uint32_t events, void *arg);

void DoDisconnect(int fd, Connection *conn, uint32_t error, int soError);

void ConnectedEventHandler(int fd, uint32_t events, void *context);

class TCPComm {
 public:
  explicit TCPComm(bool enable_ssl = false)
      : server_fd_(-1), recv_event_loop_(nullptr), send_event_loop_(nullptr), enable_ssl_(enable_ssl) {}
  TCPComm(const TCPComm &) = delete;
  TCPComm &operator=(const TCPComm &) = delete;
  ~TCPComm() = default;

  // Init the event loop for reading and writing.
  bool Initialize();

  // Destroy all the resources.
  void Finalize();

  // Create the server socket represented by url.
  // allocate_cb is the method used to allocate memory when server receiving message from the remote.
  bool StartServerSocket(const std::string &url, const MemAllocateCallback &allocate_cb);

  // Create the server socket with local IP and random port.
  bool StartServerSocket(const MemAllocateCallback &allocate_cb);

  // Connection operation for a specified destination.
  bool Connect(const std::string &dst_url, const MemFreeCallback &free_cb);
  bool IsConnected(const std::string &dst_url);
  bool Disconnect(const std::string &dst_url);

  // Send the message from the source to the destination.
  // The flag sync means if the message is sent directly or added to the task queue.
  bool Send(MessageBase *msg, size_t *const send_bytes, bool sync = false);

  // Force the data in the send buffer to be sent out.
  bool Flush(const std::string &dst_url);

  // Set the message processing handler.
  void SetMessageHandler(const MessageHandler &handler);

  // Get the file descriptor of server socket.
  int GetServerFd() const;

  /**
   * @description: Returns the allocating callback.
   * @return {const MemAllocateCallback &}
   */
  const MemAllocateCallback &allocate_cb() const { return allocate_cb_; }

 private:
  // Build the connection.
  Connection *CreateDefaultConn(const std::string &to);

  // Send a message.
  static void SendExitMsg(const std::string &from, const std::string &to);

  // Called by ReadCallBack when new message arrived.
  static int ReceiveMessage(Connection *conn);

  static int SetConnectedHandler(Connection *conn);

  static int DoConnect(Connection *conn, const struct sockaddr *sa, socklen_t saLen);

  static void DropMessage(MessageBase *msg);

  // Read and write events.
  void ReadCallBack(void *conn);
  void WriteCallBack(void *conn);
  // Connected and Disconnected events.
  void EventCallBack(void *conn);

  // The server url.
  std::string url_;

  // The socket of server.
  int server_fd_;

  // User defined handler for Handling received messages.
  MessageHandler message_handler_;

  // All the connections share the same read and write event loop objects.
  EventLoop *recv_event_loop_;
  EventLoop *send_event_loop_;

  // The connection pool used to store new connections.
  std::shared_ptr<ConnectionPool> conn_pool_;

  // The mutex for connection operations.
  std::shared_ptr<std::mutex> conn_mutex_;

  // The method used to allocate memory when tcp servers of this TcpComm receive message from the remote.
  MemAllocateCallback allocate_cb_;

  bool enable_ssl_;

  friend void OnAccept(int server, uint32_t events, void *arg);
  friend int DoConnect(const std::string &to, Connection *conn, ConnectionCallBack event_callback,
                       ConnectionCallBack write_callback, ConnectionCallBack read_callback);
};
}  // namespace rpc
}  // namespace distributed
}  // namespace mindspore

#endif
