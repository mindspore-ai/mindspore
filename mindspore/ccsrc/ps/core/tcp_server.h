/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PS_CORE_TCP_SERVER_H_
#define MINDSPORE_CCSRC_PS_CORE_TCP_SERVER_H_

#include <event2/buffer.h>
#include <event2/bufferevent.h>
#include <event2/event.h>
#include <event2/listener.h>
#include <exception>
#include <functional>
#include <iostream>
#include <map>
#include <mutex>
#include <string>
#include <memory>
#include <vector>

#include "utils/log_adapter.h"
#include "ps/core/tcp_message_handler.h"

namespace mindspore {
namespace ps {
namespace core {

class TcpServer;
class TcpConnection {
 public:
  explicit TcpConnection(struct bufferevent *bev, const evutil_socket_t &fd, const TcpServer *server)
      : buffer_event_(bev), fd_(0), server_(server) {}
  virtual ~TcpConnection() = default;

  virtual void InitConnection();
  virtual void SendMessage(const void *buffer, size_t num) const;
  void SendMessage(const CommMessage &message) const;
  virtual void OnReadHandler(const void *buffer, size_t numBytes);
  TcpServer *GetServer() const;
  const evutil_socket_t &GetFd() const;

 protected:
  struct bufferevent *buffer_event_;
  evutil_socket_t fd_;
  const TcpServer *server_;
  TcpMessageHandler tcp_message_handler_;
};

using OnServerReceiveMessage =
  std::function<void(const TcpServer &tcp_server, const TcpConnection &conn, const CommMessage &)>;

class TcpServer {
 public:
  using OnConnected = std::function<void(const TcpServer &, const TcpConnection &)>;
  using OnDisconnected = std::function<void(const TcpServer &, const TcpConnection &)>;
  using OnAccepted = std::function<const TcpConnection *(const TcpServer &)>;

  explicit TcpServer(const std::string &address, std::uint16_t port);
  virtual ~TcpServer();

  void SetServerCallback(const OnConnected &client_conn, const OnDisconnected &client_disconn,
                         const OnAccepted &client_accept);
  void Init();
  void Start();
  void StartWithNoBlock();
  void Stop();
  void SendToAllClients(const char *data, size_t len);
  void AddConnection(const evutil_socket_t &fd, const TcpConnection *connection);
  void RemoveConnection(const evutil_socket_t &fd);
  OnServerReceiveMessage GetServerReceive() const;
  void SetMessageCallback(const OnServerReceiveMessage &cb);
  static void SendMessage(const TcpConnection &conn, const CommMessage &message);
  void SendMessage(const CommMessage &message);
  uint16_t BoundPort() const;

 protected:
  static void ListenerCallback(struct evconnlistener *listener, evutil_socket_t socket, struct sockaddr *saddr,
                               int socklen, void *server);
  static void SignalCallback(evutil_socket_t sig, std::int16_t events, void *server);
  static void ReadCallback(struct bufferevent *, void *connection);
  static void EventCallback(struct bufferevent *, std::int16_t events, void *server);
  virtual TcpConnection *onCreateConnection(struct bufferevent *bev, const evutil_socket_t &fd);

  struct event_base *base_;
  struct event *signal_event_;
  struct evconnlistener *listener_;
  std::string server_address_;
  std::uint16_t server_port_;

  std::map<evutil_socket_t, const TcpConnection *> connections_;
  OnConnected client_connection_;
  OnDisconnected client_disconnection_;
  OnAccepted client_accept_;
  std::recursive_mutex connection_mutex_;
  OnServerReceiveMessage message_callback_;
};

}  // namespace core
}  // namespace ps
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PS_CORE_TCP_SERVER_H_
