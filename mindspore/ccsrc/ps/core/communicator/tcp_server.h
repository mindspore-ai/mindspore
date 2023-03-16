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

#ifndef MINDSPORE_CCSRC_PS_CORE_COMMUNICATOR_TCP_SERVER_H_
#define MINDSPORE_CCSRC_PS_CORE_COMMUNICATOR_TCP_SERVER_H_

#include <event2/buffer.h>
#include <event2/bufferevent.h>
#include <event2/event.h>
#include <event2/listener.h>
#include <event2/thread.h>
#include <event2/bufferevent_ssl.h>

#include <exception>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <thread>
#include <atomic>

#include "ps/core/communicator/tcp_message_handler.h"
#include "ps/core/communicator/ssl_wrapper.h"
#include "ps/core/cluster_config.h"
#include "utils/convert_utils_base.h"
#include "ps/core/comm_util.h"
#include "include/backend/distributed/ps/constants.h"
#include "include/backend/distributed/ps/ps_context.h"
#include "ps/core/file_configuration.h"

namespace mindspore {
namespace ps {
namespace core {
class TcpServer;
class TcpConnection {
 public:
  explicit TcpConnection(struct bufferevent *bev, const evutil_socket_t &fd, TcpServer *server)
      : buffer_event_(bev), fd_(fd), server_(server) {
    MS_LOG(WARNING) << "TcpConnection is constructed! fd is " << fd;
  }
  TcpConnection(const TcpConnection &);
  virtual ~TcpConnection();

  using Callback = std::function<void(const std::shared_ptr<CommMessage>)>;

  void InitConnection(const messageReceive &callback);
  void SendMessage(const void *buffer, size_t num) const;
  bool SendMessage(const std::shared_ptr<CommMessage> &message) const;
  bool SendMessage(const std::shared_ptr<MessageMeta> &meta, const Protos &protos, const void *data, size_t size) const;
  void OnReadHandler(const void *buffer, size_t numBytes);
  const TcpServer *GetServer() const;
  const evutil_socket_t &GetFd() const;
  void set_callback(const Callback &callback);

 protected:
  struct bufferevent *buffer_event_;
  evutil_socket_t fd_;
  TcpServer *server_;
  TcpMessageHandler tcp_message_handler_;
  Callback callback_;
};

using OnServerReceiveMessage =
  std::function<void(const std::shared_ptr<TcpConnection> &conn, const std::shared_ptr<MessageMeta> &meta,
                     const Protos &protos, const void *data, size_t size)>;

class TcpServer {
 public:
  using OnConnected = std::function<void(const TcpServer &, const TcpConnection &)>;
  using OnDisconnected = std::function<void(const TcpServer &, const TcpConnection &)>;
  using OnAccepted = std::function<std::shared_ptr<TcpConnection>(const TcpServer &)>;
  using OnTimerOnce = std::function<void(const TcpServer &)>;
  using OnTimer = std::function<void()>;

  TcpServer(const std::string &address, std::uint16_t port, Configuration *const config);
  TcpServer(const TcpServer &server);
  virtual ~TcpServer();

  void SetServerCallback(const OnConnected &client_conn, const OnDisconnected &client_disconn,
                         const OnAccepted &client_accept);
  void Init();
  void Start();
  void Stop();
  void SendToAllClients(const char *data, size_t len);
  void AddConnection(const evutil_socket_t &fd, std::shared_ptr<TcpConnection> connection);
  void RemoveConnection(const evutil_socket_t &fd);
  std::shared_ptr<TcpConnection> &GetConnectionByFd(const evutil_socket_t &fd);
  OnServerReceiveMessage GetServerReceive() const;
  void SetMessageCallback(const OnServerReceiveMessage &cb);
  bool SendMessage(const std::shared_ptr<TcpConnection> &conn, const std::shared_ptr<CommMessage> &message);
  bool SendMessage(const std::shared_ptr<TcpConnection> &conn, const std::shared_ptr<MessageMeta> &meta,
                   const Protos &protos, const void *data, size_t sizee);
  void SendMessage(const std::shared_ptr<CommMessage> &message);
  uint16_t BoundPort() const;
  std::string BoundIp() const;
  int ConnectionNum() const;
  const std::map<evutil_socket_t, std::shared_ptr<TcpConnection>> &Connections() const;

 protected:
  static void ListenerCallback(struct evconnlistener *listener, evutil_socket_t socket, struct sockaddr *saddr,
                               int socklen, void *server);
  static void ListenerCallbackInner(evutil_socket_t socket, struct sockaddr *saddr, void *server);
  static void SignalCallback(evutil_socket_t sig, std::int16_t events, void *server);
  static void SignalCallbackInner(void *server);
  static void ReadCallback(struct bufferevent *, void *connection);
  static void ReadCallbackInner(struct bufferevent *, void *connection);
  static void EventCallback(struct bufferevent *, std::int16_t events, void *server);
  static void EventCallbackInner(struct bufferevent *, std::int16_t events, void *server);
  static void SetTcpNoDelay(const evutil_socket_t &fd);
  std::shared_ptr<TcpConnection> onCreateConnection(struct bufferevent *bev, const evutil_socket_t &fd);

  struct event_base *base_;
  struct event *signal_event_;
  struct evconnlistener *listener_;
  std::string server_address_;
  std::uint16_t server_port_;
  std::atomic<bool> is_stop_;

  std::map<evutil_socket_t, std::shared_ptr<TcpConnection>> connections_;
  OnConnected client_connection_;
  OnDisconnected client_disconnection_;
  OnAccepted client_accept_;
  std::mutex connection_mutex_;
  OnServerReceiveMessage message_callback_;
  // The Configuration file
  Configuration *config_;
  int64_t max_connection_;
};
}  // namespace core
}  // namespace ps
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PS_CORE_COMMUNICATOR_TCP_SERVER_H_
