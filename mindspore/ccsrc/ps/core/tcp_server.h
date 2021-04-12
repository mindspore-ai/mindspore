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
#include <event2/thread.h>

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

#include "ps/core/tcp_message_handler.h"
#include "ps/core/cluster_metadata.h"
#include "utils/convert_utils_base.h"
#include "ps/core/comm_util.h"

namespace mindspore {
namespace ps {
namespace core {
class TcpServer;
class TcpConnection {
 public:
  explicit TcpConnection(struct bufferevent *bev, const evutil_socket_t &fd, TcpServer *server)
      : buffer_event_(bev), fd_(fd), server_(server) {}
  TcpConnection(const TcpConnection &);
  virtual ~TcpConnection() = default;

  using Callback = std::function<void(const std::shared_ptr<CommMessage>)>;

  virtual void InitConnection(const messageReceive &callback);
  virtual void SendMessage(const void *buffer, size_t num) const;
  bool SendMessage(std::shared_ptr<CommMessage> message) const;
  bool SendMessage(std::shared_ptr<MessageMeta> meta, const Protos &protos, const void *data, size_t size) const;
  virtual void OnReadHandler(const void *buffer, size_t numBytes);
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
  std::function<void(std::shared_ptr<TcpConnection> conn, std::shared_ptr<MessageMeta> meta, const Protos &protos,
                     const void *data, size_t size)>;

class TcpServer {
 public:
  using OnConnected = std::function<void(const TcpServer &, const TcpConnection &)>;
  using OnDisconnected = std::function<void(const TcpServer &, const TcpConnection &)>;
  using OnAccepted = std::function<std::shared_ptr<TcpConnection>(const TcpServer &)>;
  using OnTimerOnce = std::function<void(const TcpServer &)>;
  using OnTimer = std::function<void()>;

  TcpServer(const std::string &address, std::uint16_t port);
  TcpServer(const TcpServer &server);
  virtual ~TcpServer();

  void SetServerCallback(const OnConnected &client_conn, const OnDisconnected &client_disconn,
                         const OnAccepted &client_accept);
  void set_timer_once_callback(const OnTimerOnce &timer);
  void set_timer_callback(const OnTimer &timer);
  void Init();
  void Start();
  void StartWithNoBlock();
  void StartTimerOnlyOnce(const uint32_t &time);
  void StartTimer(const uint32_t &time);
  void Stop();
  void SendToAllClients(const char *data, size_t len);
  void AddConnection(const evutil_socket_t &fd, std::shared_ptr<TcpConnection> connection);
  void RemoveConnection(const evutil_socket_t &fd);
  std::shared_ptr<TcpConnection> GetConnectionByFd(const evutil_socket_t &fd);
  OnServerReceiveMessage GetServerReceive() const;
  void SetMessageCallback(const OnServerReceiveMessage &cb);
  bool SendMessage(std::shared_ptr<TcpConnection> conn, std::shared_ptr<CommMessage> message);
  bool SendMessage(std::shared_ptr<TcpConnection> conn, std::shared_ptr<MessageMeta> meta, const Protos &protos,
                   const void *data, size_t sizee);
  void SendMessage(std::shared_ptr<CommMessage> message);
  uint16_t BoundPort() const;
  std::string BoundIp() const;
  int ConnectionNum() const;
  const std::map<evutil_socket_t, std::shared_ptr<TcpConnection>> &Connections() const;

 protected:
  static void ListenerCallback(struct evconnlistener *listener, evutil_socket_t socket, struct sockaddr *saddr,
                               int socklen, void *server);
  static void SignalCallback(evutil_socket_t sig, std::int16_t events, void *server);
  static void ReadCallback(struct bufferevent *, void *connection);
  static void EventCallback(struct bufferevent *, std::int16_t events, void *server);
  static void TimerCallback(evutil_socket_t fd, int16_t event, void *arg);
  static void TimerOnceCallback(evutil_socket_t fd, int16_t event, void *arg);
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
  OnTimerOnce on_timer_once_callback_;
  OnTimer on_timer_callback_;
};
}  // namespace core
}  // namespace ps
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PS_CORE_TCP_SERVER_H_
