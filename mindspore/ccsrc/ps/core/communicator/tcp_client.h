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

#ifndef MINDSPORE_CCSRC_PS_CORE_COMMUNICATOR_TCP_CLIENT_H_
#define MINDSPORE_CCSRC_PS_CORE_COMMUNICATOR_TCP_CLIENT_H_

#include <event2/event.h>
#include <event2/bufferevent.h>
#include <event2/thread.h>
#include <event2/bufferevent_ssl.h>

#include <functional>
#include <string>
#include <memory>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>

#include "ps/core/cluster_config.h"
#include "utils/convert_utils_base.h"
#include "ps/core/comm_util.h"
#include "ps/core/communicator/ssl_client.h"
#include "ps/core/communicator/ssl_wrapper.h"
#include "include/backend/distributed/ps/constants.h"
#include "include/backend/distributed/ps/ps_context.h"
#include "ps/core/communicator/tcp_message_handler.h"
#include "ps/core/file_configuration.h"

namespace mindspore {
namespace ps {
namespace core {
class TcpClient {
 public:
  using OnConnected = std::function<void()>;
  using OnDisconnected = std::function<void()>;
  using OnRead = std::function<void(const void *, size_t)>;
  using OnTimeout = std::function<void()>;
  using OnMessage =
    std::function<void(const std::shared_ptr<MessageMeta> &, const Protos &, const void *, size_t size)>;
  using OnTimer = std::function<void()>;

  explicit TcpClient(const std::string &address, std::uint16_t port, NodeRole peer_role);
  virtual ~TcpClient();

  std::string GetServerAddress() const;
  void set_disconnected_callback(const OnDisconnected &disconnected);
  void set_connected_callback(const OnConnected &connected);
  bool WaitConnected(
    const uint32_t &connected_timeout = PSContext::instance()->cluster_config().cluster_available_timeout);
  void Init();
  void StartWithDelay(int seconds);
  void Stop();
  void Start();
  void StartWithNoBlock();
  void SetMessageCallback(const OnMessage &cb);
  bool SendMessage(const CommMessage &message) const;
  bool SendMessage(const std::shared_ptr<MessageMeta> &meta, const Protos &protos, const void *data, size_t size);
  void set_timer_callback(const OnTimer &timer);
  const event_base &eventbase() const;
  int connection_status() { return connection_status_; }
  static bool is_started() { return is_started_; }

 protected:
  static void SetTcpNoDelay(const evutil_socket_t &fd);
  static void TimeoutCallback(evutil_socket_t fd, std::int16_t what, void *arg);
  static void ReadCallback(struct bufferevent *bev, void *ctx);
  void ReadCallbackInner(struct bufferevent *bev);
  static void EventCallback(struct bufferevent *bev, std::int16_t events, void *ptr);
  void EventCallbackInner(struct bufferevent *bev, std::int16_t events);
  virtual void OnReadHandler(const void *buf, size_t num);
  static void TimerCallback(evutil_socket_t fd, int16_t event, void *arg);
  void NotifyConnected();
  bool EstablishSSL();

  std::string PeerRoleName() const;

 private:
  OnMessage message_callback_;
  TcpMessageHandler message_handler_;

  OnConnected connected_callback_;
  OnDisconnected disconnected_callback_;
  OnRead read_callback_;
  OnTimeout timeout_callback_;
  OnTimer on_timer_callback_;

  static event_base *event_base_;
  static std::mutex event_base_mutex_;
  static bool is_started_;

  std::mutex connection_mutex_;
  std::condition_variable connection_cond_;
  event *event_timeout_;
  bufferevent *buffer_event_;

  std::string server_address_;
  std::uint16_t server_port_;
  NodeRole peer_role_;
  // -1:disconnected, 0:connecting, 1:connected
  std::atomic<int> connection_status_;
  // The Configuration file
  Configuration *config_;
};
}  // namespace core
}  // namespace ps
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PS_CORE_COMMUNICATOR_TCP_CLIENT_H_
