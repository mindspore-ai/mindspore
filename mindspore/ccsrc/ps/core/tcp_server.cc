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

#include "ps/core/tcp_server.h"

#include <arpa/inet.h>
#include <event2/buffer.h>
#include <event2/buffer_compat.h>
#include <event2/bufferevent.h>
#include <event2/event.h>
#include <event2/listener.h>
#include <event2/util.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <csignal>
#include <utility>

namespace mindspore {
namespace ps {
namespace core {
void TcpConnection::InitConnection(const messageReceive &callback) { tcp_message_handler_.SetCallback(callback); }

void TcpConnection::OnReadHandler(const void *buffer, size_t num) { tcp_message_handler_.ReceiveMessage(buffer, num); }

void TcpConnection::SendMessage(const void *buffer, size_t num) const {
  if (bufferevent_write(buffer_event_, buffer, num) == -1) {
    MS_LOG(ERROR) << "Write message to buffer event failed!";
  }
}

const TcpServer *TcpConnection::GetServer() const { return server_; }

const evutil_socket_t &TcpConnection::GetFd() const { return fd_; }

void TcpConnection::set_callback(const Callback &callback) { callback_ = callback; }

bool TcpConnection::SendMessage(std::shared_ptr<CommMessage> message) const {
  MS_EXCEPTION_IF_NULL(buffer_event_);
  MS_EXCEPTION_IF_NULL(message);
  bufferevent_lock(buffer_event_);
  bool res = true;
  size_t buf_size = message->ByteSizeLong();
  if (bufferevent_write(buffer_event_, &buf_size, sizeof(buf_size)) == -1) {
    MS_LOG(ERROR) << "Event buffer add header failed!";
    res = false;
  }
  if (bufferevent_write(buffer_event_, message->SerializeAsString().data(), buf_size) == -1) {
    MS_LOG(ERROR) << "Event buffer add protobuf data failed!";
    res = false;
  }
  bufferevent_unlock(buffer_event_);
  return res;
}

bool TcpConnection::SendMessage(std::shared_ptr<MessageMeta> meta, const Protos &protos, const void *data,
                                size_t size) const {
  MS_EXCEPTION_IF_NULL(buffer_event_);
  MS_EXCEPTION_IF_NULL(meta);
  MS_EXCEPTION_IF_NULL(data);
  bufferevent_lock(buffer_event_);
  bool res = true;
  MessageHeader header;
  header.message_proto_ = protos;
  header.message_meta_length_ = SizeToUint(meta->ByteSizeLong());
  header.message_length_ = size + header.message_meta_length_;

  if (bufferevent_write(buffer_event_, &header, sizeof(header)) == -1) {
    MS_LOG(ERROR) << "Event buffer add header failed!";
    res = false;
  }
  if (bufferevent_write(buffer_event_, meta->SerializeAsString().data(), meta->ByteSizeLong()) == -1) {
    MS_LOG(ERROR) << "Event buffer add protobuf data failed!";
    res = false;
  }
  if (bufferevent_write(buffer_event_, data, size) == -1) {
    MS_LOG(ERROR) << "Event buffer add protobuf data failed!";
    res = false;
  }
  int result = bufferevent_flush(buffer_event_, EV_READ | EV_WRITE, BEV_FLUSH);
  if (result < 0) {
    MS_LOG(EXCEPTION) << "Bufferevent flush failed!";
  }
  bufferevent_unlock(buffer_event_);
  MS_LOG(DEBUG) << "SendMessage the request id is:" << meta->request_id() << " the current time is:"
                << std::chrono::time_point_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now())
                     .time_since_epoch()
                     .count();
  return res;
}

TcpServer::TcpServer(const std::string &address, std::uint16_t port)
    : base_(nullptr),
      signal_event_(nullptr),
      listener_(nullptr),
      server_address_(std::move(address)),
      server_port_(port),
      is_stop_(true) {}

TcpServer::~TcpServer() {
  if (signal_event_ != nullptr) {
    event_free(signal_event_);
    signal_event_ = nullptr;
  }

  if (listener_ != nullptr) {
    evconnlistener_free(listener_);
    listener_ = nullptr;
  }

  if (base_ != nullptr) {
    event_base_free(base_);
    base_ = nullptr;
  }
}

void TcpServer::SetServerCallback(const OnConnected &client_conn, const OnDisconnected &client_disconn,
                                  const OnAccepted &client_accept) {
  this->client_connection_ = client_conn;
  this->client_disconnection_ = client_disconn;
  this->client_accept_ = client_accept;
}

void TcpServer::set_timer_once_callback(const OnTimerOnce &timer) { on_timer_once_callback_ = timer; }

void TcpServer::set_timer_callback(const OnTimer &timer) { on_timer_callback_ = timer; }

void TcpServer::Init() {
  int result = evthread_use_pthreads();
  if (result != 0) {
    MS_LOG(EXCEPTION) << "Use event pthread failed!";
  }

  event_enable_debug_logging(EVENT_DBG_ALL);
  event_set_log_callback(CommUtil::LogCallback);
  is_stop_ = false;
  base_ = event_base_new();
  MS_EXCEPTION_IF_NULL(base_);
  if (!CommUtil::CheckIp(server_address_)) {
    MS_LOG(EXCEPTION) << "The tcp server ip:" << server_address_ << " is illegal!";
  }

  struct sockaddr_in sin {};
  if (memset_s(&sin, sizeof(sin), 0, sizeof(sin)) != EOK) {
    MS_LOG(EXCEPTION) << "Initialize sockaddr_in failed!";
  }
  sin.sin_family = AF_INET;
  sin.sin_port = htons(server_port_);
  sin.sin_addr.s_addr = inet_addr(server_address_.c_str());

  listener_ = evconnlistener_new_bind(base_, ListenerCallback, reinterpret_cast<void *>(this),
                                      LEV_OPT_REUSEABLE | LEV_OPT_CLOSE_ON_FREE, -1,
                                      reinterpret_cast<struct sockaddr *>(&sin), sizeof(sin));

  MS_EXCEPTION_IF_NULL(listener_);

  if (server_port_ == 0) {
    struct sockaddr_in sin_bound {};
    if (memset_s(&sin, sizeof(sin_bound), 0, sizeof(sin_bound)) != EOK) {
      MS_LOG(EXCEPTION) << "Initialize sockaddr_in failed!";
    }
    socklen_t addr_len = sizeof(struct sockaddr_in);
    if (getsockname(evconnlistener_get_fd(listener_), (struct sockaddr *)&sin_bound, &addr_len) != 0) {
      MS_LOG(EXCEPTION) << "Get sock name failed!";
    }
    server_port_ = htons(sin_bound.sin_port);
  }

  signal_event_ = evsignal_new(base_, SIGINT, SignalCallback, reinterpret_cast<void *>(this));
  MS_EXCEPTION_IF_NULL(signal_event_);
  if (event_add(signal_event_, nullptr) < 0) {
    MS_LOG(EXCEPTION) << "Cannot create signal event.";
  }
}

void TcpServer::Start() {
  MS_LOG(INFO) << "Start tcp server!";
  MS_EXCEPTION_IF_NULL(base_);
  int ret = event_base_dispatch(base_);
  MSLOG_IF(INFO, ret == 0, NoExceptionType) << "Event base dispatch success!";
  MSLOG_IF(mindspore::ERROR, ret == 1, NoExceptionType)
    << "Event base dispatch failed with no events pending or active!";
  MSLOG_IF(mindspore::ERROR, ret == -1, NoExceptionType) << "Event base dispatch failed with error occurred!";
  MSLOG_IF(mindspore::EXCEPTION, ret < -1, AbortedError) << "Event base dispatch with unexpected error code!";
}

void TcpServer::StartWithNoBlock() {
  std::lock_guard<std::mutex> lock(connection_mutex_);
  MS_LOG(INFO) << "Start tcp server with no block!";
  MS_EXCEPTION_IF_NULL(base_);
  int ret = event_base_loop(base_, EVLOOP_NONBLOCK);
  MSLOG_IF(INFO, ret == 0, NoExceptionType) << "Event base loop success!";
  MSLOG_IF(mindspore::ERROR, ret == 1, NoExceptionType) << "Event base loop failed with no events pending or active!";
  MSLOG_IF(mindspore::ERROR, ret == -1, NoExceptionType) << "Event base loop failed with error occurred!";
  MSLOG_IF(mindspore::EXCEPTION, ret < -1, AbortedError) << "Event base loop with unexpected error code!";
}

void TcpServer::StartTimerOnlyOnce(const uint32_t &time) {
  MS_EXCEPTION_IF_NULL(base_);
  if (time == 0) {
    MS_LOG(EXCEPTION) << "The time should not be 0!";
  }
  struct event *ev = nullptr;
  struct timeval timeout {};
  timeout.tv_sec = time;
  timeout.tv_usec = 0;
  ev = evtimer_new(base_, TimerOnceCallback, this);
  MS_EXCEPTION_IF_NULL(ev);
  evtimer_add(ev, &timeout);
}

void TcpServer::StartTimer(const uint32_t &time) {
  MS_EXCEPTION_IF_NULL(base_);
  struct event *ev = nullptr;
  if (time == 0) {
    MS_LOG(EXCEPTION) << "The time should not be 0!";
  }
  struct timeval timeout {};
  timeout.tv_sec = time;
  timeout.tv_usec = 0;
  ev = event_new(base_, -1, EV_PERSIST, TimerCallback, this);
  MS_EXCEPTION_IF_NULL(ev);
  evtimer_add(ev, &timeout);
}

void TcpServer::Stop() {
  std::lock_guard<std::mutex> lock(connection_mutex_);
  MS_LOG(INFO) << "Stop tcp server!";
  if (event_base_got_break(base_)) {
    MS_LOG(DEBUG) << "The event base has stopped!";
    is_stop_ = true;
    return;
  }
  if (!is_stop_.load()) {
    is_stop_ = true;
    int ret = event_base_loopbreak(base_);
    if (ret != 0) {
      MS_LOG(ERROR) << "Event base loop break failed!";
    }
  }
}

void TcpServer::SendToAllClients(const char *data, size_t len) {
  MS_EXCEPTION_IF_NULL(data);
  std::lock_guard<std::mutex> lock(connection_mutex_);
  for (auto it = connections_.begin(); it != connections_.end(); ++it) {
    it->second->SendMessage(data, len);
  }
}

void TcpServer::AddConnection(const evutil_socket_t &fd, std::shared_ptr<TcpConnection> connection) {
  MS_EXCEPTION_IF_NULL(connection);
  std::lock_guard<std::mutex> lock(connection_mutex_);
  connections_.insert(std::make_pair(fd, connection));
}

void TcpServer::RemoveConnection(const evutil_socket_t &fd) {
  std::lock_guard<std::mutex> lock(connection_mutex_);
  connections_.erase(fd);
}

std::shared_ptr<TcpConnection> TcpServer::GetConnectionByFd(const evutil_socket_t &fd) { return connections_[fd]; }

void TcpServer::ListenerCallback(struct evconnlistener *, evutil_socket_t fd, struct sockaddr *sockaddr, int,
                                 void *data) {
  auto server = reinterpret_cast<class TcpServer *>(data);
  auto base = reinterpret_cast<struct event_base *>(server->base_);
  MS_EXCEPTION_IF_NULL(server);
  MS_EXCEPTION_IF_NULL(base);
  MS_EXCEPTION_IF_NULL(sockaddr);

  struct bufferevent *bev = bufferevent_socket_new(base, fd, BEV_OPT_CLOSE_ON_FREE | BEV_OPT_THREADSAFE);
  if (bev == nullptr) {
    MS_LOG(ERROR) << "Error constructing buffer event!";
    int ret = event_base_loopbreak(base);
    if (ret != 0) {
      MS_LOG(EXCEPTION) << "event base loop break failed!";
    }
    return;
  }

  std::shared_ptr<TcpConnection> conn = server->onCreateConnection(bev, fd);
  MS_EXCEPTION_IF_NULL(conn);
  SetTcpNoDelay(fd);
  server->AddConnection(fd, conn);
  conn->InitConnection([=](std::shared_ptr<MessageMeta> meta, const Protos &protos, const void *data, size_t size) {
    OnServerReceiveMessage on_server_receive = server->GetServerReceive();
    if (on_server_receive) {
      on_server_receive(conn, meta, protos, data, size);
    }
  });
  bufferevent_setcb(bev, TcpServer::ReadCallback, nullptr, TcpServer::EventCallback,
                    reinterpret_cast<void *>(conn.get()));
  if (bufferevent_enable(bev, EV_READ | EV_WRITE) == -1) {
    MS_LOG(EXCEPTION) << "Buffer event enable read and write failed!";
  }
}

std::shared_ptr<TcpConnection> TcpServer::onCreateConnection(struct bufferevent *bev, const evutil_socket_t &fd) {
  MS_EXCEPTION_IF_NULL(bev);
  std::shared_ptr<TcpConnection> conn = nullptr;
  if (client_accept_) {
    conn = (client_accept_(*this));
  } else {
    conn = std::make_shared<TcpConnection>(bev, fd, this);
  }

  return conn;
}

OnServerReceiveMessage TcpServer::GetServerReceive() const { return message_callback_; }

void TcpServer::SignalCallback(evutil_socket_t, std::int16_t, void *data) {
  auto server = reinterpret_cast<class TcpServer *>(data);
  MS_EXCEPTION_IF_NULL(server);
  struct event_base *base = server->base_;
  struct timeval delay = {0, 0};
  MS_LOG(ERROR) << "Caught an interrupt signal; exiting cleanly in 0 seconds.";
  if (event_base_loopexit(base, &delay) == -1) {
    MS_LOG(EXCEPTION) << "Event base loop exit failed.";
  }
}

void TcpServer::ReadCallback(struct bufferevent *bev, void *connection) {
  MS_EXCEPTION_IF_NULL(bev);
  MS_EXCEPTION_IF_NULL(connection);

  auto conn = static_cast<class TcpConnection *>(connection);
  struct evbuffer *buf = bufferevent_get_input(bev);
  char read_buffer[kMessageChunkLength];
  while (EVBUFFER_LENGTH(buf) > 0) {
    int read = evbuffer_remove(buf, &read_buffer, sizeof(read_buffer));
    if (read == -1) {
      MS_LOG(EXCEPTION) << "Can not drain data from the event buffer!";
    }
    conn->OnReadHandler(read_buffer, IntToSize(read));
    MS_LOG(DEBUG) << "the current time is:"
                  << std::chrono::time_point_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now())
                       .time_since_epoch()
                       .count()
                  << " the read size is:" << read;
  }
}

void TcpServer::EventCallback(struct bufferevent *bev, std::int16_t events, void *data) {
  MS_EXCEPTION_IF_NULL(bev);
  MS_EXCEPTION_IF_NULL(data);
  struct evbuffer *output = bufferevent_get_output(bev);
  size_t remain = evbuffer_get_length(output);
  auto conn = static_cast<class TcpConnection *>(data);
  auto srv = const_cast<TcpServer *>(conn->GetServer());

  if (events & BEV_EVENT_EOF) {
    MS_LOG(INFO) << "Event buffer end of file!";
    // Notify about disconnection
    if (srv->client_disconnection_) {
      srv->client_disconnection_(*srv, *conn);
    }
    // Free connection structures
    srv->RemoveConnection(conn->GetFd());
    bufferevent_free(bev);
  } else if (events & BEV_EVENT_ERROR) {
    MS_LOG(WARNING) << "Event buffer remain data: " << remain;
    // Free connection structures
    srv->RemoveConnection(conn->GetFd());
    bufferevent_free(bev);

    // Notify about disconnection
    if (srv->client_disconnection_) {
      srv->client_disconnection_(*srv, *conn);
    }
  } else {
    MS_LOG(WARNING) << "Unhandled event!";
  }
}

void TcpServer::TimerCallback(evutil_socket_t, int16_t, void *arg) {
  MS_EXCEPTION_IF_NULL(arg);
  auto tcp_server = reinterpret_cast<TcpServer *>(arg);
  if (tcp_server->on_timer_callback_) {
    tcp_server->on_timer_callback_();
  }
}

void TcpServer::TimerOnceCallback(evutil_socket_t, int16_t, void *arg) {
  MS_EXCEPTION_IF_NULL(arg);
  auto tcp_server = reinterpret_cast<TcpServer *>(arg);
  if (tcp_server->on_timer_once_callback_) {
    tcp_server->on_timer_once_callback_(*tcp_server);
  }
}

void TcpServer::SetTcpNoDelay(const evutil_socket_t &fd) {
  const int one = 1;
  int ret = setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(int));
  if (ret < 0) {
    MS_LOG(EXCEPTION) << "Set socket no delay failed!";
  }
}

bool TcpServer::SendMessage(std::shared_ptr<TcpConnection> conn, std::shared_ptr<CommMessage> message) {
  MS_EXCEPTION_IF_NULL(conn);
  MS_EXCEPTION_IF_NULL(message);
  return conn->SendMessage(message);
}

bool TcpServer::SendMessage(std::shared_ptr<TcpConnection> conn, std::shared_ptr<MessageMeta> meta,
                            const Protos &protos, const void *data, size_t size) {
  MS_EXCEPTION_IF_NULL(conn);
  MS_EXCEPTION_IF_NULL(meta);
  MS_EXCEPTION_IF_NULL(data);
  return conn->SendMessage(meta, protos, data, size);
}

void TcpServer::SendMessage(std::shared_ptr<CommMessage> message) {
  MS_EXCEPTION_IF_NULL(message);
  std::lock_guard<std::mutex> lock(connection_mutex_);

  for (auto it = connections_.begin(); it != connections_.end(); ++it) {
    SendMessage(it->second, message);
  }
}

uint16_t TcpServer::BoundPort() const { return server_port_; }

std::string TcpServer::BoundIp() const { return server_address_; }

int TcpServer::ConnectionNum() const { return connections_.size(); }

const std::map<evutil_socket_t, std::shared_ptr<TcpConnection>> &TcpServer::Connections() const { return connections_; }

void TcpServer::SetMessageCallback(const OnServerReceiveMessage &cb) { message_callback_ = cb; }
}  // namespace core
}  // namespace ps
}  // namespace mindspore
