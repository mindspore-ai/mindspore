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
#include <event2/bufferevent.h>
#include <event2/event.h>
#include <event2/listener.h>
#include <event2/buffer_compat.h>
#include <event2/util.h>
#include <sys/socket.h>
#include <csignal>
#include <utility>

#include "ps/core/comm_util.h"

namespace mindspore {
namespace ps {
namespace core {

void TcpConnection::InitConnection() {
  tcp_message_handler_.SetCallback([&](const CommMessage &message) {
    OnServerReceiveMessage on_server_receive = server_->GetServerReceive();
    if (on_server_receive) {
      on_server_receive(*server_, *this, message);
    }
  });
}

void TcpConnection::OnReadHandler(const void *buffer, size_t num) { tcp_message_handler_.ReceiveMessage(buffer, num); }

void TcpConnection::SendMessage(const void *buffer, size_t num) const {
  if (bufferevent_write(buffer_event_, buffer, num) == -1) {
    MS_LOG(ERROR) << "Write message to buffer event failed!";
  }
}

TcpServer *TcpConnection::GetServer() const { return const_cast<TcpServer *>(server_); }

const evutil_socket_t &TcpConnection::GetFd() const { return fd_; }

void TcpConnection::SendMessage(const CommMessage &message) const {
  MS_EXCEPTION_IF_NULL(buffer_event_);
  uint32_t buf_size = message.ByteSizeLong();
  std::vector<unsigned char> serialized(buf_size);
  message.SerializeToArray(serialized.data(), static_cast<int>(buf_size));
  if (evbuffer_add(bufferevent_get_output(const_cast<struct bufferevent *>(buffer_event_)), &buf_size,
                   sizeof(buf_size)) == -1) {
    MS_LOG(EXCEPTION) << "Event buffer add header failed!";
  }
  if (evbuffer_add(bufferevent_get_output(const_cast<struct bufferevent *>(buffer_event_)), serialized.data(),
                   buf_size) == -1) {
    MS_LOG(EXCEPTION) << "Event buffer add protobuf data failed!";
  }
}

TcpServer::TcpServer(const std::string &address, std::uint16_t port)
    : base_(nullptr),
      signal_event_(nullptr),
      listener_(nullptr),
      server_address_(std::move(address)),
      server_port_(port) {}

TcpServer::~TcpServer() { Stop(); }

void TcpServer::SetServerCallback(const OnConnected &client_conn, const OnDisconnected &client_disconn,
                                  const OnAccepted &client_accept) {
  this->client_connection_ = client_conn;
  this->client_disconnection_ = client_disconn;
  this->client_accept_ = client_accept;
}

void TcpServer::Init() {
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
  std::unique_lock<std::recursive_mutex> lock(connection_mutex_);
  MS_LOG(INFO) << "Start tcp server!";
  MS_EXCEPTION_IF_NULL(base_);
  int ret = event_base_dispatch(base_);
  MSLOG_IF(INFO, ret == 0, NoExceptionType) << "Event base dispatch success!";
  MSLOG_IF(mindspore::ERROR, ret == 1, NoExceptionType)
    << "Event base dispatch failed with no events pending or active!";
  MSLOG_IF(mindspore::ERROR, ret == -1, NoExceptionType) << "Event base dispatch failed with error occurred!";
  MSLOG_IF(mindspore::EXCEPTION, ret < -1, AbortedError) << "Event base dispatch with unexpect error code!";
}

void TcpServer::StartWithNoBlock() {
  std::unique_lock<std::recursive_mutex> lock(connection_mutex_);
  MS_LOG(INFO) << "Start tcp server with no block!";
  MS_EXCEPTION_IF_NULL(base_);
  int ret = event_base_loop(base_, EVLOOP_NONBLOCK);
  MSLOG_IF(INFO, ret == 0, NoExceptionType) << "Event base loop success!";
  MSLOG_IF(mindspore::ERROR, ret == 1, NoExceptionType) << "Event base loop failed with no events pending or active!";
  MSLOG_IF(mindspore::ERROR, ret == -1, NoExceptionType) << "Event base loop failed with error occurred!";
  MSLOG_IF(mindspore::EXCEPTION, ret < -1, AbortedError) << "Event base loop with unexpect error code!";
}

void TcpServer::Stop() {
  MS_LOG(INFO) << "Stop tcp server!";
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

void TcpServer::SendToAllClients(const char *data, size_t len) {
  MS_EXCEPTION_IF_NULL(data);
  std::unique_lock<std::recursive_mutex> lock(connection_mutex_);
  for (auto it = connections_.begin(); it != connections_.end(); ++it) {
    it->second->SendMessage(data, len);
  }
}

void TcpServer::AddConnection(const evutil_socket_t &fd, const TcpConnection *connection) {
  MS_EXCEPTION_IF_NULL(connection);
  std::unique_lock<std::recursive_mutex> lock(connection_mutex_);
  connections_.insert(std::make_pair(fd, connection));
}

void TcpServer::RemoveConnection(const evutil_socket_t &fd) {
  std::unique_lock<std::recursive_mutex> lock(connection_mutex_);
  TcpConnection *connection = const_cast<TcpConnection *>(connections_.find(fd)->second);
  delete connection;
  connections_.erase(fd);
}

void TcpServer::ListenerCallback(struct evconnlistener *, evutil_socket_t fd, struct sockaddr *sockaddr, int,
                                 void *data) {
  auto server = reinterpret_cast<class TcpServer *>(data);
  auto base = reinterpret_cast<struct event_base *>(server->base_);
  MS_EXCEPTION_IF_NULL(server);
  MS_EXCEPTION_IF_NULL(base);
  MS_EXCEPTION_IF_NULL(sockaddr);

  struct bufferevent *bev = bufferevent_socket_new(base, fd, BEV_OPT_CLOSE_ON_FREE);
  if (!bev) {
    MS_LOG(ERROR) << "Error constructing buffer event!";
    event_base_loopbreak(base);
    return;
  }

  TcpConnection *conn = server->onCreateConnection(bev, fd);
  MS_EXCEPTION_IF_NULL(conn);

  conn->InitConnection();
  server->AddConnection(fd, conn);
  bufferevent_setcb(bev, TcpServer::ReadCallback, nullptr, TcpServer::EventCallback, reinterpret_cast<void *>(conn));
  if (bufferevent_enable(bev, EV_READ | EV_WRITE) == -1) {
    MS_LOG(EXCEPTION) << "Buffer event enable read and write failed!";
  }
}

TcpConnection *TcpServer::onCreateConnection(struct bufferevent *bev, const evutil_socket_t &fd) {
  TcpConnection *conn = nullptr;
  if (client_accept_) {
    conn = const_cast<TcpConnection *>(client_accept_(*this));
  } else {
    conn = new TcpConnection(bev, fd, this);
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
  char read_buffer[4096];
  while (EVBUFFER_LENGTH(buf) > 0) {
    int read = evbuffer_remove(buf, &read_buffer, sizeof(read_buffer));
    if (read == -1) {
      MS_LOG(EXCEPTION) << "Can not drain data from the event buffer!";
    }
    conn->OnReadHandler(read_buffer, static_cast<size_t>(read));
  }
}

void TcpServer::EventCallback(struct bufferevent *bev, std::int16_t events, void *data) {
  MS_EXCEPTION_IF_NULL(bev);
  MS_EXCEPTION_IF_NULL(data);
  struct evbuffer *output = bufferevent_get_output(bev);
  size_t remain = evbuffer_get_length(output);
  auto conn = reinterpret_cast<TcpConnection *>(data);
  TcpServer *srv = conn->GetServer();

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
    MS_LOG(ERROR) << "Event buffer remain data: " << remain;
    // Free connection structures
    srv->RemoveConnection(conn->GetFd());
    bufferevent_free(bev);

    // Notify about disconnection
    if (srv->client_disconnection_) {
      srv->client_disconnection_(*srv, *conn);
    }
  } else {
    MS_LOG(ERROR) << "Unhandled event!";
  }
}

void TcpServer::SendMessage(const TcpConnection &conn, const CommMessage &message) { conn.SendMessage(message); }

void TcpServer::SendMessage(const CommMessage &message) {
  std::unique_lock<std::recursive_mutex> lock(connection_mutex_);

  for (auto it = connections_.begin(); it != connections_.end(); ++it) {
    SendMessage(*it->second, message);
  }
}

uint16_t TcpServer::BoundPort() const { return server_port_; }

void TcpServer::SetMessageCallback(const OnServerReceiveMessage &cb) { message_callback_ = cb; }

}  // namespace core
}  // namespace ps
}  // namespace mindspore
