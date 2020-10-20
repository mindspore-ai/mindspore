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

#include "ps/comm/tcp_server.h"

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

#include "ps/comm/comm_util.h"

namespace mindspore {
namespace ps {
namespace comm {

void TcpConnection::InitConnection(const evutil_socket_t &fd, const struct bufferevent *bev, const TcpServer *server) {
  MS_EXCEPTION_IF_NULL(bev);
  MS_EXCEPTION_IF_NULL(server);
  buffer_event_ = const_cast<struct bufferevent *>(bev);
  fd_ = fd;
  server_ = const_cast<TcpServer *>(server);

  tcp_message_handler_.SetCallback([this, server](const void *buf, size_t num) {
    OnServerReceiveMessage message_callback = server->GetServerReceiveMessage();
    if (message_callback) message_callback(*server, *this, buf, num);
  });
}

void TcpConnection::OnReadHandler(const void *buffer, size_t num) { tcp_message_handler_.ReceiveMessage(buffer, num); }

void TcpConnection::SendMessage(const void *buffer, size_t num) const {
  if (bufferevent_write(buffer_event_, buffer, num) == -1) {
    MS_LOG(ERROR) << "Write message to buffer event failed!";
  }
}

TcpServer *TcpConnection::GetServer() const { return server_; }

evutil_socket_t TcpConnection::GetFd() const { return fd_; }

TcpServer::TcpServer(std::string address, std::uint16_t port)
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

void TcpServer::InitServer() {
  base_ = event_base_new();
  MS_EXCEPTION_IF_NULL(base_);
  CommUtil::CheckIp(server_address_);

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

  signal_event_ = evsignal_new(base_, SIGINT, SignalCallback, reinterpret_cast<void *>(this));
  MS_EXCEPTION_IF_NULL(signal_event_);
  if (event_add(signal_event_, nullptr) < 0) {
    MS_LOG(EXCEPTION) << "Cannot create signal event.";
  }
}

void TcpServer::Start() {
  std::unique_lock<std::recursive_mutex> l(connection_mutex_);
  MS_EXCEPTION_IF_NULL(base_);
  int ret = event_base_dispatch(base_);
  if (ret == 0) {
    MS_LOG(INFO) << "Event base dispatch success!";
  } else if (ret == 1) {
    MS_LOG(ERROR) << "Event base dispatch failed with no events pending or active!";
  } else if (ret == -1) {
    MS_LOG(ERROR) << "Event base dispatch failed with error occurred!";
  } else {
    MS_LOG(EXCEPTION) << "Event base dispatch with unexpect error code!";
  }
}

void TcpServer::Stop() {
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
  connections_.erase(fd);
}

void TcpServer::ListenerCallback(struct evconnlistener *, evutil_socket_t fd, struct sockaddr *, int, void *data) {
  auto server = reinterpret_cast<class TcpServer *>(data);
  auto base = reinterpret_cast<struct event_base *>(server->base_);
  MS_EXCEPTION_IF_NULL(server);
  MS_EXCEPTION_IF_NULL(base);

  struct bufferevent *bev = bufferevent_socket_new(base, fd, BEV_OPT_CLOSE_ON_FREE);
  if (!bev) {
    MS_LOG(ERROR) << "Error constructing buffer event!";
    event_base_loopbreak(base);
    return;
  }

  TcpConnection *conn = server->onCreateConnection();
  MS_EXCEPTION_IF_NULL(conn);

  conn->InitConnection(fd, bev, server);
  server->AddConnection(fd, conn);
  bufferevent_setcb(bev, TcpServer::ReadCallback, nullptr, TcpServer::EventCallback, reinterpret_cast<void *>(conn));
  if (bufferevent_enable(bev, EV_READ | EV_WRITE) == -1) {
    MS_LOG(EXCEPTION) << "buffer event enable read and write failed!";
  }
}

TcpConnection *TcpServer::onCreateConnection() {
  TcpConnection *conn = nullptr;
  if (client_accept_)
    conn = const_cast<TcpConnection *>(client_accept_(this));
  else
    conn = new TcpConnection();

  return conn;
}

OnServerReceiveMessage TcpServer::GetServerReceiveMessage() const { return message_callback_; }

void TcpServer::SignalCallback(evutil_socket_t, std::int16_t, void *data) {
  auto server = reinterpret_cast<class TcpServer *>(data);
  MS_EXCEPTION_IF_NULL(server);
  struct event_base *base = server->base_;
  struct timeval delay = {0, 0};
  MS_LOG(ERROR) << "Caught an interrupt signal; exiting cleanly in 0 seconds.";
  if (event_base_loopexit(base, &delay) == -1) {
    MS_LOG(EXCEPTION) << "event base loop exit failed.";
  }
}

void TcpServer::ReadCallback(struct bufferevent *bev, void *connection) {
  MS_EXCEPTION_IF_NULL(bev);
  MS_EXCEPTION_IF_NULL(connection);

  auto conn = static_cast<class TcpConnection *>(connection);
  struct evbuffer *buf = bufferevent_get_input(bev);
  char read_buffer[4096];
  auto read = 0;
  while ((read = EVBUFFER_LENGTH(buf)) > 0) {
    if (evbuffer_remove(buf, &read_buffer, sizeof(read_buffer)) == -1) {
      MS_LOG(EXCEPTION) << "Can not drain data from the event buffer!";
    }
    conn->OnReadHandler(read_buffer, static_cast<size_t>(read));
  }
}

void TcpServer::EventCallback(struct bufferevent *bev, std::int16_t events, void *data) {
  MS_EXCEPTION_IF_NULL(bev);
  MS_EXCEPTION_IF_NULL(data);
  auto conn = reinterpret_cast<TcpConnection *>(data);
  TcpServer *srv = conn->GetServer();

  if (events & BEV_EVENT_EOF) {
    // Notify about disconnection
    if (srv->client_disconnection_) srv->client_disconnection_(srv, conn);
    // Free connection structures
    srv->RemoveConnection(conn->GetFd());
    bufferevent_free(bev);
  } else if (events & BEV_EVENT_ERROR) {
    // Free connection structures
    srv->RemoveConnection(conn->GetFd());
    bufferevent_free(bev);

    // Notify about disconnection
    if (srv->client_disconnection_) srv->client_disconnection_(srv, conn);
  } else {
    MS_LOG(ERROR) << "unhandled event!";
  }
}

void TcpServer::ReceiveMessage(const OnServerReceiveMessage &cb) { message_callback_ = cb; }

void TcpServer::SendMessage(const TcpConnection &conn, const void *data, size_t num) {
  MS_EXCEPTION_IF_NULL(data);
  auto mc = const_cast<TcpConnection &>(conn);
  mc.SendMessage(data, num);
}

void TcpServer::SendMessage(const void *data, size_t num) {
  MS_EXCEPTION_IF_NULL(data);
  std::unique_lock<std::recursive_mutex> lock(connection_mutex_);

  for (auto it = connections_.begin(); it != connections_.end(); ++it) {
    SendMessage(*it->second, data, num);
  }
}
}  // namespace comm
}  // namespace ps
}  // namespace mindspore
