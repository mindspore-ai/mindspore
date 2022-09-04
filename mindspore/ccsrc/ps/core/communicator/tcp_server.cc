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

#include "ps/core/communicator/tcp_server.h"

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
TcpConnection::~TcpConnection() {
  MS_LOG(WARNING) << "TcpConnection is destructed! fd is " << fd_;
  bufferevent_free(buffer_event_);
}
void TcpConnection::InitConnection(const messageReceive &callback) { tcp_message_handler_.SetCallback(callback); }

void TcpConnection::OnReadHandler(const void *buffer, size_t num) {
  MS_EXCEPTION_IF_NULL(buffer);
  tcp_message_handler_.ReceiveMessage(buffer, num);
}

void TcpConnection::SendMessage(const void *buffer, size_t num) const {
  MS_EXCEPTION_IF_NULL(buffer);
  MS_EXCEPTION_IF_NULL(buffer_event_);
  bufferevent_lock(buffer_event_);
  if (bufferevent_write(buffer_event_, buffer, num) == -1) {
    MS_LOG(ERROR) << "Write message to buffer event failed!";
  }
  bufferevent_unlock(buffer_event_);
}

const TcpServer *TcpConnection::GetServer() const { return server_; }

const evutil_socket_t &TcpConnection::GetFd() const { return fd_; }

void TcpConnection::set_callback(const Callback &callback) { callback_ = callback; }

bool TcpConnection::SendMessage(const std::shared_ptr<CommMessage> &message) const {
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

bool TcpConnection::SendMessage(const std::shared_ptr<MessageMeta> &meta, const Protos &protos, const void *data,
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
    bufferevent_unlock(buffer_event_);
    MS_LOG(EXCEPTION) << "Bufferevent flush failed!";
  }
  bufferevent_unlock(buffer_event_);
  return res;
}

TcpServer::TcpServer(const std::string &address, std::uint16_t port, Configuration *const config)
    : base_(nullptr),
      signal_event_(nullptr),
      listener_(nullptr),
      server_address_(std::move(address)),
      server_port_(port),
      is_stop_(true),
      config_(config),
      max_connection_(0) {}

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

void TcpServer::Init() {
  if (PSContext::instance()->enable_ssl()) {
    MS_LOG(INFO) << "Init ssl.";
    SSLWrapper::GetInstance().InitSSL();
  }
  int result = evthread_use_pthreads();
  if (result != 0) {
    MS_LOG(EXCEPTION) << "Use event pthread failed!";
  }

  is_stop_ = false;
  base_ = event_base_new();
  MS_EXCEPTION_IF_NULL(base_);
  if (!CommUtil::CheckIp(server_address_)) {
    MS_LOG(EXCEPTION) << "The tcp server ip:" << server_address_ << " is illegal!";
  }
  MS_EXCEPTION_IF_NULL(config_);
  max_connection_ = kConnectionNumDefault;
  if (config_->Exists(kConnectionNum)) {
    max_connection_ = config_->GetInt(kConnectionNum, 0);
  }
  MS_LOG(INFO) << "The max connection is:" << max_connection_;

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
  if (listener_ == nullptr) {
    MS_LOG(EXCEPTION) << "bind ip & port failed. please check.";
  }

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
  MSLOG_IF(MsLogLevel::kInfo, ret == 0, NoExceptionType) << "Event base dispatch success!";
  MSLOG_IF(MsLogLevel::kError, ret == 1, NoExceptionType)
    << "Event base dispatch failed with no events pending or active!";
  MSLOG_IF(MsLogLevel::kError, ret == -1, NoExceptionType) << "Event base dispatch failed with error occurred!";
  MSLOG_IF(MsLogLevel::kException, ret < -1, AbortedError) << "Event base dispatch with unexpected error code!";
}

void TcpServer::Stop() {
  MS_ERROR_IF_NULL_WO_RET_VAL(base_);
  std::lock_guard<std::mutex> lock(connection_mutex_);
  MS_LOG(INFO) << "Stop tcp server!";
  if (event_base_got_break(base_)) {
    MS_LOG(DEBUG) << "The event base has already been stopped!";
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
  MS_LOG(INFO) << "Remove connection fd: " << fd;
  connections_.erase(fd);
}

std::shared_ptr<TcpConnection> &TcpServer::GetConnectionByFd(const evutil_socket_t &fd) { return connections_[fd]; }

void TcpServer::ListenerCallback(struct evconnlistener *, evutil_socket_t fd, struct sockaddr *sockaddr, int,
                                 void *const data) {
  try {
    ListenerCallbackInner(fd, sockaddr, data);
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "Catch exception: " << e.what();
  }
}

void TcpServer::ListenerCallbackInner(evutil_socket_t fd, struct sockaddr *sockaddr, void *const data) {
  auto server = reinterpret_cast<class TcpServer *>(data);
  MS_EXCEPTION_IF_NULL(server);
  auto base = reinterpret_cast<struct event_base *>(server->base_);
  MS_EXCEPTION_IF_NULL(base);
  MS_EXCEPTION_IF_NULL(sockaddr);

  if (server->ConnectionNum() >= server->max_connection_) {
    MS_LOG(WARNING) << "The current connection num:" << server->ConnectionNum() << " is greater or equal to "
                    << server->max_connection_;
    return;
  }

  struct bufferevent *bev = nullptr;

  if (!PSContext::instance()->enable_ssl()) {
    MS_LOG(INFO) << "SSL is disable.";
    bev = bufferevent_socket_new(base, fd, BEV_OPT_CLOSE_ON_FREE | BEV_OPT_THREADSAFE);
  } else {
    MS_LOG(INFO) << "Enable ssl support.";
    SSL *ssl = SSL_new(SSLWrapper::GetInstance().GetSSLCtx());
    MS_EXCEPTION_IF_NULL(ssl);
    bev = bufferevent_openssl_socket_new(base, fd, ssl, BUFFEREVENT_SSL_ACCEPTING,
                                         BEV_OPT_CLOSE_ON_FREE | BEV_OPT_THREADSAFE);
  }
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
  conn->InitConnection(
    [=](const std::shared_ptr<MessageMeta> &meta, const Protos &protos, const void *data, size_t size) {
      OnServerReceiveMessage on_server_receive = server->GetServerReceive();
      if (on_server_receive) {
        on_server_receive(conn, meta, protos, data, size);
      }
    });

  bufferevent_setcb(bev, TcpServer::ReadCallback, nullptr, TcpServer::EventCallback,
                    reinterpret_cast<void *>(conn.get()));
  MS_LOG(INFO) << "A client is connected, fd is " << fd;
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

void TcpServer::SignalCallback(evutil_socket_t, std::int16_t, void *const data) {
  try {
    SignalCallbackInner(data);
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "Catch exception: " << e.what();
  }
}

void TcpServer::SignalCallbackInner(void *const data) {
  MS_EXCEPTION_IF_NULL(data);
  auto server = reinterpret_cast<class TcpServer *>(data);
  struct event_base *base = server->base_;
  MS_EXCEPTION_IF_NULL(base);
  struct timeval delay = {0, 0};
  MS_LOG(ERROR) << "Caught an interrupt signal; exiting cleanly in 0 seconds.";
  if (event_base_loopexit(base, &delay) == -1) {
    MS_LOG(ERROR) << "Event base loop exit failed.";
  }
}

void TcpServer::ReadCallback(struct bufferevent *bev, void *const connection) {
  try {
    ReadCallbackInner(bev, connection);
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "Catch exception: " << e.what();
  }
}

void TcpServer::ReadCallbackInner(struct bufferevent *bev, void *const connection) {
  MS_EXCEPTION_IF_NULL(bev);
  MS_EXCEPTION_IF_NULL(connection);

  auto conn = static_cast<class TcpConnection *>(connection);
  struct evbuffer *buf = bufferevent_get_input(bev);
  MS_EXCEPTION_IF_NULL(buf);
  char read_buffer[kMessageChunkLength];
  while (EVBUFFER_LENGTH(buf) > 0) {
    int read = evbuffer_remove(buf, &read_buffer, sizeof(read_buffer));
    if (read == -1) {
      MS_LOG(EXCEPTION) << "Can not drain data from the event buffer!";
    }
    conn->OnReadHandler(read_buffer, IntToSize(read));
  }
}

void TcpServer::EventCallback(struct bufferevent *bev, std::int16_t events, void *const data) {
  try {
    EventCallbackInner(bev, events, data);
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "Catch exception: " << e.what();
  }
}

void TcpServer::EventCallbackInner(struct bufferevent *bev, std::int16_t events, void *const data) {
  MS_EXCEPTION_IF_NULL(bev);
  MS_EXCEPTION_IF_NULL(data);
  struct evbuffer *output = bufferevent_get_output(bev);
  MS_EXCEPTION_IF_NULL(output);
  auto conn = static_cast<class TcpConnection *>(data);
  auto srv = const_cast<TcpServer *>(conn->GetServer());
  MS_EXCEPTION_IF_NULL(srv);

  if (events & BEV_EVENT_EOF) {
    MS_LOG(INFO) << "BEV_EVENT_EOF event is trigger!";
    // Notify about disconnection
    if (srv->client_disconnection_) {
      srv->client_disconnection_(*srv, *conn);
    }
    // Free connection structures
    srv->RemoveConnection(conn->GetFd());
  } else if (events & BEV_EVENT_ERROR) {
    MS_LOG(WARNING) << "BEV_EVENT_ERROR event is trigger!";
    if (PSContext::instance()->enable_ssl()) {
      uint64_t err = bufferevent_get_openssl_error(bev);
      MS_LOG(WARNING) << "The error number is:" << err;

      MS_LOG(WARNING) << "Error message:" << ERR_reason_error_string(err)
                      << ", the error lib:" << ERR_lib_error_string(err)
                      << ", the error func:" << ERR_func_error_string(err);
    }
    // Notify about disconnection
    if (srv->client_disconnection_) {
      srv->client_disconnection_(*srv, *conn);
    }
    // Free connection structures
    srv->RemoveConnection(conn->GetFd());
  } else {
    MS_LOG(WARNING) << "Unhandled event:" << events;
  }
}

void TcpServer::SetTcpNoDelay(const evutil_socket_t &fd) {
  const int one = 1;
  int ret = setsockopt(fd, static_cast<int>(IPPROTO_TCP), static_cast<int>(TCP_NODELAY), &one, sizeof(int));
  if (ret < 0) {
    MS_LOG(EXCEPTION) << "Set socket no delay failed!";
  }
}

bool TcpServer::SendMessage(const std::shared_ptr<TcpConnection> &conn, const std::shared_ptr<CommMessage> &message) {
  MS_EXCEPTION_IF_NULL(conn);
  MS_EXCEPTION_IF_NULL(message);
  return conn->SendMessage(message);
}

bool TcpServer::SendMessage(const std::shared_ptr<TcpConnection> &conn, const std::shared_ptr<MessageMeta> &meta,
                            const Protos &protos, const void *data, size_t size) {
  MS_EXCEPTION_IF_NULL(conn);
  MS_EXCEPTION_IF_NULL(meta);
  MS_EXCEPTION_IF_NULL(data);
  return conn->SendMessage(meta, protos, data, size);
}

void TcpServer::SendMessage(const std::shared_ptr<CommMessage> &message) {
  MS_EXCEPTION_IF_NULL(message);
  std::lock_guard<std::mutex> lock(connection_mutex_);

  for (auto it = connections_.begin(); it != connections_.end(); ++it) {
    SendMessage(it->second, message);
  }
}

uint16_t TcpServer::BoundPort() const { return server_port_; }

std::string TcpServer::BoundIp() const { return server_address_; }

int TcpServer::ConnectionNum() const { return SizeToInt(connections_.size()); }

const std::map<evutil_socket_t, std::shared_ptr<TcpConnection>> &TcpServer::Connections() const { return connections_; }

void TcpServer::SetMessageCallback(const OnServerReceiveMessage &cb) { message_callback_ = cb; }
}  // namespace core
}  // namespace ps
}  // namespace mindspore
