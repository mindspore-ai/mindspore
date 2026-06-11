/**
 * Copyright 2020-2025 Huawei Technologies Co., Ltd
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

#include <arpa/inet.h>
#include <event2/buffer.h>
#include <event2/buffer_compat.h>
#include <event2/bufferevent.h>
#include <event2/bufferevent_ssl.h>
#include <event2/event.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <memory>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <utility>
#include "utils/convert_utils_base.h"
#include "include/cluster/rpc/libevent_tcp_client.h"
#include "cluster/rpc/core/communicator/ssl_client.h"

namespace mindspore {
namespace ps {
namespace core {
namespace {
// Returns the interface index for the given IPv6 address. For link-local, finds the interface that has this address.
// Falls back to first link-local interface if address not found. Returns 0 on failure.
uint32_t GetIPv6ScopeId(const struct in6_addr *addr) {
  struct ifaddrs *if_address = nullptr;
  if (getifaddrs(&if_address) != 0) {
    MS_LOG(DEBUG) << "GetIPv6ScopeId: getifaddrs failed.";
    return 0;
  }

  uint32_t scope_id = 0;
  uint32_t fallback_scope_id = 0;
  std::string fallback_if_name;
  char addr_str[INET6_ADDRSTRLEN];
  inet_ntop(AF_INET6, addr, addr_str, sizeof(addr_str));

  for (struct ifaddrs *ifa = if_address; ifa != nullptr && scope_id == 0; ifa = ifa->ifa_next) {
    if (ifa->ifa_addr == nullptr || ifa->ifa_addr->sa_family != AF_INET6) {
      continue;
    }
    // Safe to cast since we verified sa_family is AF_INET6
    auto *addr_in6 = reinterpret_cast<struct sockaddr_in6 *>(ifa->ifa_addr);
    if (memcmp(&addr_in6->sin6_addr, addr, sizeof(struct in6_addr)) == 0) {
      scope_id = if_nametoindex(ifa->ifa_name);
      MS_LOG(DEBUG) << "GetIPv6ScopeId: found exact match for " << addr_str << ", interface " << ifa->ifa_name
                    << ", scope_id=" << scope_id;
      continue;
    }
    if (IN6_IS_ADDR_LINKLOCAL(&addr_in6->sin6_addr) && fallback_scope_id == 0) {
      fallback_scope_id = if_nametoindex(ifa->ifa_name);
      fallback_if_name = ifa->ifa_name;
    }
  }
  freeifaddrs(if_address);
  if (scope_id == 0 && fallback_scope_id != 0) {
    MS_LOG(DEBUG) << "GetIPv6ScopeId: no exact match for " << addr_str << ", using fallback interface "
                  << fallback_if_name << ", scope_id=" << fallback_scope_id;
  } else if (scope_id == 0) {
    MS_LOG(DEBUG) << "GetIPv6ScopeId: no interface found for " << addr_str << ", scope_id=0";
  }
  return (scope_id != 0) ? scope_id : fallback_scope_id;
}
}  // namespace
event_base *TcpClient::event_base_ = nullptr;
std::mutex TcpClient::event_base_mutex_;
bool TcpClient::is_started_ = false;

TcpClient::TcpClient(const std::string &address, std::uint16_t port, NodeRole peer_role)
    : event_timeout_(nullptr),
      buffer_event_(nullptr),
      server_address_(std::move(address)),
      server_port_(port),
      peer_role_(peer_role),
      connection_status_(-1) {
  message_handler_.SetCallback(
    [this](const std::shared_ptr<MessageMeta> &meta, const Protos &protos, const void *data, size_t size) {
      if (message_callback_) {
        message_callback_(meta, protos, data, size);
      }
    });
}

TcpClient::~TcpClient() {
  if (buffer_event_) {
    bufferevent_free(buffer_event_);
    buffer_event_ = nullptr;
  }
  if (event_timeout_) {
    event_free(event_timeout_);
    event_timeout_ = nullptr;
  }
}

std::string TcpClient::PeerRoleName() const {
  switch (peer_role_) {
    case SERVER:
      return "Server";
    case WORKER:
      return "Worker";
    case SCHEDULER:
      return "Scheduler";
    default:
      return "RoleUndefined";
  }
}

void TcpClient::Init() {
  std::lock_guard<std::mutex> lock(connection_mutex_);
  if (connection_status_ != -1) {
    return;
  }
  connection_status_ = 0;
  if (buffer_event_) {
    bufferevent_free(buffer_event_);
    buffer_event_ = nullptr;
  }

  MS_LOG(INFO) << "server_address_:" << server_address_;

  int result = evthread_use_pthreads();
  if (result != 0) {
    MS_LOG(EXCEPTION) << "Use event pthread failed!";
  }
  if (event_base_ == nullptr) {
    event_base_ = event_base_new();
    MS_EXCEPTION_IF_NULL(event_base_);
  }

  if (!PSContext::instance()->enable_ssl()) {
    MS_LOG(INFO) << "SSL is disable.";
    buffer_event_ = bufferevent_socket_new(event_base_, -1, BEV_OPT_CLOSE_ON_FREE | BEV_OPT_THREADSAFE);
  } else {
    if (!EstablishSSL()) {
      MS_LOG(WARNING) << "Establish SSL failed.";
      return;
    }
  }

  MS_EXCEPTION_IF_NULL(buffer_event_);

  bufferevent_setcb(buffer_event_, ReadCallback, nullptr, EventCallback, this);
  if (bufferevent_enable(buffer_event_, EV_READ | EV_WRITE) == -1) {
    MS_LOG(EXCEPTION) << "Buffer event enable read and write failed!";
  }

  if (server_port_ > 0) {
    struct sockaddr_storage addr_storage {};
    struct sockaddr *sa = reinterpret_cast<struct sockaddr *>(&addr_storage);
    socklen_t sa_len;

    if (memset_s(&addr_storage, sizeof(addr_storage), 0, sizeof(addr_storage)) != EOK) {
      MS_LOG(EXCEPTION) << "Initialize sockaddr_storage failed!";
    }

    // Try IPv4 first
    int parse_result =
      inet_pton(AF_INET, server_address_.c_str(), &(reinterpret_cast<struct sockaddr_in *>(sa)->sin_addr));
    if (parse_result > 0) {
      struct sockaddr_in *sin = reinterpret_cast<struct sockaddr_in *>(sa);
      sin->sin_family = AF_INET;
      sin->sin_port = htons(server_port_);
      sa_len = sizeof(struct sockaddr_in);
    } else {
      // Try IPv6: strip zone id (e.g. fe80::1%eth0) for inet_pton, then set sin6_scope_id from zone or GetIPv6ScopeId.
      std::string addr_str = server_address_;
      std::string zone_str;
      size_t zone_pos = server_address_.find('%');
      if (zone_pos != std::string::npos) {
        addr_str = server_address_.substr(0, zone_pos);
        zone_str = server_address_.substr(zone_pos + 1);
      }
      parse_result = inet_pton(AF_INET6, addr_str.c_str(), &(reinterpret_cast<struct sockaddr_in6 *>(sa)->sin6_addr));
      if (parse_result <= 0) {
        MS_LOG(EXCEPTION) << "Invalid IP address format: " << server_address_;
        return;
      }

      struct sockaddr_in6 *sin6 = reinterpret_cast<struct sockaddr_in6 *>(sa);
      sin6->sin6_family = AF_INET6;
      sin6->sin6_port = htons(server_port_);
      sa_len = sizeof(struct sockaddr_in6);

      // Link-local IPv6 (fe80::/10) requires sin6_scope_id; otherwise connect() returns EINVAL (errno 22).
      if (IN6_IS_ADDR_LINKLOCAL(&sin6->sin6_addr)) {
        if (!zone_str.empty()) {
          sin6->sin6_scope_id = if_nametoindex(zone_str.c_str());
          MS_LOG(DEBUG) << "TcpClient: IPv6 link-local zone " << zone_str << ", sin6_scope_id=" << sin6->sin6_scope_id;
        }
        if (sin6->sin6_scope_id == 0) {
          sin6->sin6_scope_id = GetIPv6ScopeId(&sin6->sin6_addr);
          MS_LOG(DEBUG) << "TcpClient: IPv6 link-local, sin6_scope_id=" << sin6->sin6_scope_id;
        }
        if (sin6->sin6_scope_id == 0) {
          MS_LOG(WARNING) << "IPv6 link-local address but no scope_id set and no link-local interface found.";
        }
      }
    }

    int result_code = bufferevent_socket_connect(buffer_event_, sa, sa_len);
    if (result_code < 0) {
      MS_LOG(WARNING) << "Connect server ip:" << server_address_ << " and port: " << server_port_ << " is failed!";
    }
  }
}

void TcpClient::Stop() {
  MS_EXCEPTION_IF_NULL(event_base_);
  std::lock_guard<std::mutex> lock(connection_mutex_);
  if (event_base_got_break(event_base_)) {
    MS_LOG(DEBUG) << "The event base has already been stopped!";
    return;
  }

  MS_LOG(INFO) << "Stop tcp client!";
  int ret = event_base_loopbreak(event_base_);
  if (ret != 0) {
    MS_LOG(ERROR) << "Event base loop break failed!";
  }
}

void TcpClient::SetTcpNoDelay(const evutil_socket_t &fd) {
  const int one = 1;
  int ret = setsockopt(fd, static_cast<int>(IPPROTO_TCP), static_cast<int>(TCP_NODELAY), &one, sizeof(int));
  if (ret < 0) {
    MS_LOG(EXCEPTION) << "Set socket no delay failed!";
  }
}

void TcpClient::ReadCallback(struct bufferevent *bev, void *const ctx) {
  try {
    MS_EXCEPTION_IF_NULL(ctx);
    auto tcp_client = reinterpret_cast<TcpClient *>(ctx);
    tcp_client->ReadCallbackInner(bev);
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "Catch exception: " << e.what();
  }
}

void TcpClient::ReadCallbackInner(struct bufferevent *bev) {
  MS_EXCEPTION_IF_NULL(bev);

  char read_buffer[kMessageChunkLength];
  size_t read = 0;

  while ((read = bufferevent_read(bev, &read_buffer, sizeof(read_buffer))) > 0) {
    OnReadHandler(read_buffer, read);
  }
}

void TcpClient::OnReadHandler(const void *buf, size_t num) {
  MS_EXCEPTION_IF_NULL(buf);
  if (read_callback_) {
    read_callback_(buf, num);
  }
  message_handler_.ReceiveMessage(buf, num);
}

void TcpClient::NotifyConnected() {
  MS_LOG(INFO) << "Client connected to the server! Peer " << PeerRoleName() << " ip: " << server_address_
               << ", port: " << server_port_;
  connection_status_ = 1;
  connection_cond_.notify_all();
}

bool TcpClient::EstablishSSL() {
  MS_LOG(INFO) << "Enable ssl support.";

  SSL *ssl = SSL_new(SSLClient::GetInstance().GetSSLCtx());
  MS_ERROR_IF_NULL_W_RET_VAL(ssl, false);
  MS_ERROR_IF_NULL_W_RET_VAL(event_base_, false);

  buffer_event_ = bufferevent_openssl_socket_new(event_base_, -1, ssl, BUFFEREVENT_SSL_CONNECTING,
                                                 BEV_OPT_CLOSE_ON_FREE | BEV_OPT_THREADSAFE);
  return true;
}

void TcpClient::EventCallback(struct bufferevent *bev, std::int16_t events, void *const ptr) {
  try {
    MS_EXCEPTION_IF_NULL(ptr);
    auto tcp_client = reinterpret_cast<TcpClient *>(ptr);
    tcp_client->EventCallbackInner(bev, events);
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "Catch exception: " << e.what();
  }
}

void TcpClient::EventCallbackInner(struct bufferevent *bev, std::int16_t events) {
  MS_EXCEPTION_IF_NULL(bev);
  if (events & BEV_EVENT_CONNECTED) {
    // Connected
    if (connected_callback_) {
      connected_callback_();
    }
    NotifyConnected();
    evutil_socket_t fd = bufferevent_getfd(bev);
    SetTcpNoDelay(fd);
    MS_LOG(INFO) << "Client connected! Peer " << PeerRoleName() << " ip: " << server_address_
                 << ", port: " << server_port_;
  } else if (events & BEV_EVENT_ERROR) {
    if (PSContext::instance()->enable_ssl()) {
      uint64_t err = bufferevent_get_openssl_error(bev);
      const uint64_t server_not_start_err = 5;
      if (err != server_not_start_err) {
        MS_LOG(WARNING) << "The error number is:" << err << ", error message:" << ERR_reason_error_string(err)
                        << ", the error lib:" << ERR_lib_error_string(err)
                        << ", the error func:" << ERR_func_error_string(err);
        return;
      }
    }
    connection_status_ = -1;
    if (disconnected_callback_) {
      MS_LOG(WARNING) << "The client will retry to connect to the server! Peer " << PeerRoleName()
                      << " ip: " << server_address_ << ", port: " << server_port_;
      disconnected_callback_();
    }
  } else if (events & BEV_EVENT_EOF) {
    MS_LOG(WARNING) << "Client connected end of file! Peer " << PeerRoleName() << " ip: " << server_address_
                    << ", port: " << server_port_;
    connection_status_ = -1;
    if (disconnected_callback_) {
      disconnected_callback_();
    }
  }
}

void TcpClient::Start() {
  event_base_mutex_.lock();
  if (is_started_) {
    event_base_mutex_.unlock();
    return;
  }
  is_started_ = true;
  event_base_mutex_.unlock();
  MS_EXCEPTION_IF_NULL(event_base_);
  int ret = event_base_dispatch(event_base_);
  // is_started_ should be false when finish dispatch
  is_started_ = false;
  MSLOG_IF(MsLogLevel::kInfo, ret == 0, NoExceptionType, nullptr) << "Event base dispatch success!";
  MSLOG_IF(MsLogLevel::kError, ret == 1, NoExceptionType, nullptr)
    << "Event base dispatch failed with no events pending or active!";
  MSLOG_IF(MsLogLevel::kError, ret == -1, NoExceptionType, nullptr)
    << "Event base dispatch failed with error occurred!";
  MSLOG_IF(MsLogLevel::kException, ret < -1, AbortedError, nullptr)
    << "Event base dispatch with unexpected error code!";
}

void TcpClient::SetMessageCallback(const OnMessage &cb) { message_callback_ = cb; }

bool TcpClient::SendMessage(const CommMessage &message) const {
  MS_EXCEPTION_IF_NULL(buffer_event_);
  bufferevent_lock(buffer_event_);
  bool res = true;
  size_t buf_size = message.ByteSizeLong();
  uint32_t meta_size = SizeToUint(message.pb_meta().ByteSizeLong());
  MessageHeader header;
  header.message_proto_ = Protos::PROTOBUF;
  header.message_length_ = buf_size;
  header.message_meta_length_ = meta_size;
  if (bufferevent_write(buffer_event_, &header, sizeof(header)) == -1) {
    MS_LOG(ERROR) << "Event buffer add header failed!";
    res = false;
  }
  if (bufferevent_write(buffer_event_, message.pb_meta().SerializeAsString().data(), meta_size) == -1) {
    MS_LOG(ERROR) << "Event buffer add protobuf data failed!";
    res = false;
  }
  if (bufferevent_write(buffer_event_, message.data().data(), message.data().length()) == -1) {
    MS_LOG(ERROR) << "Event buffer add protobuf data failed!";
    res = false;
  }
  bufferevent_unlock(buffer_event_);
  return res;
}

bool TcpClient::SendMessage(const std::shared_ptr<MessageMeta> &meta, const Protos &protos, const void *data,
                            size_t size) {
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
    MS_LOG(ERROR) << "Bufferevent flush failed!";
    res = false;
  }
  bufferevent_unlock(buffer_event_);
  return res;
}

void TcpClient::set_timer_callback(const OnTimer &timer) { on_timer_callback_ = timer; }

const event_base &TcpClient::eventbase() const { return *event_base_; }
}  // namespace core
}  // namespace ps
}  // namespace mindspore
