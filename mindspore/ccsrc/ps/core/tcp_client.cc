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

#include "ps/core/tcp_client.h"

#include <arpa/inet.h>
#include <event2/buffer.h>
#include <event2/bufferevent.h>
#include <event2/buffer_compat.h>
#include <event2/event.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <utility>
#include <string>

#include "ps/core/comm_util.h"

namespace mindspore {
namespace ps {
namespace core {

TcpClient::TcpClient(const std::string &address, std::uint16_t port)
    : event_base_(nullptr),
      event_timeout_(nullptr),
      buffer_event_(nullptr),
      server_address_(std::move(address)),
      server_port_(port) {
  message_handler_.SetCallback([this](const CommMessage &message) {
    if (message_callback_) {
      message_callback_(*this, message);
    }
  });
}

TcpClient::~TcpClient() { Stop(); }

std::string TcpClient::GetServerAddress() const { return server_address_; }

void TcpClient::SetCallback(const OnConnected &conn, const OnDisconnected &disconn, const OnRead &read,
                            const OnTimeout &timeout) {
  connected_callback_ = conn;
  disconnected_callback_ = disconn;
  read_callback_ = read;
  timeout_callback_ = timeout;
}

void TcpClient::Init() {
  if (buffer_event_) {
    return;
  }
  if (!CommUtil::CheckIp(server_address_)) {
    MS_LOG(EXCEPTION) << "The tcp client ip:" << server_address_ << " is illegal!";
  }

  event_base_ = event_base_new();
  MS_EXCEPTION_IF_NULL(event_base_);

  sockaddr_in sin{};
  if (memset_s(&sin, sizeof(sin), 0, sizeof(sin)) != EOK) {
    MS_LOG(EXCEPTION) << "Initialize sockaddr_in failed!";
  }
  sin.sin_family = AF_INET;
  sin.sin_addr.s_addr = inet_addr(server_address_.c_str());
  sin.sin_port = htons(server_port_);

  buffer_event_ = bufferevent_socket_new(event_base_, -1, BEV_OPT_CLOSE_ON_FREE);
  MS_EXCEPTION_IF_NULL(buffer_event_);

  bufferevent_setcb(buffer_event_, ReadCallback, nullptr, EventCallback, this);
  if (bufferevent_enable(buffer_event_, EV_READ | EV_WRITE) == -1) {
    MS_LOG(EXCEPTION) << "Buffer event enable read and write failed!";
  }

  int result_code = bufferevent_socket_connect(buffer_event_, reinterpret_cast<struct sockaddr *>(&sin), sizeof(sin));
  if (result_code < 0) {
    MS_LOG(EXCEPTION) << "Connect server ip:" << server_address_ << " and port: " << server_port_ << " is failed!";
  }
}

void TcpClient::StartWithDelay(int seconds) {
  if (buffer_event_) {
    return;
  }

  event_base_ = event_base_new();

  timeval timeout_value{};
  timeout_value.tv_sec = seconds;
  timeout_value.tv_usec = 0;

  event_timeout_ = evtimer_new(event_base_, TimeoutCallback, this);
  if (evtimer_add(event_timeout_, &timeout_value) == -1) {
    MS_LOG(EXCEPTION) << "Event timeout failed!";
  }
}

void TcpClient::Stop() {
  if (buffer_event_) {
    bufferevent_free(buffer_event_);
    buffer_event_ = nullptr;
  }

  if (event_timeout_) {
    event_free(event_timeout_);
    event_timeout_ = nullptr;
  }

  if (event_base_) {
    event_base_free(event_base_);
    event_base_ = nullptr;
  }
}

void TcpClient::SetTcpNoDelay(const evutil_socket_t &fd) {
  const int one = 1;
  int ret = setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(int));
  if (ret < 0) {
    MS_LOG(EXCEPTION) << "Set socket no delay failed!";
  }
}

void TcpClient::TimeoutCallback(evutil_socket_t, std::int16_t, void *arg) {
  MS_EXCEPTION_IF_NULL(arg);
  auto tcp_client = reinterpret_cast<TcpClient *>(arg);
  tcp_client->Init();
}

void TcpClient::ReadCallback(struct bufferevent *bev, void *ctx) {
  MS_EXCEPTION_IF_NULL(bev);
  MS_EXCEPTION_IF_NULL(ctx);
  auto tcp_client = reinterpret_cast<TcpClient *>(ctx);
  struct evbuffer *input = bufferevent_get_input(const_cast<struct bufferevent *>(bev));
  MS_EXCEPTION_IF_NULL(input);

  char read_buffer[4096];

  while (EVBUFFER_LENGTH(input) > 0) {
    int read = evbuffer_remove(input, &read_buffer, sizeof(read_buffer));
    if (read == -1) {
      MS_LOG(EXCEPTION) << "Can not drain data from the event buffer!";
    }
    tcp_client->OnReadHandler(read_buffer, read);
  }
}

void TcpClient::OnReadHandler(const void *buf, size_t num) {
  MS_EXCEPTION_IF_NULL(buf);
  if (read_callback_) {
    read_callback_(*this, buf, num);
  }
  message_handler_.ReceiveMessage(buf, num);
}

void TcpClient::SendHeartBeatCallback(evutil_socket_t, int16_t, void *arg) {
  MS_EXCEPTION_IF_NULL(arg);
  auto tcp_client = reinterpret_cast<TcpClient *>(arg);
  MessageMeta meta;
  meta.set_cmd(ClusterCommand::HEARTBEAT);
  CommMessage message;
  message.set_allocated_pb_meta(&meta);
  tcp_client->SendMessage(message);

  struct event *ev;
  struct timeval timeout {};
  timeout.tv_sec = ClusterConfig::heartbeat_interval();
  timeout.tv_usec = 0;
  ev = evtimer_new(tcp_client->event_base_, SendHeartBeatCallback, arg);
  evtimer_add(ev, &timeout);
}

void TcpClient::EventCallback(struct bufferevent *bev, std::int16_t events, void *ptr) {
  MS_EXCEPTION_IF_NULL(bev);
  MS_EXCEPTION_IF_NULL(ptr);
  auto tcp_client = reinterpret_cast<TcpClient *>(ptr);
  if (events & BEV_EVENT_CONNECTED) {
    // Connected
    if (tcp_client->connected_callback_) {
      tcp_client->connected_callback_(*tcp_client);
    }
    evutil_socket_t fd = bufferevent_getfd(const_cast<struct bufferevent *>(bev));
    SetTcpNoDelay(fd);
    MS_LOG(INFO) << "Client connected!";
  } else if (events & BEV_EVENT_ERROR) {
    MS_LOG(ERROR) << "Client connected error!";
    if (tcp_client->disconnected_callback_) {
      tcp_client->disconnected_callback_(*tcp_client, errno);
    }
  } else if (events & BEV_EVENT_EOF) {
    MS_LOG(ERROR) << "Client connected end of file";
    if (tcp_client->disconnected_callback_) {
      tcp_client->disconnected_callback_(*tcp_client, 0);
    }
  }
}

void TcpClient::Start() {
  MS_EXCEPTION_IF_NULL(event_base_);
  int ret = event_base_dispatch(event_base_);
  MSLOG_IF(INFO, ret == 0, NoExceptionType) << "Event base dispatch success!";
  MSLOG_IF(mindspore::ERROR, ret == 1, NoExceptionType)
    << "Event base dispatch failed with no events pending or active!";
  MSLOG_IF(mindspore::ERROR, ret == -1, NoExceptionType) << "Event base dispatch failed with error occurred!";
  MSLOG_IF(mindspore::EXCEPTION, ret < -1, AbortedError) << "Event base dispatch with unexpect error code!";
}

void TcpClient::StartWithNoBlock() {
  MS_LOG(INFO) << "Start tcp client with no block!";
  MS_EXCEPTION_IF_NULL(event_base_);
  int ret = event_base_loop(event_base_, EVLOOP_NONBLOCK);
  MSLOG_IF(INFO, ret == 0, NoExceptionType) << "Event base loop success!";
  MSLOG_IF(mindspore::ERROR, ret == 1, NoExceptionType) << "Event base loop failed with no events pending or active!";
  MSLOG_IF(mindspore::ERROR, ret == -1, NoExceptionType) << "Event base loop failed with error occurred!";
  MSLOG_IF(mindspore::EXCEPTION, ret < -1, AbortedError) << "Event base loop with unexpect error code!";
}

void TcpClient::SetMessageCallback(const OnMessage &cb) { message_callback_ = cb; }

void TcpClient::SendMessage(const CommMessage &message) const {
  MS_EXCEPTION_IF_NULL(buffer_event_);
  uint32_t buf_size = message.ByteSizeLong();
  std::vector<unsigned char> serialized(buf_size);
  message.SerializeToArray(serialized.data(), static_cast<int>(buf_size));
  if (evbuffer_add(bufferevent_get_output(buffer_event_), &buf_size, sizeof(buf_size)) == -1) {
    MS_LOG(EXCEPTION) << "Event buffer add header failed!";
  }
  if (evbuffer_add(bufferevent_get_output(buffer_event_), serialized.data(), buf_size) == -1) {
    MS_LOG(EXCEPTION) << "Event buffer add protobuf data failed!";
  }
}

void TcpClient::SendMessageWithTimer() {
  MS_EXCEPTION_IF_NULL(buffer_event_);
  struct event *ev = nullptr;
  struct timeval timeout {};
  timeout.tv_sec = 0;
  timeout.tv_usec = 0;
  ev = evtimer_new(event_base_, SendHeartBeatCallback, this);
  evtimer_add(ev, &timeout);
}

}  // namespace core
}  // namespace ps
}  // namespace mindspore
