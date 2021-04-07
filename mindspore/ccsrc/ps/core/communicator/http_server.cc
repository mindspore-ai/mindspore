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

#include "ps/core/communicator/http_server.h"
#include "ps/core/communicator/http_message_handler.h"
#include "ps/core/comm_util.h"

#ifdef WIN32
#include <WinSock2.h>
#endif
#include <arpa/inet.h>
#include <event.h>
#include <event2/buffer.h>
#include <event2/bufferevent.h>
#include <event2/bufferevent_compat.h>
#include <event2/http.h>
#include <event2/http_compat.h>
#include <event2/http_struct.h>
#include <event2/listener.h>
#include <event2/util.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <regex>

namespace mindspore {
namespace ps {
namespace core {
HttpServer::~HttpServer() { Stop(); }

bool HttpServer::InitServer() {
  if (!CommUtil::CheckIp(server_address_)) {
    MS_LOG(ERROR) << "The http server ip:" << server_address_ << " is illegal!";
    return false;
  }

  is_stop_ = false;
  int result = evthread_use_pthreads();
  if (result != 0) {
    MS_LOG(ERROR) << "Use event pthread failed!";
    return false;
  }

  fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd_ < 0) {
    MS_LOG(ERROR) << "Socker error!";
    return false;
  }

  int one = 1;
  result = setsockopt(fd_, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<char *>(&one), sizeof(int));
  if (result < 0) {
    MS_LOG(ERROR) << "Set sock opt error!";
    return false;
  }

  struct sockaddr_in addr;
  errno_t ret = memset_s(&addr, sizeof(addr), 0, sizeof(addr));
  if (ret != EOK) {
    MS_LOG(EXCEPTION) << "Memset failed.";
  }

  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = inet_addr(server_address_.c_str());
  addr.sin_port = htons(server_port_);

  result = ::bind(fd_, (struct sockaddr *)&addr, sizeof(addr));
  if (result < 0) {
    MS_LOG(ERROR) << "Bind ip:" << server_address_ << " port:" << server_port_ << "failed!";
    return false;
  }

  result = ::listen(fd_, backlog_);
  if (result < 0) {
    MS_LOG(ERROR) << "Listen ip:" << server_address_ << " port:" << server_port_ << "failed!";
    return false;
  }

  int flags = 0;
  if ((flags = fcntl(fd_, F_GETFL, 0)) < 0 || fcntl(fd_, F_SETFL, flags | O_NONBLOCK) < 0) {
    MS_LOG(ERROR) << "Set fcntl O_NONBLOCK failed!";
    return false;
  }

  return true;
}

void HttpServer::SetTimeOut(int seconds) {
  if (seconds < 0) {
    MS_LOG(EXCEPTION) << "The timeout seconds:" << seconds << "is less than 0!";
  }
  request_timeout_ = seconds;
}

bool HttpServer::RegisterRoute(const std::string &url, OnRequestReceive *function) {
  if (!function) {
    return false;
  }
  request_handlers_[url] = function;
  return true;
}

bool HttpServer::Start() {
  MS_LOG(INFO) << "Start http server!";
  for (size_t i = 0; i < thread_num_; i++) {
    auto http_request_handler = std::make_shared<HttpRequestHandler>();
    http_request_handler->Initialize(fd_, request_handlers_);
    http_request_handlers.push_back(http_request_handler);
    worker_threads_.emplace_back(std::make_shared<std::thread>(&HttpRequestHandler::Run, http_request_handler));
  }
  return true;
}

bool HttpServer::Wait() {
  for (size_t i = 0; i < thread_num_; i++) {
    worker_threads_[i]->join();
    worker_threads_[i].reset();
  }
  return true;
}

void HttpServer::Stop() {
  MS_LOG(INFO) << "Stop http server!";

  if (!is_stop_.load()) {
    for (size_t i = 0; i < thread_num_; i++) {
      http_request_handlers[i]->Stop();
    }
    is_stop_ = true;
  }
}
}  // namespace core
}  // namespace ps
}  // namespace mindspore
