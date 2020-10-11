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

#include "ps/comm/http_server.h"
#include "ps/comm/http_message_handler.h"

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
namespace comm {

HttpServer::~HttpServer() {
  if (event_http_) {
    evhttp_free(event_http_);
    event_http_ = nullptr;
  }
  if (event_base_) {
    event_base_free(event_base_);
    event_base_ = nullptr;
  }
}

bool HttpServer::InitServer() {
  if (!CheckIp(server_address_)) {
    MS_LOG(EXCEPTION) << "Server address" << server_address_ << " illegal!";
  }
  int64_t uAddr = inet_addr(server_address_.c_str());
  if (INADDR_NONE == uAddr) {
    MS_LOG(EXCEPTION) << "Server address illegal, inet_addr converting failed!";
  }
  if (server_port_ <= 0) {
    MS_LOG(EXCEPTION) << "Server port:" << server_port_ << " illegal!";
  }

  event_base_ = event_base_new();
  MS_EXCEPTION_IF_NULL(event_base_);
  event_http_ = evhttp_new(event_base_);
  MS_EXCEPTION_IF_NULL(event_http_);
  int ret = evhttp_bind_socket(event_http_, server_address_.c_str(), server_port_);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "Http bind server addr:" << server_address_.c_str() << " port:" << server_port_ << "failed";
  }
  is_init_ = true;
  return true;
}

bool HttpServer::CheckIp(const std::string &ip) {
  std::regex pattern("((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?).){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)");
  std::smatch res;
  if (regex_match(ip, res, pattern)) {
    return true;
  }
  return false;
}

void HttpServer::SetTimeOut(int seconds) {
  MS_EXCEPTION_IF_NULL(event_http_);
  if (seconds < 0) {
    MS_LOG(EXCEPTION) << "The timeout seconds:" << seconds << "is less than 0!";
  }
  evhttp_set_timeout(event_http_, seconds);
}

void HttpServer::SetAllowedMethod(HttpMethodsSet methods) {
  MS_EXCEPTION_IF_NULL(event_http_);
  evhttp_set_allowed_methods(event_http_, methods);
}

void HttpServer::SetMaxHeaderSize(size_t num) {
  MS_EXCEPTION_IF_NULL(event_http_);
  if (num < 0) {
    MS_LOG(EXCEPTION) << "The header num:" << num << "is less than 0!";
  }
  evhttp_set_max_headers_size(event_http_, num);
}

void HttpServer::SetMaxBodySize(size_t num) {
  MS_EXCEPTION_IF_NULL(event_http_);
  if (num < 0) {
    MS_LOG(EXCEPTION) << "The max body num:" << num << "is less than 0!";
  }
  evhttp_set_max_body_size(event_http_, num);
}

bool HttpServer::RegisterRoute(const std::string &url, handle_t *function) {
  if ((!is_init_) && (!InitServer())) {
    MS_LOG(EXCEPTION) << "Init http server failed!";
  }
  HandlerFunc func = function;
  if (!func) {
    return false;
  }

  auto TransFunc = [](struct evhttp_request *req, void *arg) {
    MS_EXCEPTION_IF_NULL(req);
    MS_EXCEPTION_IF_NULL(arg);
    HttpMessageHandler httpReq(req);
    httpReq.InitHttpMessage();
    handle_t *f = reinterpret_cast<handle_t *>(arg);
    f(&httpReq);
  };
  handle_t **pph = func.target<handle_t *>();
  MS_EXCEPTION_IF_NULL(pph);
  MS_EXCEPTION_IF_NULL(event_http_);

  // O SUCCESS,-1 ALREADY_EXIST,-2 FAILURE
  int ret = evhttp_set_cb(event_http_, url.c_str(), TransFunc, reinterpret_cast<void *>(*pph));
  if (ret == 0) {
    MS_LOG(INFO) << "Ev http register handle of:" << url.c_str() << " success.";
  } else if (ret == -1) {
    MS_LOG(WARNING) << "Ev http register handle of:" << url.c_str() << " exist.";
  } else {
    MS_LOG(ERROR) << "Ev http register handle of:" << url.c_str() << " failed.";
    return false;
  }
  return true;
}

bool HttpServer::UnRegisterRoute(const std::string &url) {
  MS_EXCEPTION_IF_NULL(event_http_);
  return (evhttp_del_cb(event_http_, url.c_str()) == 0);
}

bool HttpServer::Start() {
  MS_EXCEPTION_IF_NULL(event_base_);
  int ret = event_base_dispatch(event_base_);
  if (ret == 0) {
    MS_LOG(INFO) << "Event base dispatch success!";
    return true;
  } else if (ret == 1) {
    MS_LOG(ERROR) << "Event base dispatch failed with no events pending or active!";
    return false;
  } else if (ret == -1) {
    MS_LOG(ERROR) << "Event base dispatch failed with error occurred!";
    return false;
  } else {
    MS_LOG(EXCEPTION) << "Event base dispatch with unexpect error code!";
  }
}

void HttpServer::Stop() {
  if (event_http_) {
    evhttp_free(event_http_);
    event_http_ = nullptr;
  }
  if (event_base_) {
    event_base_free(event_base_);
    event_base_ = nullptr;
  }
}

}  // namespace comm
}  // namespace ps
}  // namespace mindspore
