/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "ps/core/communicator/http_request_handler.h"

namespace mindspore {
namespace ps {
namespace core {
HttpRequestHandler::~HttpRequestHandler() {
  if (evbase_) {
    event_base_free(evbase_);
    evbase_ = nullptr;
  }
}

bool HttpRequestHandler::Initialize(int fd, const std::unordered_map<std::string, OnRequestReceive *> &handlers) {
  evbase_ = event_base_new();
  MS_EXCEPTION_IF_NULL(evbase_);
  struct evhttp *http = evhttp_new(evbase_);
  MS_EXCEPTION_IF_NULL(http);

  if (PSContext::instance()->enable_ssl()) {
    MS_LOG(INFO) << "Enable ssl support.";
    if (!SSL_CTX_set_options(SSLHTTP::GetInstance().GetSSLCtx(), SSL_OP_SINGLE_DH_USE | SSL_OP_SINGLE_ECDH_USE |
                                                                   SSL_OP_NO_SSLv2 | SSL_OP_NO_SSLv3 | SSL_OP_NO_TLSv1 |
                                                                   SSL_OP_NO_TLSv1_1)) {
      if (evbase_) {
        event_base_free(evbase_);
        evbase_ = nullptr;
      }
      evhttp_free(http);
      http = nullptr;
      MS_LOG(EXCEPTION) << "SSL_CTX_set_options failed.";
    }
    evhttp_set_bevcb(http, BuffereventCallback, SSLHTTP::GetInstance().GetSSLCtx());
  }

  int result = evhttp_accept_socket(http, fd);
  if (result < 0) {
    MS_LOG(ERROR) << "Evhttp accept socket failed!";
    return false;
  }

  for (const auto &handler : handlers) {
    auto TransFunc = [](struct evhttp_request *req, void *arg) {
      MS_EXCEPTION_IF_NULL(req);
      MS_EXCEPTION_IF_NULL(arg);
      auto httpReq = std::make_shared<HttpMessageHandler>();
      MS_EXCEPTION_IF_NULL(httpReq);
      httpReq->set_request(req);
      httpReq->InitHttpMessage();
      OnRequestReceive *func = reinterpret_cast<OnRequestReceive *>(arg);
      (*func)(httpReq);
    };

    // O SUCCESS,-1 ALREADY_EXIST,-2 FAILURE
    MS_EXCEPTION_IF_NULL(handler.second);
    int ret = evhttp_set_cb(http, handler.first.c_str(), TransFunc, reinterpret_cast<void *>(handler.second));
    std::string log_prefix = "Ev http register handle of:";
    if (ret == 0) {
      MS_LOG(INFO) << log_prefix << handler.first.c_str() << " success.";
    } else if (ret == -1) {
      MS_LOG(WARNING) << log_prefix << handler.first.c_str() << " exist.";
    } else {
      MS_LOG(ERROR) << log_prefix << handler.first.c_str() << " failed.";
      return false;
    }
  }
  return true;
}

void HttpRequestHandler::Run() {
  MS_LOG(INFO) << "Start http server!";
  MS_EXCEPTION_IF_NULL(evbase_);
  int ret = event_base_dispatch(evbase_);
  if (ret == 0) {
    MS_LOG(INFO) << "Event base dispatch success!";
  } else if (ret == 1) {
    MS_LOG(ERROR) << "Event base dispatch failed with no events pending or active!";
  } else if (ret == -1) {
    MS_LOG(ERROR) << "Event base dispatch failed with error occurred!";
  } else {
    MS_LOG(ERROR) << "Event base dispatch with unexpected error code!";
  }
}

bool HttpRequestHandler::Stop() {
  MS_LOG(INFO) << "Stop http server!";

  MS_EXCEPTION_IF_NULL(evbase_);
  int ret = event_base_loopbreak(evbase_);
  if (ret != 0) {
    MS_LOG(ERROR) << "event base loop break failed!";
    return false;
  }
  return true;
}

bufferevent *HttpRequestHandler::BuffereventCallback(event_base *base, void *arg) {
  MS_EXCEPTION_IF_NULL(base);
  MS_EXCEPTION_IF_NULL(arg);
  SSL_CTX *ctx = reinterpret_cast<SSL_CTX *>(arg);
  SSL *ssl = SSL_new(ctx);
  MS_EXCEPTION_IF_NULL(ssl);
  bufferevent *bev =
    bufferevent_openssl_socket_new(base, -1, ssl, BUFFEREVENT_SSL_ACCEPTING, static_cast<int>(BEV_OPT_CLOSE_ON_FREE));
  MS_EXCEPTION_IF_NULL(bev);
  return bev;
}
}  // namespace core
}  // namespace ps
}  // namespace mindspore
