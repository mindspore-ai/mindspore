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

#include "ps/core/communicator/http_client.h"

#include <arpa/inet.h>
#include <event2/buffer.h>
#include <event2/buffer_compat.h>
#include <event2/bufferevent.h>
#include <event2/event.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <utility>

namespace mindspore {
namespace ps {
namespace core {
HttpClient::HttpClient(const std::string &server_domain)
    : server_domain_(std::move(server_domain)),
      event_base_(nullptr),
      buffer_event_(nullptr),
      http_req_(nullptr),
      evhttp_conn_(nullptr),
      uri(nullptr) {}

HttpClient::~HttpClient() {
  if (buffer_event_) {
    bufferevent_free(buffer_event_);
    buffer_event_ = nullptr;
  }
  if (evhttp_conn_) {
    evhttp_connection_free(evhttp_conn_);
    evhttp_conn_ = nullptr;
  }
  if (event_base_) {
    event_base_free(event_base_);
    event_base_ = nullptr;
  }
}

void HttpClient::Init() {
  std::lock_guard<std::mutex> lock(connection_mutex_);
  if (buffer_event_) {
    bufferevent_free(buffer_event_);
    buffer_event_ = nullptr;
  }
  if (!CommUtil::CheckHttpUrl(server_domain_)) {
    MS_LOG(EXCEPTION) << "The http client address:" << server_domain_ << " is illegal!";
  }

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
      MS_LOG(EXCEPTION) << "Establish SSL failed.";
    }
  }
  MS_EXCEPTION_IF_NULL(buffer_event_);

  if (bufferevent_enable(buffer_event_, EV_READ | EV_WRITE) == -1) {
    MS_LOG(EXCEPTION) << "Buffer event enable read and write failed!";
  }

  uri = evhttp_uri_parse(server_domain_.c_str());
  int port = evhttp_uri_get_port(uri);
  if (port == -1) {
    MS_LOG(EXCEPTION) << "Http uri port is invalid.";
  }

  if (evhttp_conn_) {
    evhttp_connection_free(evhttp_conn_);
    evhttp_conn_ = nullptr;
  }

  evhttp_conn_ =
    evhttp_connection_base_bufferevent_new(event_base_, nullptr, buffer_event_, evhttp_uri_get_host(uri), port);
  MS_LOG(INFO) << "Host is:" << evhttp_uri_get_host(uri) << ", port is:" << port;
}

bool HttpClient::Stop() {
  MS_ERROR_IF_NULL_W_RET_VAL(event_base_, false);
  std::lock_guard<std::mutex> lock(connection_mutex_);
  if (event_base_got_break(event_base_)) {
    MS_LOG(WARNING) << "The event base has already been stopped!";
    return false;
  }

  MS_LOG(INFO) << "Stop http client!";
  int ret = event_base_loopbreak(event_base_);
  if (ret != 0) {
    MS_LOG(ERROR) << "Event base loop break failed!";
    return false;
  }
  return true;
}

bool HttpClient::BreakLoopEvent() {
  MS_ERROR_IF_NULL_W_RET_VAL(event_base_, false);
  int ret = event_base_loopbreak(event_base_);
  if (ret != 0) {
    MS_LOG(ERROR) << "Event base loop break failed!";
    return false;
  }
  return true;
}

bool HttpClient::EstablishSSL() {
  MS_LOG(INFO) << "Enable http ssl support.";

  SSL *ssl = SSL_new(SSLClient::GetInstance().GetSSLCtx());
  MS_ERROR_IF_NULL_W_RET_VAL(ssl, false);
  MS_ERROR_IF_NULL_W_RET_VAL(event_base_, false);

  buffer_event_ = bufferevent_openssl_socket_new(event_base_, -1, ssl, BUFFEREVENT_SSL_CONNECTING,
                                                 BEV_OPT_CLOSE_ON_FREE | BEV_OPT_THREADSAFE);
  return true;
}

void HttpClient::ReadCallback(struct evhttp_request *http_req, void *const arg) {
  MS_ERROR_IF_NULL_WO_RET_VAL(http_req);
  MS_ERROR_IF_NULL_WO_RET_VAL(arg);

  auto http_client = reinterpret_cast<HttpClient *>(arg);
  MS_ERROR_IF_NULL_WO_RET_VAL(http_client);

  event_base *base = http_client->get_event_base();
  MS_ERROR_IF_NULL_WO_RET_VAL(base);
  MS_LOG(DEBUG) << "http_req->response_code is" << http_req->response_code;
  switch (http_req->response_code) {
    case HTTP_OK: {
      struct evbuffer *evbuf = evhttp_request_get_input_buffer(http_req);
      size_t length = evbuffer_get_length(evbuf);
      MS_LOG(DEBUG) << "data length is:" << length;

      auto response_msg = std::make_shared<std::vector<unsigned char>>(length);
      int ret = memcpy_s(response_msg->data(), length, evbuffer_pullup(evbuf, -1), length);
      if (ret != 0) {
        MS_LOG(ERROR) << "memcpy_s error, errorno(" << ret << ")";
        return;
      }
      http_client->OnReadHandler(http_client->request_id(), http_client->kernel_path(), response_msg);
      event_base_loopbreak(base);
      break;
    }
    case HTTP_MOVEPERM:
      MS_LOG(WARNING) << "the uri moved permanently";
      break;
    default:
      MS_LOG(WARNING) << "Default: event base loop break.";
      event_base_loopbreak(base);
  }
}

void HttpClient::OnReadHandler(const size_t request_id, const std::string kernel_path,
                               const std::shared_ptr<std::vector<unsigned char>> &response_msg) {
  MS_EXCEPTION_IF_NULL(response_msg->data());
  message_callback_(request_id, kernel_path, response_msg);
}

void HttpClient::SetMessageCallback(const OnMessage &cb) { message_callback_ = cb; }

event_base *HttpClient::get_event_base() const { return event_base_; }

void HttpClient::set_request_id(size_t request_id) { request_id_ = request_id; }

size_t HttpClient::request_id() const { return request_id_; }

void HttpClient::set_kernel_path(const std::string kernel_path) { kernel_path_ = kernel_path; }

std::string HttpClient::kernel_path() const { return kernel_path_; }

bool HttpClient::SendMessage(const std::string &kernel_path, const std::string &content_type, const void *data,
                             size_t data_size, const size_t request_id) {
  std::lock_guard<std::mutex> lock(connection_mutex_);
  MS_LOG(DEBUG) << "kernel_path is:" << kernel_path << ", data_size is:" << data_size
                << ", kernel_path is:" << kernel_path << ", request id:" << request_id;
  set_request_id(request_id);
  set_kernel_path(kernel_path);
  http_req_ = evhttp_request_new(ReadCallback, this);

  /** Set the post data */
  evbuffer_add(http_req_->output_buffer, data, data_size);
  evhttp_add_header(http_req_->output_headers, "Content-Type", content_type.c_str());
  evhttp_add_header(http_req_->output_headers, "Host", evhttp_uri_get_host(uri));
  evhttp_make_request(evhttp_conn_, http_req_, EVHTTP_REQ_POST, kernel_path.c_str());

  int ret = event_base_dispatch(event_base_);
  if (ret != 0) {
    MS_LOG(ERROR) << "Event base dispatch failed!";
    return false;
  }
  return true;
}
}  // namespace core
}  // namespace ps
}  // namespace mindspore
