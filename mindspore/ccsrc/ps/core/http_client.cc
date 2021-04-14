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

#include "ps/core/http_client.h"

namespace mindspore {
namespace ps {
namespace core {
HttpClient::~HttpClient() {
  if (event_base_ != nullptr) {
    event_base_free(event_base_);
    event_base_ = nullptr;
  }
}

void HttpClient::Init() {
  event_base_ = event_base_new();
  MS_EXCEPTION_IF_NULL(event_base_);
  dns_base_ = evdns_base_new(event_base_, 1);
  MS_EXCEPTION_IF_NULL(dns_base_);
}

Status HttpClient::Post(const std::string &url, const void *body, size_t len, std::shared_ptr<std::vector<char>> output,
                        const std::map<std::string, std::string> &headers) {
  MS_EXCEPTION_IF_NULL(body);
  MS_EXCEPTION_IF_NULL(output);
  auto handler = std::make_shared<HttpMessageHandler>();
  output->clear();
  handler->set_body(output);

  struct evhttp_request *request = evhttp_request_new(ReadCallback, reinterpret_cast<void *>(handler.get()));
  MS_EXCEPTION_IF_NULL(request);

  InitRequest(handler, url, request);

  struct evhttp_connection *connection =
    evhttp_connection_base_new(event_base_, dns_base_, handler->GetHostByUri(), handler->GetUriPort());
  if (connection == nullptr) {
    MS_LOG(ERROR) << "Create http connection failed!";
    return Status::BADREQUEST;
  }

  struct evbuffer *buffer = evhttp_request_get_output_buffer(request);
  if (evbuffer_add(buffer, body, len) != 0) {
    MS_LOG(ERROR) << "Add buffer failed!";
    return Status::INTERNAL;
  }

  AddHeaders(headers, request, handler);

  return CreateRequest(handler, connection, request, HttpMethod::HM_POST);
}

Status HttpClient::Get(const std::string &url, std::shared_ptr<std::vector<char>> output,
                       const std::map<std::string, std::string> &headers) {
  MS_EXCEPTION_IF_NULL(output);
  auto handler = std::make_shared<HttpMessageHandler>();
  output->clear();
  handler->set_body(output);

  struct evhttp_request *request = evhttp_request_new(ReadCallback, reinterpret_cast<void *>(handler.get()));
  MS_EXCEPTION_IF_NULL(request);

  InitRequest(handler, url, request);

  struct evhttp_connection *connection =
    evhttp_connection_base_new(event_base_, dns_base_, handler->GetHostByUri(), handler->GetUriPort());
  if (connection == nullptr) {
    MS_LOG(ERROR) << "Create http connection failed!";
    return Status::BADREQUEST;
  }

  AddHeaders(headers, request, handler);

  return CreateRequest(handler, connection, request, HttpMethod::HM_GET);
}

void HttpClient::set_connection_timeout(const int &timeout) { connection_timout_ = timeout; }

void HttpClient::ReadCallback(struct evhttp_request *request, void *arg) {
  MS_EXCEPTION_IF_NULL(request);
  MS_EXCEPTION_IF_NULL(arg);
  auto handler = static_cast<HttpMessageHandler *>(arg);
  if (event_base_loopexit(const_cast<event_base *>(handler->http_base()), nullptr) != 0) {
    MS_LOG(EXCEPTION) << "event base loop exit failed!";
  }
}

int HttpClient::ReadHeaderDoneCallback(struct evhttp_request *request, void *arg) {
  MS_EXCEPTION_IF_NULL(request);
  MS_EXCEPTION_IF_NULL(arg);
  auto handler = static_cast<HttpMessageHandler *>(arg);
  handler->set_request(request);
  struct evkeyvalq *headers = evhttp_request_get_input_headers(request);
  MS_EXCEPTION_IF_NULL(headers);
  struct evkeyval *header = nullptr;
  TAILQ_FOREACH(header, headers, next) {
    MS_LOG(DEBUG) << "The key:" << header->key << ",The value:" << header->value;
    std::string len = "Content-Length";
    if (!strcmp(header->key, len.c_str())) {
      handler->set_content_len(strtouq(header->value, nullptr, 10));
      handler->InitBodySize();
    }
  }
  return 0;
}

void HttpClient::ReadChunkDataCallback(struct evhttp_request *request, void *arg) {
  MS_EXCEPTION_IF_NULL(request);
  MS_EXCEPTION_IF_NULL(arg);
  auto handler = static_cast<HttpMessageHandler *>(arg);
  char buf[kMessageChunkLength];
  struct evbuffer *evbuf = evhttp_request_get_input_buffer(request);
  MS_EXCEPTION_IF_NULL(evbuf);
  int n = 0;
  while ((n = evbuffer_remove(evbuf, &buf, sizeof(buf))) > 0) {
    handler->ReceiveMessage(buf, n);
  }
}

void HttpClient::RequestErrorCallback(enum evhttp_request_error error, void *arg) {
  MS_EXCEPTION_IF_NULL(arg);
  auto handler = static_cast<HttpMessageHandler *>(arg);
  MS_LOG(ERROR) << "The request failed, the error is:" << error;
  if (event_base_loopexit(const_cast<event_base *>(handler->http_base()), nullptr) != 0) {
    MS_LOG(EXCEPTION) << "event base loop exit failed!";
  }
}

void HttpClient::ConnectionCloseCallback(struct evhttp_connection *connection, void *arg) {
  MS_EXCEPTION_IF_NULL(connection);
  MS_EXCEPTION_IF_NULL(arg);
  MS_LOG(ERROR) << "Remote connection closed!";
  if (event_base_loopexit((struct event_base *)arg, nullptr) != 0) {
    MS_LOG(EXCEPTION) << "event base loop exit failed!";
  }
}

void HttpClient::AddHeaders(const std::map<std::string, std::string> &headers, const struct evhttp_request *request,
                            std::shared_ptr<HttpMessageHandler> handler) {
  MS_EXCEPTION_IF_NULL(request);
  if (evhttp_add_header(evhttp_request_get_output_headers(const_cast<evhttp_request *>(request)), "Host",
                        handler->GetHostByUri()) != 0) {
    MS_LOG(EXCEPTION) << "Add header failed!";
  }
  for (auto &header : headers) {
    if (evhttp_add_header(evhttp_request_get_output_headers(const_cast<evhttp_request *>(request)), header.first.data(),
                          header.second.data()) != 0) {
      MS_LOG(EXCEPTION) << "Add header failed!";
    }
  }
}

void HttpClient::InitRequest(std::shared_ptr<HttpMessageHandler> handler, const std::string &url,
                             const struct evhttp_request *request) {
  MS_EXCEPTION_IF_NULL(request);
  MS_EXCEPTION_IF_NULL(handler);
  handler->set_http_base(event_base_);
  handler->ParseUrl(url);
  evhttp_request_set_header_cb(const_cast<evhttp_request *>(request), ReadHeaderDoneCallback);
  evhttp_request_set_chunked_cb(const_cast<evhttp_request *>(request), ReadChunkDataCallback);
  evhttp_request_set_error_cb(const_cast<evhttp_request *>(request), RequestErrorCallback);

  MS_LOG(DEBUG) << "The url is:" << url << ", The host is:" << handler->GetHostByUri()
                << ", The port is:" << handler->GetUriPort() << ", The request_url is:" << handler->GetRequestPath();
}

Status HttpClient::CreateRequest(std::shared_ptr<HttpMessageHandler> handler, struct evhttp_connection *connection,
                                 struct evhttp_request *request, HttpMethod method) {
  MS_EXCEPTION_IF_NULL(handler);
  MS_EXCEPTION_IF_NULL(connection);
  MS_EXCEPTION_IF_NULL(request);
  evhttp_connection_set_closecb(connection, ConnectionCloseCallback, event_base_);
  evhttp_connection_set_timeout(connection, connection_timout_);

  if (evhttp_make_request(connection, request, evhttp_cmd_type(method), handler->GetRequestPath().c_str()) != 0) {
    MS_LOG(ERROR) << "Make request failed!";
    return Status::INTERNAL;
  }

  if (!Start()) {
    MS_LOG(ERROR) << "Start http client failed!";
    return Status::INTERNAL;
  }

  if (handler->request()) {
    return Status(evhttp_request_get_response_code(handler->request()));
  }
  return Status::INTERNAL;
}

bool HttpClient::Start() {
  MS_EXCEPTION_IF_NULL(event_base_);
  int ret = event_base_loop(event_base_, 0);
  if (ret == 0) {
    MS_LOG(DEBUG) << "Event base dispatch success!";
    return true;
  } else if (ret == 1) {
    MS_LOG(ERROR) << "Event base dispatch failed with no events pending or active!";
    return false;
  } else if (ret == -1) {
    MS_LOG(ERROR) << "Event base dispatch failed with error occurred!";
    return false;
  } else {
    MS_LOG(EXCEPTION) << "Event base dispatch with unexpected error code!";
  }
  return true;
}
}  // namespace core
}  // namespace ps
}  // namespace mindspore
