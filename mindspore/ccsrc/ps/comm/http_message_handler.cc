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

#include "ps/comm/http_message_handler.h"

#include <event2/event.h>
#include <event2/buffer.h>
#include <event2/bufferevent.h>
#include <event2/bufferevent_compat.h>
#include <event2/http.h>
#include <event2/http_compat.h>
#include <event2/http_struct.h>
#include <event2/listener.h>
#include <event2/util.h>

#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <functional>

namespace mindspore {
namespace ps {
namespace comm {

void HttpMessageHandler::InitHttpMessage() {
  MS_EXCEPTION_IF_NULL(event_request_);
  event_uri_ = evhttp_request_get_evhttp_uri(event_request_);
  MS_EXCEPTION_IF_NULL(event_uri_);

  const char *query = evhttp_uri_get_query(event_uri_);
  if (query) {
    evhttp_parse_query_str(query, &path_params_);
  }

  head_params_ = evhttp_request_get_input_headers(event_request_);
  resp_headers_ = evhttp_request_get_output_headers(event_request_);
  resp_buf_ = evhttp_request_get_output_buffer(event_request_);
}

std::string HttpMessageHandler::GetHeadParam(const std::string &key) {
  MS_EXCEPTION_IF_NULL(head_params_);
  const char *val = evhttp_find_header(head_params_, key.c_str());
  MS_EXCEPTION_IF_NULL(val);
  return std::string(val);
}

std::string HttpMessageHandler::GetPathParam(const std::string &key) {
  const char *val = evhttp_find_header(&path_params_, key.c_str());
  MS_EXCEPTION_IF_NULL(val);
  return std::string(val);
}

void HttpMessageHandler::ParsePostParam() {
  MS_EXCEPTION_IF_NULL(event_request_);
  size_t len = evbuffer_get_length(event_request_->input_buffer);
  if (len == 0) {
    MS_LOG(EXCEPTION) << "The post parameter size is: " << len;
  }
  post_param_parsed_ = true;
  const char *post_message = reinterpret_cast<const char *>(evbuffer_pullup(event_request_->input_buffer, -1));
  MS_EXCEPTION_IF_NULL(post_message);
  body_ = std::make_unique<std::string>(post_message, len);
  int ret = evhttp_parse_query_str(body_->c_str(), &post_params_);
  if (ret == -1) {
    MS_LOG(EXCEPTION) << "Parse post parameter failed!";
  }
}

std::string HttpMessageHandler::GetPostParam(const std::string &key) {
  if (!post_param_parsed_) {
    ParsePostParam();
  }

  const char *val = evhttp_find_header(&post_params_, key.c_str());
  MS_EXCEPTION_IF_NULL(val);
  return std::string(val);
}

std::string HttpMessageHandler::GetRequestUri() {
  MS_EXCEPTION_IF_NULL(event_request_);
  const char *uri = evhttp_request_get_uri(event_request_);
  MS_EXCEPTION_IF_NULL(uri);
  return std::string(uri);
}

std::string HttpMessageHandler::GetRequestHost() {
  MS_EXCEPTION_IF_NULL(event_request_);
  const char *host = evhttp_request_get_host(event_request_);
  MS_EXCEPTION_IF_NULL(host);
  return std::string(host);
}

int HttpMessageHandler::GetUriPort() {
  MS_EXCEPTION_IF_NULL(event_uri_);
  return evhttp_uri_get_port(event_uri_);
}

std::string HttpMessageHandler::GetUriPath() {
  MS_EXCEPTION_IF_NULL(event_uri_);
  const char *path = evhttp_uri_get_path(event_uri_);
  MS_EXCEPTION_IF_NULL(path);
  return std::string(path);
}

std::string HttpMessageHandler::GetUriQuery() {
  MS_EXCEPTION_IF_NULL(event_uri_);
  const char *query = evhttp_uri_get_query(event_uri_);
  MS_EXCEPTION_IF_NULL(query);
  return std::string(query);
}

std::string HttpMessageHandler::GetUriFragment() {
  MS_EXCEPTION_IF_NULL(event_uri_);
  const char *fragment = evhttp_uri_get_fragment(event_uri_);
  MS_EXCEPTION_IF_NULL(fragment);
  return std::string(fragment);
}

std::string HttpMessageHandler::GetPostMsg() {
  MS_EXCEPTION_IF_NULL(event_request_);
  if (body_ != nullptr) {
    return *body_;
  }
  size_t len = evbuffer_get_length(event_request_->input_buffer);
  if (len == 0) {
    MS_LOG(EXCEPTION) << "The post message is empty!";
  }
  const char *post_message = reinterpret_cast<const char *>(evbuffer_pullup(event_request_->input_buffer, -1));
  MS_EXCEPTION_IF_NULL(post_message);
  body_ = std::make_unique<std::string>(post_message, len);
  return *body_;
}

void HttpMessageHandler::AddRespHeadParam(const std::string &key, const std::string &val) {
  MS_EXCEPTION_IF_NULL(resp_headers_);
  if (evhttp_add_header(resp_headers_, key.c_str(), val.c_str()) != 0) {
    MS_LOG(EXCEPTION) << "Add parameter of response header failed.";
  }
}

void HttpMessageHandler::AddRespHeaders(const HttpHeaders &headers) {
  for (auto iter = headers.begin(); iter != headers.end(); ++iter) {
    auto list = iter->second;
    for (auto iterator_val = list.begin(); iterator_val != list.end(); ++iterator_val) {
      AddRespHeadParam(iter->first, *iterator_val);
    }
  }
}

void HttpMessageHandler::AddRespString(const std::string &str) {
  MS_EXCEPTION_IF_NULL(resp_buf_);
  if (evbuffer_add_printf(resp_buf_, "%s", str.c_str()) == -1) {
    MS_LOG(EXCEPTION) << "Add string to response body failed.";
  }
}

void HttpMessageHandler::SetRespCode(int code) { resp_code_ = code; }

void HttpMessageHandler::SendResponse() {
  MS_EXCEPTION_IF_NULL(event_request_);
  MS_EXCEPTION_IF_NULL(resp_buf_);
  evhttp_send_reply(event_request_, resp_code_, nullptr, resp_buf_);
}

void HttpMessageHandler::QuickResponse(int code, const std::string &body) {
  MS_EXCEPTION_IF_NULL(event_request_);
  MS_EXCEPTION_IF_NULL(resp_buf_);
  AddRespString(body);
  evhttp_send_reply(event_request_, code, nullptr, resp_buf_);
}

void HttpMessageHandler::SimpleResponse(int code, const HttpHeaders &headers, const std::string &body) {
  MS_EXCEPTION_IF_NULL(event_request_);
  MS_EXCEPTION_IF_NULL(resp_buf_);
  AddRespHeaders(headers);
  AddRespString(body);
  MS_EXCEPTION_IF_NULL(resp_buf_);
  evhttp_send_reply(event_request_, resp_code_, nullptr, resp_buf_);
}

void HttpMessageHandler::RespError(int nCode, const std::string &message) {
  MS_EXCEPTION_IF_NULL(event_request_);
  if (message.empty()) {
    evhttp_send_error(event_request_, nCode, nullptr);
  } else {
    evhttp_send_error(event_request_, nCode, message.c_str());
  }
}
}  // namespace comm
}  // namespace ps
}  // namespace mindspore
