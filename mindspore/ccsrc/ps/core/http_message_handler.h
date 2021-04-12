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

#ifndef MINDSPORE_CCSRC_PS_CORE_HTTP_MESSAGE_HANDLER_H_
#define MINDSPORE_CCSRC_PS_CORE_HTTP_MESSAGE_HANDLER_H_

#include <event2/buffer.h>
#include <event2/event.h>
#include <event2/http.h>
#include <event2/keyvalq_struct.h>
#include <event2/listener.h>
#include <event2/util.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <list>
#include <map>
#include <memory>
#include <vector>

#include "ps/core/comm_util.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace ps {
namespace core {
using HttpHeaders = std::map<std::string, std::list<std::string>>;
using VectorPtr = std::shared_ptr<std::vector<char>>;

class HttpMessageHandler {
 public:
  HttpMessageHandler()
      : event_request_(nullptr),
        event_uri_(nullptr),
        path_params_{0},
        head_params_(nullptr),
        post_params_{0},
        post_param_parsed_(false),
        post_message_(nullptr),
        body_(nullptr),
        resp_headers_(nullptr),
        resp_buf_(nullptr),
        resp_code_(HTTP_OK),
        content_len_(0),
        event_base_(nullptr),
        offset_(0) {}

  virtual ~HttpMessageHandler() = default;

  void InitHttpMessage();
  void ParseUrl(const std::string &url);

  std::string GetRequestUri();
  std::string GetRequestHost();
  const char *GetHostByUri();
  std::string GetHeadParam(const std::string &key);
  std::string GetPathParam(const std::string &key);
  std::string GetPostParam(const std::string &key);
  uint64_t GetPostMsg(unsigned char **buffer);
  std::string GetUriPath();
  std::string GetRequestPath();
  std::string GetUriQuery();

  // It will return -1 if no port set
  int GetUriPort();

  // Useless to get from a request url, fragment is only for browser to locate sth.
  std::string GetUriFragment();

  void AddRespHeadParam(const std::string &key, const std::string &val);
  void AddRespHeaders(const HttpHeaders &headers);
  void AddRespString(const std::string &str);
  void SetRespCode(int code);

  // Make sure code and all response body has finished set
  void SendResponse();
  void QuickResponse(int code, const unsigned char *body, size_t len);
  void SimpleResponse(int code, const HttpHeaders &headers, const std::string &body);

  // If message is empty, libevent will use default error code message instead
  void RespError(int nCode, const std::string &message);
  // Body length should no more than MAX_POST_BODY_LEN, default 64kB
  void ParsePostParam();
  void ReceiveMessage(const void *buffer, size_t num);
  void set_content_len(const uint64_t &len);
  uint64_t content_len();
  const event_base *http_base();
  void set_http_base(const struct event_base *base);
  void set_request(const struct evhttp_request *req);
  const struct evhttp_request *request();
  void InitBodySize();
  VectorPtr body();
  void set_body(VectorPtr body);

 private:
  struct evhttp_request *event_request_;
  const struct evhttp_uri *event_uri_;
  struct evkeyvalq path_params_;
  struct evkeyvalq *head_params_;
  struct evkeyvalq post_params_;
  bool post_param_parsed_;
  std::unique_ptr<std::string> post_message_;
  VectorPtr body_;
  struct evkeyvalq *resp_headers_;
  struct evbuffer *resp_buf_;
  int resp_code_;
  uint64_t content_len_;
  struct event_base *event_base_;
  uint64_t offset_;
};
}  // namespace core
}  // namespace ps
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PS_CORE_HTTP_MESSAGE_HANDLER_H_
