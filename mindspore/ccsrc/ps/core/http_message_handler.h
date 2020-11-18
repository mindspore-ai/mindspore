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
#include "utils/log_adapter.h"

namespace mindspore {
namespace ps {
namespace core {

using HttpHeaders = std::map<std::string, std::list<std::string>>;

class HttpMessageHandler {
 public:
  explicit HttpMessageHandler(struct evhttp_request *req)
      : event_request_(req),
        event_uri_(nullptr),
        path_params_{0},
        head_params_(nullptr),
        post_params_{0},
        post_param_parsed_(false),
        body_(nullptr),
        resp_headers_(nullptr),
        resp_buf_(nullptr),
        resp_code_(HTTP_OK) {}

  virtual ~HttpMessageHandler() = default;

  void InitHttpMessage();
  std::string GetRequestUri();
  std::string GetRequestHost();
  std::string GetHeadParam(const std::string &key);
  std::string GetPathParam(const std::string &key);
  std::string GetPostParam(const std::string &key);
  uint64_t GetPostMsg(unsigned char **buffer);
  std::string GetUriPath();
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
  void QuickResponse(int code, const std::string &body);
  void SimpleResponse(int code, const HttpHeaders &headers, const std::string &body);

  // If message is empty, libevent will use default error code message instead
  void RespError(int nCode, const std::string &message);

 private:
  struct evhttp_request *event_request_;
  const struct evhttp_uri *event_uri_;
  struct evkeyvalq path_params_;
  struct evkeyvalq *head_params_;
  struct evkeyvalq post_params_;
  bool post_param_parsed_;
  std::unique_ptr<std::string> body_;
  struct evkeyvalq *resp_headers_;
  struct evbuffer *resp_buf_;
  int resp_code_;

  // Body length should no more than MAX_POST_BODY_LEN, default 64kB
  void ParsePostParam();
};

}  // namespace core
}  // namespace ps
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PS_CORE_HTTP_MESSAGE_HANDLER_H_
