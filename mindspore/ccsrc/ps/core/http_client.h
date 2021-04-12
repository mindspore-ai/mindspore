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

#ifndef MINDSPORE_CCSRC_PS_CORE_HTTP_CLIENT_H_
#define MINDSPORE_CCSRC_PS_CORE_HTTP_CLIENT_H_

#include <event2/buffer.h>
#include <event2/event.h>
#include <event2/http.h>
#include <event2/keyvalq_struct.h>
#include <event2/listener.h>
#include <event2/util.h>
#include <event2/http_struct.h>
#include <event2/dns.h>
#include <event2/thread.h>
#include <sys/queue.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <atomic>
#include <thread>
#include <vector>
#include <map>

#include "ps/core/http_message_handler.h"
#include "ps/core/comm_util.h"

namespace mindspore {
namespace ps {
namespace core {
enum class HttpMethod { HM_GET = 1 << 0, HM_POST = 1 << 1 };

enum class Status : int {
  OK = 200,          // request completed ok
  BADREQUEST = 400,  // invalid http request was made
  NOTFOUND = 404,    // could not find content for uri
  INTERNAL = 500     // internal error
};

class HttpClient {
 public:
  HttpClient() : event_base_(nullptr), dns_base_(nullptr), is_init_(false), connection_timout_(kConnectionTimeout) {
    Init();
  }

  virtual ~HttpClient();

  Status Post(const std::string &url, const void *body, size_t len, std::shared_ptr<std::vector<char>> output,
              const std::map<std::string, std::string> &headers = {});
  Status Get(const std::string &url, std::shared_ptr<std::vector<char>> output,
             const std::map<std::string, std::string> &headers = {});

  void set_connection_timeout(const int &timeout);

 private:
  static void ReadCallback(struct evhttp_request *remote_rsp, void *arg);
  static int ReadHeaderDoneCallback(struct evhttp_request *remote_rsp, void *arg);
  static void ReadChunkDataCallback(struct evhttp_request *remote_rsp, void *arg);
  static void RequestErrorCallback(enum evhttp_request_error error, void *arg);
  static void ConnectionCloseCallback(struct evhttp_connection *connection, void *arg);

  void AddHeaders(const std::map<std::string, std::string> &headers, const struct evhttp_request *request,
                  std::shared_ptr<HttpMessageHandler> handler);
  void InitRequest(std::shared_ptr<HttpMessageHandler> handler, const std::string &url,
                   const struct evhttp_request *request);
  Status CreateRequest(std::shared_ptr<HttpMessageHandler> handler, struct evhttp_connection *connection,
                       struct evhttp_request *request, HttpMethod method);

  bool Start();
  void Init();

  struct event_base *event_base_;
  struct evdns_base *dns_base_;
  bool is_init_;
  int connection_timout_;
};
}  // namespace core
}  // namespace ps
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PS_CORE_HTTP_CLIENT_H_
