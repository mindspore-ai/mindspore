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

#ifndef MINDSPORE_CCSRC_PS_CORE_COMMUNICATOR_HTTP_REQUEST_HANDLER_H_
#define MINDSPORE_CCSRC_PS_CORE_COMMUNICATOR_HTTP_REQUEST_HANDLER_H_

#include <event2/event.h>
#include <event2/http.h>
#include <event2/http_struct.h>
#include <event2/bufferevent.h>
#include <event2/bufferevent_ssl.h>

#include <string>
#include <memory>
#include <unordered_map>

#include "utils/log_adapter.h"
#include "ps/core/communicator/http_message_handler.h"
#include "ps/core/communicator/ssl_http.h"
#include "include/backend/distributed/ps/constants.h"
#include "include/backend/distributed/ps/ps_context.h"

namespace mindspore {
namespace ps {
namespace core {
using OnRequestReceive = std::function<void(std::shared_ptr<HttpMessageHandler>)>;

/* Each thread corresponds to one HttpRequestHandler, which is used to create one eventbase. All eventbase are listened
 * on the same fd. Every evhttp_request is executed in one thread.
 */
class HttpRequestHandler {
 public:
  HttpRequestHandler() : evbase_(nullptr) {}
  virtual ~HttpRequestHandler();

  bool Initialize(int fd, const std::unordered_map<std::string, OnRequestReceive *> &handlers);
  void Run();
  bool Stop();
  static bufferevent *BuffereventCallback(event_base *base, void *arg);

 private:
  struct event_base *evbase_;
};
}  // namespace core
}  // namespace ps
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PS_CORE_COMMUNICATOR_HTTP_REQUEST_HANDLER_H_
