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

#ifndef MINDSPORE_CCSRC_PS_CORE_COMMUNICATOR_WORKER_QUEUE_H_
#define MINDSPORE_CCSRC_PS_CORE_COMMUNICATOR_WORKER_QUEUE_H_

#include <event2/event.h>
#include <event2/http.h>
#include <event2/http_struct.h>

#include <string>
#include <memory>
#include <unordered_map>

#include "utils/log_adapter.h"
#include "ps/core/communicator/http_message_handler.h"

namespace mindspore {
namespace ps {
namespace core {
using OnRequestReceive = std::function<void(std::shared_ptr<HttpMessageHandler>)>;
class WorkerQueue {
 public:
  WorkerQueue() : evbase_(nullptr) {}
  virtual ~WorkerQueue() = default;

  bool Initialize(int fd, std::unordered_map<std::string, OnRequestReceive *> handlers);
  void Run();
  void Stop();

 private:
  struct event_base *evbase_;
};
}  // namespace core
}  // namespace ps
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PS_CORE_COMMUNICATOR_WORKER_QUEUE_H_
