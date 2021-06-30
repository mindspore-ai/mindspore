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

#ifndef MINDSPORE_CCSRC_PS_CORE_COMMUNICATOR_HTTP_COMMUNICATOR_H_
#define MINDSPORE_CCSRC_PS_CORE_COMMUNICATOR_HTTP_COMMUNICATOR_H_

#include <string>
#include <memory>
#include <unordered_map>
#include "ps/core/communicator/http_server.h"
#include "ps/core/communicator/http_message_handler.h"
#include "ps/core/communicator/task_executor.h"
#include "ps/core/communicator/communicator_base.h"
#include "ps/core/communicator/http_msg_handler.h"
#include "utils/ms_exception.h"

namespace mindspore {
namespace ps {
namespace core {
class HttpCommunicator : public CommunicatorBase {
 public:
  explicit HttpCommunicator(const std::string &ip, uint16_t port, const std::shared_ptr<TaskExecutor> &task_executor)
      : task_executor_(task_executor), http_server_(nullptr), ip_(ip), port_(port) {
    http_server_ = std::make_shared<HttpServer>(ip_, port_, kThreadNum);
  }

  ~HttpCommunicator() = default;

  bool Start() override;
  bool Stop() override;
  void RegisterMsgCallBack(const std::string &msg_type, const MessageCallback &cb) override;

 private:
  std::shared_ptr<TaskExecutor> task_executor_;
  std::shared_ptr<HttpServer> http_server_;
  std::unordered_map<std::string, HttpMsgCallback> http_msg_callbacks_;

  std::string ip_;
  uint16_t port_;
};
}  // namespace core
}  // namespace ps
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PS_CORE_COMMUNICATOR_HTTP_COMMUNICATOR_H_
