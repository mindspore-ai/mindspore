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

#ifndef MINDSPORE_CCSRC_PS_COMM_TCP_MESSAGE_HANDLER_H_
#define MINDSPORE_CCSRC_PS_COMM_TCP_MESSAGE_HANDLER_H_

#include <functional>
#include <iostream>
#include <memory>

#include "utils/log_adapter.h"

namespace mindspore {
namespace ps {
namespace comm {

using messageReceive = std::function<void(const void *buffer, size_t len)>;

class TcpMessageHandler {
 public:
  TcpMessageHandler() = default;
  virtual ~TcpMessageHandler() = default;

  void SetCallback(messageReceive cb);
  void ReceiveMessage(const void *buffer, size_t num);

 private:
  messageReceive message_callback_;
};
}  // namespace comm
}  // namespace ps
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PS_COMM_TCP_MESSAGE_HANDLER_H_
