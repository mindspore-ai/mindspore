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

#include "ps/comm/tcp_message_handler.h"
#include <iostream>
#include <utility>

namespace mindspore {
namespace ps {
namespace comm {

void TcpMessageHandler::SetCallback(messageReceive message_receive) { message_callback_ = std::move(message_receive); }

void TcpMessageHandler::ReceiveMessage(const void *buffer, size_t num) {
  MS_EXCEPTION_IF_NULL(buffer);

  if (message_callback_) {
    message_callback_(buffer, num);
  }
}
}  // namespace comm
}  // namespace ps
}  // namespace mindspore
