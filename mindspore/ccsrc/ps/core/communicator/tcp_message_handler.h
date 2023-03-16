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

#ifndef MINDSPORE_CCSRC_PS_CORE_COMMUNICATOR_TCP_MESSAGE_HANDLER_H_
#define MINDSPORE_CCSRC_PS_CORE_COMMUNICATOR_TCP_MESSAGE_HANDLER_H_

#include <functional>
#include <iostream>
#include <string>
#include <memory>
#include <vector>

#include "utils/log_adapter.h"
#include "ps/core/communicator/message.h"
#include "proto/comm.pb.h"
#include "proto/ps.pb.h"
#include "utils/convert_utils_base.h"
#include "include/backend/distributed/ps/constants.h"

namespace mindspore {
namespace ps {
namespace core {
using messageReceive =
  std::function<void(const std::shared_ptr<MessageMeta> &, const Protos &, const void *, size_t size)>;

constexpr size_t kHeaderLen = sizeof(MessageHeader);

class TcpMessageHandler {
 public:
  TcpMessageHandler() : remaining_length_(0), cur_header_len_(0), last_copy_len_(0) {}
  virtual ~TcpMessageHandler() = default;

  void SetCallback(const messageReceive &cb);
  void ReceiveMessage(const void *buffer, size_t num);

  void Reset();

 private:
  messageReceive message_callback_;
  std::vector<uint8_t> message_buffer_;
  uint8_t header_[kHeaderLen]{0};
  size_t remaining_length_;
  size_t cur_header_len_ = 0;
  size_t last_copy_len_;
  MessageHeader message_header_;
};
}  // namespace core
}  // namespace ps
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PS_CORE_COMMUNICATOR_TCP_MESSAGE_HANDLER_H_
