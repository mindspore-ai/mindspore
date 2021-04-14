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

#ifndef MINDSPORE_CCSRC_PS_CORE_TCP_MESSAGE_HANDLER_H_
#define MINDSPORE_CCSRC_PS_CORE_TCP_MESSAGE_HANDLER_H_

#include <functional>
#include <iostream>
#include <string>
#include <memory>
#include <vector>

#include "utils/log_adapter.h"
#include "ps/core/message.h"
#include "proto/comm.pb.h"
#include "proto/ps.pb.h"

namespace mindspore {
namespace ps {
namespace core {
using messageReceive = std::function<void(std::shared_ptr<MessageMeta>, const Protos &, const void *, size_t size)>;
constexpr int kHeaderLen = 16;

class TcpMessageHandler {
 public:
  TcpMessageHandler()
      : is_parsed_(false), message_buffer_(nullptr), remaining_length_(0), header_index_(-1), last_copy_len_(0) {}
  virtual ~TcpMessageHandler() = default;

  void SetCallback(const messageReceive &cb);
  void ReceiveMessage(const void *buffer, size_t num);

 private:
  messageReceive message_callback_;
  bool is_parsed_;
  std::unique_ptr<unsigned char> message_buffer_;
  size_t remaining_length_;
  char header_[16]{0};
  int header_index_;
  size_t last_copy_len_;
  MessageHeader message_header_;
  std::string mBuffer;
};
}  // namespace core
}  // namespace ps
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PS_CORE_TCP_MESSAGE_HANDLER_H_
