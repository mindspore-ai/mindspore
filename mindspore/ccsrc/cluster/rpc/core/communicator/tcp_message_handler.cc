/**
 * Copyright 2020-2025 Huawei Technologies Co., Ltd
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

#include "cluster/rpc/core/communicator/tcp_message_handler.h"

#include <arpa/inet.h>
#include <iostream>
#include <utility>
#include <memory>
#include "include/utils/utils.h"
#include "utils/convert_utils_base.h"

namespace mindspore {
namespace ps {
namespace core {
void TcpMessageHandler::SetCallback(const messageReceive &message_receive) { message_callback_ = message_receive; }

void TcpMessageHandler::ReceiveMessage(const void *buffer, size_t num) {
  MS_EXCEPTION_IF_NULL(buffer);
  auto buffer_data = reinterpret_cast<const uint8_t *>(buffer);

  while (num > 0) {
    if (remaining_length_ == 0) {
      for (size_t i = 0; cur_header_len_ < kHeaderLen && num > 0; ++i) {
        header_[cur_header_len_] = buffer_data[i];
        cur_header_len_ += 1;
        --num;
        if (cur_header_len_ == kHeaderLen) {
          message_header_.message_proto_ = *reinterpret_cast<const Protos *>(header_);
          if (message_header_.message_proto_ != Protos::RAW && message_header_.message_proto_ != Protos::FLATBUFFERS &&
              message_header_.message_proto_ != Protos::PROTOBUF) {
            MS_LOG(WARNING)
              << "The proto:" << message_header_.message_proto_
              << " is illegal! Failed to parse the message header and the message body will fail to be received.";
            Reset();
            return;
          }
          message_header_.message_meta_length_ =
            *reinterpret_cast<const uint32_t *>(header_ + sizeof(message_header_.message_proto_));
          message_header_.message_length_ = *reinterpret_cast<const uint64_t *>(
            header_ + sizeof(message_header_.message_proto_) + sizeof(message_header_.message_meta_length_));
          if (message_header_.message_length_ >= UINT64_MAX) {
            MS_LOG(WARNING)
              << "The message len:" << message_header_.message_length_
              << " is too long.! Failed to parse the message header and the message body will fail to be received.";
            Reset();
            return;
          }
          if (message_header_.message_meta_length_ > message_header_.message_length_) {
            MS_LOG(WARNING) << "The message meta len " << message_header_.message_meta_length_ << " > the message len "
                            << message_header_.message_length_
                            << "! Failed to parse the message header and the message body will fail to be received.";
            Reset();
            return;
          }
          remaining_length_ = message_header_.message_length_;
          message_buffer_.resize(remaining_length_);
          buffer_data += (i + 1);
          break;
        }
      }
    }

    if (remaining_length_ > 0 && num > 0) {
      size_t copy_len = remaining_length_ <= num ? remaining_length_ : num;
      remaining_length_ -= copy_len;
      num -= copy_len;

      size_t dest_size = message_buffer_.size() - last_copy_len_;
      size_t src_size = copy_len;
      auto ret = Memcpy(message_buffer_.data() + last_copy_len_, dest_size, buffer_data, src_size);
      last_copy_len_ += copy_len;
      buffer_data += copy_len;
      if (ret != EOK) {
        MS_LOG(EXCEPTION) << "The Memcpy error, errorno(" << ret << ")";
      }

      if (remaining_length_ == 0) {
        if (message_callback_) {
          std::shared_ptr<MessageMeta> pb_message = std::make_shared<MessageMeta>();
          MS_EXCEPTION_IF_NULL(pb_message);
          if (!pb_message->ParseFromArray(message_buffer_.data(), UintToInt(message_header_.message_meta_length_))) {
            MS_LOG(ERROR) << "Parse protobuf MessageMeta failed! Failed to receive the message body.";
            Reset();
            return;
          }
          message_callback_(pb_message, message_header_.message_proto_,
                            message_buffer_.data() + message_header_.message_meta_length_,
                            message_header_.message_length_ - message_header_.message_meta_length_);
        }
        Reset();
      }
    }
  }
}

void TcpMessageHandler::Reset() {
  message_buffer_.clear();
  cur_header_len_ = 0;
  last_copy_len_ = 0;
  remaining_length_ = 0;
}
}  // namespace core
}  // namespace ps
}  // namespace mindspore
