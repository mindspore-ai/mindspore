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

#include "ps/core/communicator/http_msg_handler.h"
#include <memory>

namespace mindspore {
namespace ps {
namespace core {
HttpMsgHandler::HttpMsgHandler(const std::shared_ptr<HttpMessageHandler> &http_msg)
    : http_msg_(http_msg), data_(nullptr), len_(0) {
  if (http_msg != nullptr) {
    len_ = http_msg_->GetPostMsg(&data_);
  }
}

void *HttpMsgHandler::data() const {
  MS_ERROR_IF_NULL_W_RET_VAL(data_, nullptr);
  return data_;
}

size_t HttpMsgHandler::len() const { return len_; }

bool HttpMsgHandler::SendResponse(const void *data, const size_t &len) {
  MS_ERROR_IF_NULL_W_RET_VAL(data, false);
  http_msg_->QuickResponse(kHttpSuccess, reinterpret_cast<unsigned char *>(const_cast<void *>(data)), len);
  return true;
}
}  // namespace core
}  // namespace ps
}  // namespace mindspore
