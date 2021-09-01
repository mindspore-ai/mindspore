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

#include "ps/core/communicator/tcp_msg_handler.h"
#include <memory>

namespace mindspore {
namespace ps {
namespace core {
TcpMsgHandler::TcpMsgHandler(AbstractNode *const abstract_node, const std::shared_ptr<core::TcpConnection> &conn,
                             const std::shared_ptr<MessageMeta> &meta, const DataPtr &data, size_t size)
    : abstract_node_(abstract_node), tcp_conn_(conn), meta_(meta), data_ptr_(data), data_(nullptr), len_(size) {
  if (data_ptr_ != nullptr) {
    data_ = data_ptr_.get();
  }
}

void *TcpMsgHandler::data() const {
  MS_ERROR_IF_NULL_W_RET_VAL(data_, nullptr);
  return data_;
}

size_t TcpMsgHandler::len() const { return len_; }

bool TcpMsgHandler::SendResponse(const void *data, const size_t &len) {
  MS_ERROR_IF_NULL_W_RET_VAL(tcp_conn_, false);
  MS_ERROR_IF_NULL_W_RET_VAL(meta_, false);
  MS_ERROR_IF_NULL_W_RET_VAL(data, false);
  MS_ERROR_IF_NULL_W_RET_VAL(abstract_node_, false);
  abstract_node_->Response(tcp_conn_, meta_, const_cast<void *>(data), len);
  return true;
}
}  // namespace core
}  // namespace ps
}  // namespace mindspore
