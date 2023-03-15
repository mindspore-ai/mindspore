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

#ifndef MINDSPORE_CCSRC_PS_CORE_COMMUNICATOR_TCP_MSG_HANDLER_H_
#define MINDSPORE_CCSRC_PS_CORE_COMMUNICATOR_TCP_MSG_HANDLER_H_

#include <memory>
#include "proto/ps.pb.h"
#include "ps/core/abstract_node.h"
#include "ps/core/communicator/message_handler.h"
#include "include/backend/distributed/ps/constants.h"

namespace mindspore {
namespace ps {
namespace core {
class TcpMsgHandler : public MessageHandler {
 public:
  TcpMsgHandler(AbstractNode *abstract_node, const std::shared_ptr<core::TcpConnection> &conn,
                const std::shared_ptr<MessageMeta> &meta, DataPtr data, size_t size);
  ~TcpMsgHandler() override = default;

  void *data() const override;
  size_t len() const override;
  bool SendResponse(const void *data, const size_t &len) override;

 private:
  AbstractNode *abstract_node_;
  std::shared_ptr<TcpConnection> tcp_conn_;
  // core::MessageMeta is used for server to get the user command and to find communication peer when responding.
  std::shared_ptr<MessageMeta> meta_;
  // We use data of shared_ptr array so that the raw pointer won't be released until the reference is 0.
  DataPtr data_ptr_;
  void *data_;
  size_t len_;
};
}  // namespace core
}  // namespace ps
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PS_CORE_COMMUNICATOR_TCP_MSG_HANDLER_H_
