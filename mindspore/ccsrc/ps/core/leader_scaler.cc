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

#include "ps/core/leader_scaler.h"

namespace mindspore {
namespace ps {
namespace core {
void LeaderScaler::ScaleOutAsync(const std::shared_ptr<TcpClient> &client, const NodeManager &manager) {
  MS_EXCEPTION_IF_NULL(client);
  MS_EXCEPTION_IF_NULL(node_);
  auto message_meta = std::make_shared<MessageMeta>();
  MS_EXCEPTION_IF_NULL(message_meta);
  message_meta->set_cmd(NodeCommand::SCALE_OUT);

  ScaleOutMessage scale_out_message;
  scale_out_message.set_worker_num(manager.worker_num());
  scale_out_message.set_server_num(manager.server_num());

  if (!node_->SendMessageSync(client, message_meta, Protos::PROTOBUF, scale_out_message.SerializeAsString().data(),
                              scale_out_message.ByteSizeLong())) {
    MS_LOG(WARNING) << "Send scale out timeout!";
  }

  MS_LOG(INFO) << "The scheduler is sending scale out to workers and servers!";
}

void LeaderScaler::ScaleInAsync(const std::shared_ptr<TcpClient> &client, const NodeManager &manager,
                                bool is_node_scale_in) {
  MS_EXCEPTION_IF_NULL(client);
  MS_EXCEPTION_IF_NULL(node_);
  auto message_meta = std::make_shared<MessageMeta>();
  MS_EXCEPTION_IF_NULL(message_meta);
  message_meta->set_cmd(NodeCommand::SCALE_IN);

  ScaleInMessage scale_in_message;
  scale_in_message.set_worker_num(manager.worker_num());
  scale_in_message.set_server_num(manager.server_num());
  scale_in_message.set_is_node_scale_in(is_node_scale_in);

  if (!node_->SendMessageSync(client, message_meta, Protos::PROTOBUF, scale_in_message.SerializeAsString().data(),
                              scale_in_message.ByteSizeLong())) {
    MS_LOG(WARNING) << "Send scale in timeout!";
  }

  MS_LOG(INFO) << "The scheduler is sending scale in to workers and servers!";
}
}  // namespace core
}  // namespace ps
}  // namespace mindspore
