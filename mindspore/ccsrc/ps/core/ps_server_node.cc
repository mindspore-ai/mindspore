/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "ps/core/ps_server_node.h"

namespace mindspore {
namespace ps {
namespace core {
void PSServerNode::Register(const std::shared_ptr<TcpClient> &client) {
  MS_EXCEPTION_IF_NULL(client);
  auto message_meta = std::make_shared<MessageMeta>();
  MS_EXCEPTION_IF_NULL(message_meta);
  message_meta->set_cmd(NodeCommand::REGISTER);
  message_meta->set_rank_id(node_info_.rank_id_);

  RegisterMessage register_message;
  register_message.set_node_id(node_info_.node_id_);
  register_message.set_role(node_info_.node_role_);
  register_message.set_ip(node_info_.ip_);
  register_message.set_port(node_info_.port_);

  MS_LOG(INFO) << "The node role:" << CommUtil::NodeRoleToString(node_info_.node_role_)
               << " the node id:" << node_info_.node_id_ << " begin to register to the scheduler!";

  if (!SendMessageSync(client, message_meta, Protos::PROTOBUF, register_message.SerializeAsString().data(),
                       register_message.ByteSizeLong())) {
    MS_LOG(EXCEPTION) << "The node role:" << CommUtil::NodeRoleToString(node_info_.node_role_)
                      << " the node id:" << node_info_.node_id_ << " register timeout!";
  } else {
    MS_LOG(INFO) << "The node role:" << CommUtil::NodeRoleToString(node_info_.node_role_)
                 << " the node id:" << node_info_.node_id_ << " send register success!";
  }

  // Registrations of alive nodes or registrations request exceeding the set total number of nodes should be
  // rejected, and the process exits after being rejected.
  if (node_info_.rank_id_ == UINT32_MAX) {
    MS_LOG(EXCEPTION) << "Register is rejected, and finish the node.";
  }
}
}  // namespace core
}  // namespace ps
}  // namespace mindspore
