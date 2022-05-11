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

#include <utility>
#include "utils/log_adapter.h"
#include "distributed/cluster/topology/utils.h"
#include "distributed/cluster/topology/common.h"
#include "proto/topology.pb.h"
#include "distributed/cluster/topology/compute_graph_node.h"

namespace mindspore {
namespace distributed {
namespace cluster {
namespace topology {
bool ComputeGraphNode::Initialize() {
  // Init the address of meta server node.
  RETURN_IF_FALSE_WITH_LOG(FillMetaServerAddress(&meta_server_addr_),
                           "Failed to init the address of meta server node.");

  // Init the TCP client.
  tcp_client_ = std::make_unique<rpc::TCPClient>();
  MS_EXCEPTION_IF_NULL(tcp_client_);
  RETURN_IF_FALSE_WITH_LOG(tcp_client_->Initialize(), "Failed to create the TCP client.");

  // Register itself to meta server node.
  EXECUTE_WITH_RETRY(Register, kExecuteRetryNum, kExecuteInterval, "Failed to register");
  return true;
}

bool ComputeGraphNode::Initialized() { return authenticated_; }

bool ComputeGraphNode::Finalize(bool force) {
  // Exit the compute graph node from the cluster topology.
  EXECUTE_WITH_RETRY(Unregister, kExecuteRetryNum, kExecuteInterval, "Failed to unregister");

  // Release the TCP client.
  if (tcp_client_ != nullptr) {
    const auto &server_url = meta_server_addr_.GetUrl();
    tcp_client_->Disconnect(server_url);
    tcp_client_->Finalize();
    tcp_client_.reset();
  }
  return true;
}

bool ComputeGraphNode::Register() {
  MS_EXCEPTION_IF_NULL(tcp_client_);
  const auto &server_url = meta_server_addr_.GetUrl();
  RETURN_IF_FALSE_WITH_LOG(tcp_client_->Disconnect(server_url),
                           "Failed to disconnect from the meta server node url: " << server_url);
  RETURN_IF_FALSE_WITH_LOG(tcp_client_->Connect(server_url),
                           "Failed to connect to the meta server node url: " << server_url);
  RegistrationMessage reg_msg;
  reg_msg.set_node_id(node_id_);

  std::string content = reg_msg.SerializeAsString();
  auto message = CreateMessage(server_url, MessageName::kRegistration, content);
  MS_EXCEPTION_IF_NULL(message);

  MessageBase *response = tcp_client_->ReceiveSync(std::move(message));
  if (response == nullptr) {
    return false;
  }
  auto body = response->body;
  delete response;
  response = nullptr;

  RegistrationRespMessage reg_resp_msg;
  reg_resp_msg.ParseFromArray(body.c_str(), body.length());

  if (reg_resp_msg.success()) {
    authenticated_ = true;
    rank_id_ = reg_resp_msg.rank_id();
    return true;
  } else {
    return false;
  }
}

bool ComputeGraphNode::Unregister() {
  MS_EXCEPTION_IF_NULL(tcp_client_);

  UnregistrationMessage unreg_msg;
  unreg_msg.set_node_id(node_id_);

  std::string content = unreg_msg.SerializeAsString();
  auto message = CreateMessage(meta_server_addr_.GetUrl(), MessageName::kUnregistration, content);
  MS_EXCEPTION_IF_NULL(message);

  MessageBase *response = tcp_client_->ReceiveSync(std::move(message));
  if (response == nullptr) {
    return false;
  }
  auto unreg_rt = response->body;
  delete response;
  response = nullptr;

  if (std::to_string(static_cast<int>(MessageName::kSuccess)) == unreg_rt) {
    return true;
  } else {
    return false;
  }
}

bool ComputeGraphNode::Heartbeat() {
  MS_EXCEPTION_IF_NULL(tcp_client_);

  HeartbeatMessage hb_msg;
  hb_msg.set_node_id(node_id_);

  const auto &server_url = meta_server_addr_.GetUrl();
  std::string content = hb_msg.SerializeAsString();
  auto message = CreateMessage(server_url, MessageName::kHeartbeat, content);
  MS_EXCEPTION_IF_NULL(message);

  tcp_client_->SendSync(std::move(message));
  return true;
}

bool ComputeGraphNode::SendMessageToMSN(const std::string msg_name, const std::string &msg_body) {
  MS_EXCEPTION_IF_NULL(tcp_client_);

  auto message = CreateMessage(meta_server_addr_.GetUrl(), msg_name, msg_body);
  MS_EXCEPTION_IF_NULL(message);

  auto retval = tcp_client_->SendSync(std::move(message));
  if (retval > 0) {
    return true;
  } else {
    return false;
  }
}

std::shared_ptr<std::string> ComputeGraphNode::RetrieveMessageFromMSN(const std::string &msg_name, uint32_t timeout) {
  MS_EXCEPTION_IF_NULL(tcp_client_);

  auto message = CreateMessage(meta_server_addr_.GetUrl(), msg_name, msg_name);
  MS_EXCEPTION_IF_NULL(message);

  auto retval = tcp_client_->ReceiveSync(std::move(message), timeout);
  if (retval != rpc::NULL_MSG) {
    return std::make_shared<std::string>(retval->body);
  }
  return nullptr;
}

bool ComputeGraphNode::PutMetadata(const std::string &name, const std::string &value) {
  MetadataMessage metadata;
  metadata.set_name(name);
  metadata.set_value(value);
  return SendMessageToMSN(std::to_string(static_cast<int>(MessageName::kWriteMetadata)), metadata.SerializeAsString());
}

std::string ComputeGraphNode::GetMetadata(const std::string &name, uint32_t timeout) {
  MetadataMessage metadata;
  metadata.set_name(name);

  auto message = CreateMessage(meta_server_addr_.GetUrl(), std::to_string(static_cast<int>(MessageName::kReadMetadata)),
                               metadata.SerializeAsString());
  MS_EXCEPTION_IF_NULL(message);

  MS_EXCEPTION_IF_NULL(tcp_client_);
  auto retval = tcp_client_->ReceiveSync(std::move(message), timeout);
  if (retval != rpc::NULL_MSG && (retval->name == std::to_string(static_cast<int>(MessageName::kValidMetadata)))) {
    metadata.ParseFromArray(retval->body.c_str(), retval->body.length());
    return metadata.value();
  }
  return "";
}
}  // namespace topology
}  // namespace cluster
}  // namespace distributed
}  // namespace mindspore
