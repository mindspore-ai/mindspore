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

#include <functional>
#include <algorithm>
#include <string>
#include "proto/topology.pb.h"
#include "distributed/rpc/tcp/constants.h"
#include "distributed/cluster/topology/utils.h"
#include "distributed/cluster/topology/meta_server_node.h"

namespace mindspore {
namespace distributed {
namespace cluster {
namespace topology {
bool MetaServerNode::Initialize() {
  // Init the address of meta server node.
  RETURN_IF_FALSE_WITH_LOG(FillMetaServerAddress(&meta_server_addr_),
                           "Failed to init the address of meta server node.");

  // Init the TCP server.
  RETURN_IF_FALSE_WITH_LOG(InitTCPServer(), "Failed to create the TCP server.");

  start_time_ = Now();

  // Init the thread for monitoring the state of the cluster topo.
  topo_monitor_ = std::thread(&MetaServerNode::UpdateTopoState, this);
  return true;
}

bool MetaServerNode::Finalize() {
  // Release the TCP server.
  if (tcp_server_ != nullptr) {
    tcp_server_->Finalize();
    tcp_server_.reset();
  }

  // Stop the topo monitor thread.
  enable_monitor_ = false;
  topo_monitor_.join();
  return true;
}

bool MetaServerNode::InitTCPServer() {
  tcp_server_ = std::make_unique<rpc::TCPServer>();
  MS_EXCEPTION_IF_NULL(tcp_server_);
  RETURN_IF_FALSE_WITH_LOG(tcp_server_->Initialize(meta_server_addr_.GetUrl()), "Failed to init the tcp server.");
  tcp_server_->SetMessageHandler(std::bind(&MetaServerNode::HandleMessage, this, std::placeholders::_1));

  // Configure the message processors for the TCP server.
  system_msg_handlers_[MessageName::kRegistration] =
    std::bind(&MetaServerNode::ProcessRegister, this, std::placeholders::_1);
  system_msg_handlers_[MessageName::kUnregistration] =
    std::bind(&MetaServerNode::ProcessUnregister, this, std::placeholders::_1);
  system_msg_handlers_[MessageName::kHeartbeat] =
    std::bind(&MetaServerNode::ProcessHeartbeat, this, std::placeholders::_1);
  return true;
}

MessageBase *const MetaServerNode::HandleMessage(MessageBase *const message) {
  const auto &name = message->Name();

  // Handle system messages.
  if (std::all_of(name.begin(), name.end(), ::isdigit)) {
    const auto &message_name = static_cast<MessageName>(std::stoi(message->Name()));
    const auto &handler = system_msg_handlers_.find(message_name);
    if (handler == system_msg_handlers_.end()) {
      MS_LOG(ERROR) << "Unknown system message name: " << message->Name();
      return rpc::NULL_MSG;
    }
    system_msg_handlers_[message_name](message);
    return rpc::NULL_MSG;

    // Handle user defined messages.
  } else {
    const auto &handler = message_handlers_.find(name);
    if (handler == message_handlers_.end()) {
      MS_LOG(ERROR) << "Unknown message name: " << name;
      return rpc::NULL_MSG;
    }
    (*message_handlers_[name])(message->Body());
    return rpc::NULL_MSG;
  }
}

MessageBase *const MetaServerNode::ProcessRegister(MessageBase *const message) {
  RegistrationMessage registration;
  const std::string &body = message->Body();
  registration.ParseFromArray(body.c_str(), body.length());

  // Add the compute graph node into registered nodes.
  const auto &node_id = registration.node_id();
  std::unique_lock<std::shared_mutex> lock(nodes_mutex_);
  if (nodes_.find(node_id) == nodes_.end()) {
    std::shared_ptr<ComputeGraphNodeState> node_state = std::make_shared<ComputeGraphNodeState>(node_id);
    nodes_[node_id] = node_state;
    MS_LOG(INFO) << "The new node: " << node_id << " is registered successfully.";
  } else {
    MS_LOG(ERROR) << "The node: " << node_id << " have been registered before.";
  }
  return rpc::NULL_MSG;
}

MessageBase *const MetaServerNode::ProcessUnregister(MessageBase *const message) {
  UnregistrationMessage unregistration;
  const std::string &body = message->Body();
  unregistration.ParseFromArray(body.c_str(), body.length());

  const auto &node_id = unregistration.node_id();
  std::unique_lock<std::shared_mutex> lock(nodes_mutex_);
  if (nodes_.find(node_id) == nodes_.end()) {
    MS_LOG(ERROR) << "Received unregistration message from invalid compute graph node: " << node_id;
    return rpc::NULL_MSG;
  }
  nodes_.erase(node_id);
  return rpc::NULL_MSG;
}

MessageBase *const MetaServerNode::ProcessHeartbeat(MessageBase *const message) {
  HeartbeatMessage heartbeat;
  const std::string &body = message->Body();
  heartbeat.ParseFromArray(body.c_str(), body.length());

  // Update the state(timestamp) of this node.
  const auto &node_id = heartbeat.node_id();
  std::shared_lock<std::shared_mutex> lock(nodes_mutex_);
  if (nodes_.find(node_id) != nodes_.end()) {
    auto &node = nodes_[node_id];
    time(&(node->last_update));
  } else {
    MS_LOG(ERROR) << "Invalid node: " << node_id << ".";
  }
  return rpc::NULL_MSG;
}

void MetaServerNode::UpdateTopoState() {
  while (enable_monitor_) {
    if (topo_state_ == TopoState::kInitializing) {
      // Set the state of topo to `kFailed` if the topology is still in process of initializtion but timed out.
      if (ElapsedTime(start_time_) > kTopoInitTimeout) {
        MS_LOG(ERROR) << "Failed to initialize the cluster topology after waiting for " << kTopoInitTimeout.count()
                      << " milliseconds.";
        topo_state_ = TopoState::kFailed;
        continue;
      }

      std::shared_lock<std::shared_mutex> lock(nodes_mutex_);
      if (nodes_.size() == total_node_num_) {
        MS_LOG(INFO) << "The cluster topology has been constructed successfully";
        topo_state_ = TopoState::kInitialized;
        continue;
      }
      MS_LOG(INFO) << "The cluster topology is in the process of constructing, current alive node num: ("
                   << nodes_.size() << "/" << total_node_num_ << ")";
    }
    static const size_t interval = 3;
    sleep(interval);
  }
}

TopoState MetaServerNode::TopologyState() { return topo_state_; }

size_t MetaServerNode::GetAliveNodeNum() {
  std::shared_lock<std::shared_mutex> lock(nodes_mutex_);
  return nodes_.size();
}

bool MetaServerNode::RegisterMessageHandler(const std::string &name,
                                            std::shared_ptr<std::function<std::string(const std::string &)>> handler) {
  if (message_handlers_.find(name) != message_handlers_.end()) {
    MS_LOG(ERROR) << "The message name: " << name << " have already been registered";
    return false;
  }
  message_handlers_[name] = handler;
  return true;
}
}  // namespace topology
}  // namespace cluster
}  // namespace distributed
}  // namespace mindspore
