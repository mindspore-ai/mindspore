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
#include "distributed/recovery/recovery_context.h"
#include "distributed/recovery/file_configuration.h"
#include "distributed/cluster/topology/meta_server_node.h"

namespace mindspore {
namespace distributed {
namespace cluster {
namespace topology {
// The keys for the persisted metadata of compute node states.
constexpr char kComputeNodeStates[] = "compute_node_states";
constexpr char kNodeId[] = "node_id";
constexpr char kRecoveryFileName[] = "recovery.dat";

bool MetaServerNode::Initialize() {
  // Init the address of meta server node.
  RETURN_IF_FALSE_WITH_LOG(FillMetaServerAddress(&meta_server_addr_),
                           "Failed to init the address of meta server node.");

  // Init the TCP server.
  RETURN_IF_FALSE_WITH_LOG(InitTCPServer(), "Failed to create the TCP server.");

  // The meta server node is restarted and the metadata of cluster needs to be recovered.
  if (recovery::IsEnableRecovery()) {
    RETURN_IF_FALSE_WITH_LOG(Recovery(), "Failed to recover from configuration.");
  }

  start_time_ = Now();

  // Init the thread for monitoring the state of the cluster topo.
  topo_monitor_ = std::thread(&MetaServerNode::UpdateTopoState, this);
  return true;
}

bool MetaServerNode::Initialized() {
  return topo_state_ == TopoState::kInitialized || topo_state_ == TopoState::kFinished;
}

bool MetaServerNode::Finalize(bool force) {
  if (topo_state_ != TopoState::kFinished && !force) {
    MS_LOG(WARNING) << "The meta server node can not be finalized because there are still " << nodes_.size()
                    << " alive nodes.";
    return false;
  } else {
    // Release the TCP server.
    if (tcp_server_ != nullptr) {
      tcp_server_->Finalize();
      tcp_server_.reset();
    }

    // Stop the topo monitor thread.
    enable_monitor_ = false;
    topo_monitor_.join();
    if (force) {
      MS_LOG(INFO) << "The meta server node is forced to finalized.";
    }
    return true;
  }
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
  system_msg_handlers_[MessageName::kWriteMetadata] =
    std::bind(&MetaServerNode::ProcessWriteMetadata, this, std::placeholders::_1);
  system_msg_handlers_[MessageName::kReadMetadata] =
    std::bind(&MetaServerNode::ProcessReadMetadata, this, std::placeholders::_1);
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
    return system_msg_handlers_[message_name](message);

    // Handle user defined messages.
  } else {
    const auto &handler = message_handlers_.find(name);
    if (handler == message_handlers_.end()) {
      MS_LOG(ERROR) << "Unknown message name: " << name;
      return rpc::NULL_MSG;
    }
    const auto &result = (*message_handlers_[name])(message->Body());
    if (result.length() > 0) {
      auto rt_msg = CreateMessage(meta_server_addr_.GetUrl(), name, result);
      MS_EXCEPTION_IF_NULL(rt_msg);
      return rt_msg.release();
    } else {
      return rpc::NULL_MSG;
    }
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
    std::shared_ptr<NodeInfo> node_info = std::make_shared<NodeInfo>(node_id);
    node_info->state = NodeState::kRegistered;
    time(&(node_info->last_update));
    nodes_[node_id] = node_info;
    MS_LOG(INFO) << "The new node: " << node_id << " is registered successfully.";
    TransitionToInitialized();

    RegistrationRespMessage reg_resp_msg;
    reg_resp_msg.set_success(true);
    reg_resp_msg.set_rank_id(++next_rank_id_);
    reg_resp_msg.set_node_num(total_node_num_);
    std::string content = reg_resp_msg.SerializeAsString();

    auto message = CreateMessage(meta_server_addr_.GetUrl(), MessageName::kSuccess, content);
    MS_EXCEPTION_IF_NULL(message);
    return message.release();
  } else {
    MS_LOG(ERROR) << "The node: " << node_id << " have been registered before.";

    RegistrationRespMessage reg_resp_msg;
    reg_resp_msg.set_success(true);
    std::string content = reg_resp_msg.SerializeAsString();

    auto response = CreateMessage(meta_server_addr_.GetUrl(), MessageName::kSuccess, content);
    MS_EXCEPTION_IF_NULL(response);
    return response.release();
  }
}

MessageBase *const MetaServerNode::ProcessUnregister(MessageBase *const message) {
  UnregistrationMessage unregistration;
  const std::string &body = message->Body();
  unregistration.ParseFromArray(body.c_str(), body.length());

  const auto &node_id = unregistration.node_id();

  if (topo_state_ != TopoState::kInitialized) {
    MS_LOG(ERROR) << "Unable to process unreg message from node " << node_id << " because the state of the topology is "
                  << topo_state_;
    auto response = CreateMessage(meta_server_addr_.GetUrl(), MessageName::kUninitTopo,
                                  std::to_string(static_cast<int>(MessageName::kUninitTopo)));
    MS_EXCEPTION_IF_NULL(response);
    return response.release();
  }

  std::unique_lock<std::shared_mutex> lock(nodes_mutex_);
  if (nodes_.find(node_id) == nodes_.end()) {
    MS_LOG(ERROR) << "Received unregistration message from invalid compute graph node: " << node_id;
    auto response = CreateMessage(meta_server_addr_.GetUrl(), MessageName::kInvalidNode,
                                  std::to_string(static_cast<int>(MessageName::kInvalidNode)));
    MS_EXCEPTION_IF_NULL(response);
    return response.release();
  }
  nodes_.erase(node_id);
  if (nodes_.size() == 0) {
    topo_state_ = TopoState::kFinished;
  }
  auto response = CreateMessage(meta_server_addr_.GetUrl(), MessageName::kSuccess,
                                std::to_string(static_cast<int>(MessageName::kSuccess)));
  MS_EXCEPTION_IF_NULL(response);
  return response.release();
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

    auto response = CreateMessage(meta_server_addr_.GetUrl(), MessageName::kSuccess,
                                  std::to_string(static_cast<int>(MessageName::kSuccess)));
    MS_EXCEPTION_IF_NULL(response);
    return response.release();
  } else {
    MS_LOG(ERROR) << "Invalid node: " << node_id << ".";
    return rpc::NULL_MSG;
  }
}

MessageBase *const MetaServerNode::ProcessWriteMetadata(MessageBase *const message) {
  const std::string &body = message->Body();
  MetadataMessage meta_msg;
  meta_msg.ParseFromArray(body.c_str(), body.length());
  if (meta_msg.name().length() == 0) {
    MS_LOG(ERROR) << "Empty metadata name.";
    return rpc::NULL_MSG;
  }
  std::shared_lock<std::shared_mutex> lock(meta_mutex_);
  metadata_[meta_msg.name()] = meta_msg.value();
  return rpc::NULL_MSG;
}

MessageBase *const MetaServerNode::ProcessReadMetadata(MessageBase *const message) {
  const std::string &body = message->Body();
  MetadataMessage meta_msg;
  meta_msg.ParseFromArray(body.c_str(), body.length());

  std::shared_lock<std::shared_mutex> lock(meta_mutex_);
  MessageName result;
  std::unique_ptr<MessageBase> response;

  if (metadata_.find(meta_msg.name()) == metadata_.end()) {
    result = MessageName::kInvalidMetadata;
  } else {
    result = MessageName::kValidMetadata;
    std::string meta_value = metadata_[meta_msg.name()];
    meta_msg.set_value(meta_value);
  }
  response = CreateMessage(meta_server_addr_.GetUrl(), result, meta_msg.SerializeAsString());
  MS_EXCEPTION_IF_NULL(response);
  return response.release();
}

void MetaServerNode::UpdateTopoState() {
  while (enable_monitor_) {
    nodes_mutex_.lock();

    // Update the state of topology.
    if (topo_state_ == TopoState::kInitializing) {
      // Set the state of topo to `kFailed` if the topology is still in process of initializtion but timed out.
      if (ElapsedTime(start_time_) > kTopoInitTimeout) {
        MS_LOG(ERROR) << "Failed to initialize the cluster topology after waiting for " << kTopoInitTimeout.count()
                      << " milliseconds.";
        topo_state_ = TopoState::kFailed;
        continue;
      }

      if (TransitionToInitialized()) {
        continue;
      }
      MS_LOG(INFO) << "The cluster topology is in the process of constructing, current alive node num: ("
                   << nodes_.size() << "/" << total_node_num_ << ")";
    } else if (topo_state_ == TopoState::kInitialized) {
      if (nodes_.size() == 0) {
        topo_state_ = TopoState::kFinished;
      }

      // Update the state of compute graph nodes.
      for (auto iter = nodes_.begin(); iter != nodes_.end(); ++iter) {
        auto node_id = iter->first;
        auto node_info = iter->second;
        MS_EXCEPTION_IF_NULL(node_info);
        time_t now = time(&now);
        auto elapsed = difftime(now, node_info->last_update);
        if (elapsed > node_timeout_) {
          MS_LOG(ERROR) << "The node: " << node_id << " is timed out.";
          node_info->state = NodeState::kTimeout;
        }
      }
    }
    nodes_mutex_.unlock();

    static const size_t interval = 3;
    sleep(interval);
  }
}

bool MetaServerNode::TransitionToInitialized() {
  if (nodes_.size() == total_node_num_) {
    // Persist the cluster metadata into storage through configuration.
    if (recovery::IsEnableRecovery() && configuration_->Empty()) {
      if (!Persist()) {
        MS_LOG(EXCEPTION) << "Failed to persist the metadata of the cluster.";
      }
    }
    topo_state_ = TopoState::kInitialized;
    MS_LOG(INFO) << "The cluster topology has been constructed successfully";
    return true;
  }
  return false;
}

bool MetaServerNode::Recovery() {
  std::shared_lock<std::shared_mutex> lock(nodes_mutex_);
  std::string recovery_path = recovery::RecoveryPath();
  configuration_ = std::make_unique<recovery::FileConfiguration>(recovery_path + "/" + kRecoveryFileName);

  RETURN_IF_FALSE_WITH_LOG(configuration_->Initialize(),
                           "Failed to initialize the recovery file configuration from file path: " << recovery_path);

  if (configuration_->Empty()) {
    MS_LOG(INFO) << "The meta server node is started for the first time.";
    return true;

    // The meta server node is restarted and the metadata of cluster needs to be recovered.
  } else {
    std::string states_key = kComputeNodeStates;
    RETURN_IF_FALSE_WITH_LOG(configuration_->Exists(states_key),
                             "Can not find the key " + states_key + " in configuration.");

    // Check the validation of the previous metadata.
    const auto &states = configuration_->Get(states_key, "");
    nlohmann::json node_states = nlohmann::json::parse(states);
    RETURN_IF_FALSE_WITH_LOG(node_states.size() == total_node_num_,
                             "Invalid number of node in configuration: " + std::to_string(node_states.size()) +
                               ", expected total number of node: " + std::to_string(total_node_num_));

    // Restore the nodes state.
    for (auto iter = node_states.begin(); iter != node_states.end(); ++iter) {
      const auto &node_id = iter.key();
      std::shared_ptr<NodeInfo> node_info = std::make_shared<NodeInfo>(node_id);
      time(&(node_info->last_update));
      node_info->state = NodeState::kRegistered;
      nodes_[node_id] = node_info;
    }

    if (nodes_.size() == total_node_num_) {
      topo_state_ = TopoState::kInitialized;
    }
  }
  return true;
}

bool MetaServerNode::Persist() {
  MS_EXCEPTION_IF_NULL(configuration_);
  if (total_node_num_ != nodes_.size()) {
    MS_LOG(ERROR) << "Invalid number of alive node: " << nodes_.size()
                  << ", the expected total number of node is: " << total_node_num_;
    return false;
  }

  // The thread safety of nodes_ visiting has been guarded by the caller.
  nlohmann::json node_states;
  for (auto iter = nodes_.begin(); iter != nodes_.end(); ++iter) {
    const auto &node_id = iter->first;
    nlohmann::json node_state;
    node_state[kNodeId] = node_id;
    node_states[node_id] = node_state.dump();
  }

  configuration_->Put(kComputeNodeStates, node_states.dump());
  configuration_->Flush();
  return true;
}

TopoState MetaServerNode::TopologyState() { return topo_state_; }

size_t MetaServerNode::GetAliveNodeNum() {
  std::shared_lock<std::shared_mutex> lock(nodes_mutex_);
  size_t count = 0;
  for (auto iter = nodes_.begin(); iter != nodes_.end(); ++iter) {
    auto node_info = iter->second;
    MS_EXCEPTION_IF_NULL(node_info);

    // Only the node which has been authenticated is alive.
    if (node_info->state == NodeState::kRegistered) {
      ++count;
    }
  }
  return count;
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
