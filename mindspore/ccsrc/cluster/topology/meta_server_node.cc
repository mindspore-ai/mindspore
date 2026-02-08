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

#include "cluster/topology/meta_server_node.h"
#include <functional>
#include <algorithm>
#include <string>
#include <vector>
#include <utility>
#include <unordered_map>
#include <fstream>
#include "nlohmann/json.hpp"
#include "utils/ms_context.h"
#include "utils/ms_exception.h"
#include "proto/topology.pb.h"
#include "include/cluster/topology/ps_context.h"
#include "cluster/rpc/tcp/constants.h"
#include "include/runtime/hardware_abstract/collective/collective_manager.h"
#include "utils/convert_utils_base.h"
#include "utils/file_utils.h"

namespace mindspore {
namespace distributed {
namespace cluster {
namespace topology {
// The keys for parsed information of rank table file.
constexpr char kRankTableServerList[] = "server_list";
constexpr char kRankTableClusterList[] = "cluster_list";
constexpr char kRankTableDevice[] = "device";
constexpr char kRankTablePodIp[] = "pod_ip";
constexpr char kRankTableRankId[] = "rank_id";

MetaServerNode::~MetaServerNode() {
  try {
    (void)Finalize(true);
  } catch (std::exception &) {
    MS_LOG(ERROR) << "Failed to finalize MetaServerNode.";
  }
}

bool MetaServerNode::Initialize() {
  // Init metadata for the cluster.
  SetMetaData();

  MS_LOG(WARNING) << "Begin to Initialize MetaServerNode.";
  if (address_id_.empty()) {
    MS_LOG(DEBUG) << "MetaServerNode address_id_ is None.";
    // Init the address of meta server node.
    RETURN_IF_FALSE_WITH_LOG(FillMetaServerAddress(&meta_server_addr_),
                             "Failed to init the address of meta server node.");
  } else {
    MS_LOG(DEBUG) << "Begin to parse address_id.";
    // Init the address of meta server node.
    RETURN_IF_FALSE_WITH_LOG(ParseIPPort(address_id_, &meta_server_addr_), "Failed to Parse IP or Port.");
  }

  // Init the TCP server.
  RETURN_IF_FALSE_WITH_LOG(InitTCPServer(), "Failed to create the TCP server.");

  start_time_ = Now();

  // Init the thread for monitoring the state of the cluster topo.
  topo_monitor_ = std::thread(&MetaServerNode::UpdateTopoState, this);
  MS_LOG(WARNING) << "Finish to Initialize MetaServerNode.";
  return true;
}

bool MetaServerNode::Initialized() {
  return topo_state_ == TopoState::kInitialized || topo_state_ == TopoState::kFinished;
}

bool MetaServerNode::Finalize(bool force) {
  if (finalized_) {
    return true;
  }
  if (topo_state_ != TopoState::kFinished && !force &&
      (enable_recovery_ || (abnormal_node_num_ == 0 && !enable_recovery_))) {
    MS_LOG(WARNING) << "Cluster currently has " << nodes_.size() << " alive nodes.";
    return false;
  } else {
    if (abnormal_node_num_ > 0) {
      MS_LOG(ERROR) << "There are " << abnormal_node_num_ << " abnormal compute graph nodes.";
    }

    // Release the TCP server.
    if (tcp_server_ != nullptr) {
      tcp_server_->Finalize();
      tcp_server_.reset();
    }

    // Stop the topo monitor thread.
    enable_monitor_ = false;
    if (topo_monitor_.joinable()) {
      topo_monitor_.join();
    }
    if (force) {
      MS_LOG(INFO) << "The meta server node is forced to finalized.";
    }
    finalized_ = true;
    MsException::Instance().CheckException();
    return true;
  }
}

void MetaServerNode::SetMetaData() {
  // The validation check happened in cluster_context.cc, so we don't validating in this method.
  if (!common::GetEnv(kEnvWorkerNum).empty()) {
    role_expect_num_[kEnvRoleOfWorker] = IntToUint(std::stoi(common::GetEnv(kEnvWorkerNum)));
  }
  if (!common::GetEnv(kEnvServerNum).empty()) {
    role_expect_num_[kEnvRoleOfServer] = IntToUint(std::stoi(common::GetEnv(kEnvServerNum)));
    role_expect_num_[kEnvRoleOfPServer] = IntToUint(std::stoi(common::GetEnv(kEnvServerNum)));
  }
}

bool MetaServerNode::InitTCPServer() {
  bool enable_ssl = ps::PSContext::instance()->enable_ssl();
  tcp_server_ = std::make_unique<rpc::TCPServer>(enable_ssl);
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
  system_msg_handlers_[MessageName::kDeleteMetadata] =
    std::bind(&MetaServerNode::ProcessDeleteMetadata, this, std::placeholders::_1);
  system_msg_handlers_[MessageName::kGetHostNames] =
    std::bind(&MetaServerNode::ProcessGetHostNames, this, std::placeholders::_1);
  system_msg_handlers_[MessageName::kAddMetadata] =
    std::bind(&MetaServerNode::ProcessAddMetadata, this, std::placeholders::_1);
  return true;
}

MessageBase *const MetaServerNode::HandleMessage(MessageBase *const message) {
  MS_ERROR_IF_NULL_W_RET_VAL(message, rpc::NULL_MSG);
  const auto &name = message->Name();

  // Handle system messages.
  if (std::all_of(name.begin(), name.end(), ::isdigit)) {
    const auto &message_name = static_cast<MessageName>(std::stoi(message->Name()));
    const auto &handler = system_msg_handlers_.find(message_name);
    if (handler == system_msg_handlers_.end()) {
      MS_LOG(ERROR) << "Unknown system message name: " << message->Name();
      delete message;
      return rpc::NULL_MSG;
    }
    auto ret_msg = system_msg_handlers_[message_name](message);
    delete message;
    return ret_msg;
  } else {
    // Handle user defined messages.
    const auto &handler = message_handlers_.find(name);
    if (handler == message_handlers_.end()) {
      MS_LOG(ERROR) << "Unknown message name: " << name;
      delete message;
      return rpc::NULL_MSG;
    }
    const auto &result = (*message_handlers_[name])(message->Body());
    if (result.length() > 0) {
      auto rt_msg = CreateMessage(meta_server_addr_.GetUrl(), name, result);
      delete message;
      MS_EXCEPTION_IF_NULL(rt_msg);
      return rt_msg.release();
    } else {
      delete message;
      return rpc::NULL_MSG;
    }
  }
}

MessageBase *const MetaServerNode::ProcessRegister(MessageBase *const message) {
  MS_ERROR_IF_NULL_W_RET_VAL(message, rpc::NULL_MSG);
  RegistrationMessage registration;
  const std::string &body = message->Body();
  (void)registration.ParseFromArray(body.c_str(), SizeToInt(body.length()));

  // Add the compute graph node into registered nodes.
  const auto &node_id = registration.node_id();
  const auto &host_name = registration.host_name();
  const auto &host_ip = registration.host_ip();
  const auto &role = registration.role();
  const auto &device_id = registration.device_id();
  std::unique_lock<std::shared_mutex> lock(nodes_mutex_);
  if (nodes_.find(node_id) == nodes_.end()) {
    uint32_t rank_id = 0;
    if (common::IsStrNumeric(node_id)) {
      // This means node id is not randomly generated. So directly convert to int.
      rank_id = static_cast<uint32_t>(std::atoi(node_id.c_str()));
    }
    if (role != kEnvRoleOfClient) {
      if (!common::IsStrNumeric(node_id)) {
        rank_id = AllocateRankId(role);
      }

      // Check validation of this registered node.
      std::string reject_reason = "";
      if (!CheckRankIdValidation(node_id, role, rank_id, host_ip, &reject_reason)) {
        RegistrationRespMessage reg_resp_msg;
        reg_resp_msg.set_success(false);
        reg_resp_msg.set_error_reason(reject_reason);
        auto response =
          CreateMessage(meta_server_addr_.GetUrl(), MessageName::kInvalidNode, reg_resp_msg.SerializeAsString());
        return response.release();
      }
    }
    std::shared_ptr<NodeInfo> node_info = std::make_shared<NodeInfo>(node_id);
    MS_ERROR_IF_NULL_W_RET_VAL(node_info, rpc::NULL_MSG);
    node_info->host_name = host_name;
    node_info->host_ip = host_ip;
    node_info->role = role;
    node_info->rank_id = rank_id;
    node_info->device_id = device_id;
    node_info->state = NodeState::kRegistered;
    (void)time(&(node_info->last_update));
    nodes_[node_id] = node_info;
    size_t nodes_size = nodes_.size();
    (void)TransitionToInitialized();
    lock.unlock();

    RegistrationRespMessage reg_resp_msg;
    reg_resp_msg.set_success(true);
    reg_resp_msg.set_rank_id(rank_id);
    reg_resp_msg.set_node_num(SizeToUint(total_node_num_));
    std::string content = reg_resp_msg.SerializeAsString();

    auto message = CreateMessage(meta_server_addr_.GetUrl(), MessageName::kSuccess, content);
    MS_EXCEPTION_IF_NULL(message);
    MS_LOG(WARNING) << "The new node: " << node_id << "(role: " << role << ")"
                    << ", rank id: " << rank_id << ", device id: " << node_info->device_id
                    << ", hostname: " << node_info->host_name << ", ip: " << host_ip
                    << " is registered successfully. Currently registered node number: " << nodes_size
                    << ", expected node number: " << total_node_num_;
    return message.release();
  } else {
    if ((role != kEnvRoleOfClient) && !enable_recovery_) {
      MS_LOG(WARNING) << "Node " << node_id << " registered repeatedly. It's host ip is " << host_ip
                      << ". Reject this node.";
      RegistrationRespMessage reg_resp_msg;
      reg_resp_msg.set_success(false);
      reg_resp_msg.set_error_reason(
        "Repeated registration node: " + node_id +
        " to the scheduler. Please check if there's another scheduler process with port:" +
        std::to_string(meta_server_addr_.port) +
        " still running, or this is an extra node for distributed job. You can run command: 'netstat -anp|grep " +
        std::to_string(meta_server_addr_.port) +
        "' to check residual scheduler process. If another residual scheduler's still running, please kill it or "
        "change '--master_port' to a unoccupied port number of 'msrun' command and "
        "retry.");
      auto response =
        CreateMessage(meta_server_addr_.GetUrl(), MessageName::kInvalidNode, reg_resp_msg.SerializeAsString());
      return response.release();
    }
    auto node_info = nodes_[node_id];
    MS_EXCEPTION_IF_NULL(node_info);
    node_info->host_ip = host_ip;
    MS_LOG(WARNING) << "The node: " << node_id << " have been recovered. IP address: " << host_ip
                    << ", rank id: " << node_info->rank_id;
    (void)metadata_.insert(std::make_pair(node_info->role + node_info->node_id, std::to_string(node_info->rank_id)));

    RegistrationRespMessage reg_resp_msg;
    reg_resp_msg.set_success(true);
    reg_resp_msg.set_rank_id(node_info->rank_id);
    std::string content = reg_resp_msg.SerializeAsString();

    auto response = CreateMessage(meta_server_addr_.GetUrl(), MessageName::kSuccess, content);
    MS_EXCEPTION_IF_NULL(response);
    return response.release();
  }
}

MessageBase *const MetaServerNode::ProcessUnregister(MessageBase *const message) {
  MS_ERROR_IF_NULL_W_RET_VAL(message, rpc::NULL_MSG);
  UnregistrationMessage unregistration;
  const std::string &body = message->Body();
  (void)unregistration.ParseFromArray(body.c_str(), SizeToInt(body.length()));

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
  (void)nodes_.erase(node_id);
  MS_LOG(WARNING) << "Node " << node_id << " has unregistered.";
  if (nodes_.size() == 0) {
    topo_state_ = TopoState::kFinished;
  }
  lock.unlock();
  auto response = CreateMessage(meta_server_addr_.GetUrl(), MessageName::kSuccess,
                                std::to_string(static_cast<int>(MessageName::kSuccess)));
  MS_EXCEPTION_IF_NULL(response);
  return response.release();
}

MessageBase *const MetaServerNode::ProcessHeartbeat(MessageBase *const message) {
  MS_ERROR_IF_NULL_W_RET_VAL(message, rpc::NULL_MSG);
  HeartbeatMessage heartbeat;
  const std::string &body = message->Body();
  (void)heartbeat.ParseFromArray(body.c_str(), SizeToInt(body.length()));

  // Update the state(timestamp) of this node.
  const auto &node_id = heartbeat.node_id();
  std::shared_lock<std::shared_mutex> lock(nodes_mutex_);
  if (nodes_.find(node_id) != nodes_.end()) {
    auto &node = nodes_.at(node_id);
    MS_ERROR_IF_NULL_W_RET_VAL(node, rpc::NULL_MSG);
    (void)time(&(node->last_update));
    node->state = NodeState::kRegistered;
    lock.unlock();

    HeartbeatRespMessage resp_msg;
    resp_msg.set_success(static_cast<bool>(MessageName::kSuccess));
    resp_msg.set_topo_state(static_cast<uint32_t>(topo_state_));
    resp_msg.set_nodes_num(SizeToUint(total_node_num_));
    resp_msg.set_abnormal_nodes_num(SizeToUint(abnormal_node_num_));
    auto content = resp_msg.SerializeAsString();
    auto response = CreateMessage(meta_server_addr_.GetUrl(), MessageName::kSuccess, content);
    MS_EXCEPTION_IF_NULL(response);
    return response.release();
  } else {
    MS_LOG(ERROR) << "Invalid node: " << node_id << ".";
    return rpc::NULL_MSG;
  }
}

MessageBase *const MetaServerNode::ProcessWriteMetadata(MessageBase *const message) {
  MS_ERROR_IF_NULL_W_RET_VAL(message, rpc::NULL_MSG);
  const std::string &body = message->Body();
  MetadataMessage meta_msg;
  (void)meta_msg.ParseFromArray(body.c_str(), SizeToInt(body.length()));
  if (meta_msg.name().length() == 0) {
    MS_LOG(ERROR) << "Empty metadata name.";
    return rpc::NULL_MSG;
  }
  std::shared_lock<std::shared_mutex> lock(meta_mutex_);
  metadata_[meta_msg.name()] = meta_msg.value();
  return rpc::NULL_MSG;
}

MessageBase *const MetaServerNode::ProcessReadMetadata(MessageBase *const message) {
  MS_ERROR_IF_NULL_W_RET_VAL(message, rpc::NULL_MSG);
  const std::string &body = message->Body();
  MetadataMessage meta_msg;
  (void)meta_msg.ParseFromArray(body.c_str(), SizeToInt(body.length()));

  std::shared_lock<std::shared_mutex> lock(meta_mutex_);
  MessageName result;
  std::unique_ptr<MessageBase> response;

  if (metadata_.find(meta_msg.name()) == metadata_.end()) {
    result = MessageName::kInvalidMetadata;
  } else {
    result = MessageName::kValidMetadata;
    std::string meta_value = metadata_.at(meta_msg.name());
    meta_msg.set_value(meta_value);
  }
  response = CreateMessage(meta_server_addr_.GetUrl(), result, meta_msg.SerializeAsString());
  MS_EXCEPTION_IF_NULL(response);
  return response.release();
}

MessageBase *const MetaServerNode::ProcessDeleteMetadata(MessageBase *const message) {
  MS_ERROR_IF_NULL_W_RET_VAL(message, rpc::NULL_MSG);
  const std::string &body = message->Body();
  MetadataMessage meta_msg;
  (void)meta_msg.ParseFromArray(body.c_str(), SizeToInt(body.length()));

  std::shared_lock<std::shared_mutex> lock(meta_mutex_);
  MessageName result;
  std::unique_ptr<MessageBase> response;

  if (metadata_.find(meta_msg.name()) == metadata_.end()) {
    result = MessageName::kInvalidMetadata;
  } else {
    result = MessageName::kValidMetadata;
    (void)metadata_.erase(meta_msg.name());
  }
  response = CreateMessage(meta_server_addr_.GetUrl(), result, meta_msg.SerializeAsString());
  MS_EXCEPTION_IF_NULL(response);
  return response.release();
}

MessageBase *const MetaServerNode::ProcessAddMetadata(MessageBase *const message) {
  MS_ERROR_IF_NULL_W_RET_VAL(message, rpc::NULL_MSG);
  const std::string &body = message->Body();
  MetadataMessage meta_msg;
  (void)meta_msg.ParseFromArray(body.c_str(), SizeToInt(body.length()));

  std::shared_lock<std::shared_mutex> lock(meta_mutex_);
  MessageName result = MessageName::kValidMetadata;
  std::unique_ptr<MessageBase> response;
  int64_t addVal = 0;
  bool is_exception = false;
  try {
    addVal = std::stoll(meta_msg.value());
  } catch (const std::exception &e) {
    result = MessageName::kInvalidMetadata;
    is_exception = true;
    MS_LOG(ERROR) << "Add input value is illegal, input is " << meta_msg.value();
  }
  if (metadata_.find(meta_msg.name()) != metadata_.end()) {
    std::string meta_value = metadata_.at(meta_msg.name());
    try {
      addVal += std::stoll(meta_value);
    } catch (const std::exception &e) {
      result = MessageName::kInvalidMetadata;
      is_exception = true;
      MS_LOG(ERROR) << "The value corresponding to the key value of Add is illegal, value is " << meta_value;
    }
  }
  if (!is_exception) {
    metadata_[meta_msg.name()] = std::to_string(addVal);
  }
  meta_msg.set_value(std::to_string(addVal));
  response = CreateMessage(meta_server_addr_.GetUrl(), result, meta_msg.SerializeAsString());
  MS_EXCEPTION_IF_NULL(response);
  return response.release();
}

MessageBase *const MetaServerNode::ProcessGetHostNames(MessageBase *const message) {
  MS_ERROR_IF_NULL_W_RET_VAL(message, rpc::NULL_MSG);
  // Convert result to the message.
  nlohmann::json hostnames = nlohmann::json::array();
  nlohmann::json retval = nlohmann::json::object();
  MessageName result;
  auto node_role = message->body;

  // all_hostname_hash_.count(node_role) == 0 condition is to ensure some nodes' getting valid hostnames even if others
  // nodes have already unregistered.
  if (nodes_.size() != total_node_num_ && all_hostname_hash_.count(node_role) == 0) {
    result = MessageName::kInvalidMetadata;
    retval[kHostNames] = hostnames;
    auto response = CreateMessage(meta_server_addr_.GetUrl(), result, retval.dump());
    MS_EXCEPTION_IF_NULL(response);
    return response.release();
  } else if (all_hostname_hash_.count(node_role) == 0) {
    result = MessageName::kValidMetadata;

    // Collect all the hostnames from nodes info.
    std::vector<std::string> tmp_hostnames(nodes_.size(), "");
    std::shared_lock<std::shared_mutex> lock(nodes_mutex_);

    // The hostnames must are sorted strictly by the rank id.
    for (auto iter = nodes_.begin(); iter != nodes_.end(); ++iter) {
      auto node_info = iter->second;
      MS_EXCEPTION_IF_NULL(node_info);
      if (node_info->role != node_role) {
        continue;
      }
      if (node_info->rank_id >= 0 && node_info->rank_id < tmp_hostnames.size()) {
        tmp_hostnames[node_info->rank_id] = node_info->host_name;
      } else {
        MS_LOG(ERROR) << "Invalid rank id: " << node_info->rank_id << " for node: " << node_info->node_id;
        continue;
      }
    }
    lock.unlock();

    // The hostname of the node whose role name not match is empty, and should be skipped.
    for (size_t i = 0; i < tmp_hostnames.size(); ++i) {
      if (tmp_hostnames[i] != "") {
        hostnames.push_back(tmp_hostnames[i]);
      }
    }
    retval[kHostNames] = hostnames;
    all_hostname_hash_[node_role] = retval.dump();
  } else {
    result = MessageName::kValidMetadata;
  }

  auto response = CreateMessage(meta_server_addr_.GetUrl(), result, all_hostname_hash_[node_role]);
  MS_EXCEPTION_IF_NULL(response);
  return response.release();
}

void MetaServerNode::UpdateTopoState() {
  try {
    while (enable_monitor_) {
      nodes_mutex_.lock();

      // Update the state of topology.
      if (topo_state_ == TopoState::kInitializing) {
        if (TransitionToInitialized()) {
          nodes_mutex_.unlock();
          continue;
        }
        MS_LOG(INFO) << "The cluster topology is in the process of constructing, current alive node num: ("
                     << nodes_.size() << "/" << total_node_num_ << ")";
      } else if (topo_state_ == TopoState::kInitialized) {
        if (nodes_.size() == 0) {
          topo_state_ = TopoState::kFinished;
        }
      }

      if (!disable_heartbeat_) {
        // Update the state of compute graph nodes if heartbeat is enabled.
        size_t abnormal_node_num = 0;
        std::vector<std::string> time_out_node_ids = {};
        for (auto iter = nodes_.begin(); iter != nodes_.end(); ++iter) {
          auto node_id = iter->first;
          auto node_info = iter->second;
          MS_EXCEPTION_IF_NULL(node_info);
          time_t now = time(&now);
          auto elapsed = difftime(now, node_info->last_update);
          if (elapsed > node_timeout_) {
            node_info->state = NodeState::kTimeout;
            ++abnormal_node_num;
            time_out_node_ids.push_back(node_id);
            MS_LOG(ERROR) << "The node: " << node_id
                          << " is timed out. It may exit with exception, please check this node's log.";
          }
        }
        abnormal_node_num_ = abnormal_node_num;
        if (abnormal_node_num_ > 0 && !enable_recovery_) {
          MS_LOG(EXCEPTION) << "The total number of timed out node is " << abnormal_node_num_
                            << ". Timed out node list is: " << time_out_node_ids << ", worker " << time_out_node_ids[0]
                            << " is the first one timed out, please check its log.";
        }
      }

      nodes_mutex_.unlock();
      static const size_t interval = 3;
      SleepBasedOnScale(interval);
    }
  } catch (const std::exception &e) {
    nodes_mutex_.unlock();
    MsException::Instance().SetException();
  }
}

bool MetaServerNode::TransitionToInitialized() {
  if (nodes_.size() == total_node_num_) {
    // If env RANK_TABLE_FILE is set, reassign rank ids based on provided rank table file. Any irregular behavior of the
    // rank table file will make rank ids not be reassigned.
    if (!ReassignNodeRankFromRanktablefile()) {
      // After all nodes are successfully registered, reassign rank ids so they could be continuous.
      ReassignNodeRank();
    }

    topo_state_ = TopoState::kInitialized;
    MS_LOG(INFO) << "The cluster topology has been constructed successfully.";
    MS_VLOG(VL_DISTRIBUTED_TRACE) << "Distribute networking cost : " << ElapsedTime(start_time_).count() << " ms.";
    return true;
  }
  return false;
}

uint32_t MetaServerNode::AllocateRankId(const std::string &role) {
  std::shared_lock<std::shared_mutex> lock(rank_mutex_);
  if (role_expect_num_.find(role) == role_expect_num_.end()) {
    MS_LOG(WARNING) << "Role: " << role << " is invalid.";
    return UINT32_MAX;
  }
  if (next_rank_ids_.count(role) == 0) {
    next_rank_ids_[role] = 0;
  } else {
    // If this role's rank id has exceeded, do not increase next_rank_ids_ and return an exceeded rank id. The caller
    // will check rank id's validation and reject this request.
    if (next_rank_ids_[role] == role_expect_num_[role] - 1) {
      return next_rank_ids_[role] + 1;
    }
    next_rank_ids_[role] += 1;
  }
  return next_rank_ids_[role];
}

bool MetaServerNode::CheckRankIdValidation(const std::string &node_id, const std::string &role, uint32_t rank_id,
                                           const std::string &host_ip, std::string *reject_reason) {
  if (role_expect_num_.find(role) == role_expect_num_.end()) {
    MS_LOG(WARNING) << "Registered node role: " << role << " is invalid.";
    return false;
  }
  // Whether rank id has already exists.
  bool rank_id_exist = false;
  if (rank_role_to_node_info_.find(std::make_pair(rank_id, role)) != rank_role_to_node_info_.end()) {
    rank_id_exist = true;
  }

  // Whether rank id exceeds upper bound.
  bool is_extra_node = (rank_id >= role_expect_num_[role]);
  if (rank_id_exist) {
    *reject_reason = "Rank id:" + std::to_string(rank_id) + " for role:" + role + " exists.";
  }
  if (is_extra_node) {
    *reject_reason = "This node is extra or rank id exceeds. Total node number for role " + role + " is " +
                     std::to_string(role_expect_num_[role]) + " but got rank id " + std::to_string(rank_id);
  }
  if (rank_id_exist || is_extra_node) {
    MS_LOG(WARNING) << "Rejecting registration request for node " << node_id << " from host " << host_ip
                    << ". Rejection reason: " << *reject_reason;
    return false;
  }
  return true;
}

bool MetaServerNode::ReassignNodeRankFromRanktablefile() {
  std::string rank_table_file_path = common::GetEnv("RANK_TABLE_FILE");
  if (rank_table_file_path.empty()) {
    return false;
  }

  MS_LOG(INFO) << "Start reassigning rank ids for nodes according to rank table file, json file path: "
               << rank_table_file_path;
  auto realpath = FileUtils::GetRealPath(rank_table_file_path.c_str());
  if (!realpath.has_value()) {
    MS_LOG(WARNING) << "Failed to get real path. Won't reassign rank id based on rank table file.";
    return false;
  }
  std::ifstream jsonFile(realpath.value(), std::ifstream::in);
  if (!jsonFile.is_open()) {
    MS_LOG(WARNING)
      << "Failed to open rank table file. Won't reassign rank id based on rank table file. This may be because the "
         "path of rank table file is incorrect or the access of json file is not permitted.";
    return false;
  }

  nlohmann::json rank_table_file_data;
  try {
    rank_table_file_data = nlohmann::json::parse(jsonFile);
    if (rank_table_file_data.is_null()) {
      MS_LOG(WARNING) << "Failed to read data from rank table file. Won't reassign rank id based on rank table file.";
      return false;
    }

    std::map<std::string, std::vector<std::string>> mapped_rank_id;
    for (const auto &server_list : rank_table_file_data[kRankTableServerList]) {
      if (server_list.find(kRankTablePodIp) == server_list.end()) {
        MS_LOG(WARNING) << "Cannot find key 'pod_ip' in 'server_list' from rank table file. Won't reassign rank id "
                           "based on rank table file.";
        return false;
      }
      std::string pod_ip = server_list[kRankTablePodIp];
      for (const auto &device : server_list[kRankTableDevice]) {
        std::string rank_id = device[kRankTableRankId];
        (void)mapped_rank_id[pod_ip].push_back(rank_id);
      }
    }

    std::map<std::string, uint32_t> mapped_local_rank_size;
    for (auto &n : nodes_) {
      std::shared_ptr<NodeInfo> &node_info = n.second;
      mapped_local_rank_size[node_info->host_ip] = mapped_local_rank_size[node_info->host_ip] + 1;
    }

    for (auto &n : nodes_) {
      std::shared_ptr<NodeInfo> &node_info = n.second;
      const std::string &role = node_info->role;
      uint32_t device_id = node_info->device_id;
      if (device_id == UINT32_MAX) {
        MS_LOG(WARNING) << "Device id is set incorrectly in the scenario where importing rank table file. Won't "
                           "reassign rank id based on rank table file.";
        return false;
      }
      if (mapped_rank_id.find(node_info->host_ip) == mapped_rank_id.end()) {
        MS_LOG(WARNING) << "Current node's HOST_IP cannot be found in rank table file. Won't reassign rank id based "
                           "on rank table file.";
        return false;
      }
      if (mapped_local_rank_size[node_info->host_ip] != mapped_rank_id[node_info->host_ip].size()) {
        MS_LOG(WARNING) << "Current node's DEVICE_ID [" << mapped_local_rank_size[node_info->host_ip]
                        << "] is not equal to total number of devices [" << mapped_rank_id[node_info->host_ip].size()
                        << "] in rank table file. Won't reassign rank id based on rank table file.";
        return false;
      }
      std::string new_rank = mapped_rank_id[node_info->host_ip][device_id];
      MS_LOG(WARNING) << "Reassign rank id from rank table file, node id: " << node_info->node_id << ", role: " << role
                      << ", with host ip: " << node_info->host_ip << ", device id: " << node_info->device_id
                      << ", old rank id: " << node_info->rank_id << ", new rank id: " << new_rank;
      node_info->rank_id = std::stoul(new_rank);
      (void)metadata_.insert(std::make_pair(role + node_info->node_id, std::to_string(node_info->rank_id)));
    }
  } catch (const std::exception &e) {
    MS_LOG(WARNING) << "Rank table file is incorrect. Won't reassign rank id based on rank table file. Json error: "
                    << e.what();
    return false;
  }
  return true;
}

void MetaServerNode::ReassignNodeRank() {
  if (std::all_of(nodes_.begin(), nodes_.end(), [](const auto &node) { return common::IsStrNumeric(node.first); })) {
    MS_LOG(WARNING) << "Rank ids are already set by numeric node ids. No need to reassign them.";
    for (const auto &n : nodes_) {
      const std::shared_ptr<NodeInfo> &node_info = n.second;
      const std::string &role = node_info->role;
      (void)metadata_.insert(std::make_pair(role + node_info->node_id, std::to_string(node_info->rank_id)));
    }
    return;
  }

  MS_LOG(INFO) << "Start sorting and reassigning rank ids for nodes according to node ips and node ids.";
  std::map<std::string, std::map<NodeKey, uint32_t>> node_ranks;
  for (auto &n : nodes_) {
    std::shared_ptr<NodeInfo> &node_info = n.second;
    NodeKey node_key = {node_info->host_ip, node_info->node_id};
    (void)node_ranks[node_info->role].insert(std::make_pair(node_key, 0));
  }

  for (auto &n : node_ranks) {
    std::map<NodeKey, uint32_t> &node_key_ranks = n.second;
    uint32_t accum_rank_id = 0;
    for (auto &node_rank : node_key_ranks) {
      node_rank.second = accum_rank_id++;
    }
  }

  for (auto &n : nodes_) {
    std::shared_ptr<NodeInfo> &node_info = n.second;
    const std::string &role = node_info->role;
    NodeKey node_key = {node_info->host_ip, node_info->node_id};
    uint32_t new_rank = node_ranks[role][node_key];

    MS_LOG(WARNING) << "Assign rank id of node id: " << node_info->node_id << ", role: " << role
                    << ", with host ip: " << node_info->host_ip << ", old rank id: " << node_info->rank_id
                    << ", new rank id: " << new_rank;

    node_info->rank_id = new_rank;
    (void)metadata_.insert(std::make_pair(role + node_info->node_id, std::to_string(node_info->rank_id)));
  }
}

TopoState MetaServerNode::TopologyState() const { return topo_state_; }

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

bool MetaServerNode::RegisterMessageHandler(
  const std::string &name, const std::shared_ptr<std::function<std::string(const std::string &)>> &handler) {
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
