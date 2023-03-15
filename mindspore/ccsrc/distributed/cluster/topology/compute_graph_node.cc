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
#include <nlohmann/json.hpp>
#include "utils/log_adapter.h"
#include "utils/ms_exception.h"
#include "include/backend/distributed/cluster/topology/common.h"
#include "include/backend/distributed/recovery/recovery_context.h"
#include "include/backend/distributed/constants.h"
#include "proto/topology.pb.h"
#include "ps/ps_context.h"
#include "include/backend/distributed/cluster/topology/compute_graph_node.h"
#include "include/backend/distributed/rpc/tcp/constants.h"

namespace mindspore {
namespace distributed {
namespace cluster {
namespace topology {
constexpr char kStartExchangeMetaPrefix[] = "START_EXCHANGE_META_";
constexpr char kExchangeMetaDonePrefix[] = "EXCHANGE_META_DONE_";
constexpr char kMetaFlagValue[] = "1";
constexpr char kMetaDeleteFlagValue[] = "";

ComputeGraphNode::~ComputeGraphNode() {
  if (!finalized_) {
    try {
      (void)Finalize(true);
    } catch (std::exception &) {
      MS_LOG(ERROR) << "Failed to finalize ComputeGraphNode.";
    }
  }
}

bool ComputeGraphNode::Initialize() {
  // Init the address of meta server node.
  RETURN_IF_FALSE_WITH_LOG(FillMetaServerAddress(&meta_server_addr_),
                           "Failed to init the address of meta server node.");

  // Init the TCP client.
  bool enable_ssl = ps::PSContext::instance()->enable_ssl();
  tcp_client_ = std::make_unique<rpc::TCPClient>(enable_ssl);
  MS_EXCEPTION_IF_NULL(tcp_client_);
  RETURN_IF_FALSE_WITH_LOG(tcp_client_->Initialize(), "Failed to create the TCP client.");

  hb_client_ = std::make_unique<rpc::TCPClient>(enable_ssl);
  MS_EXCEPTION_IF_NULL(hb_client_);
  RETURN_IF_FALSE_WITH_LOG(hb_client_->Initialize(), "Failed to create the heartbeat tcp client.");

  // Register itself to meta server node.
  bool success = false;
  if (!enable_ssl) {
    success = ReconnectIfNeeded(std::bind(&ComputeGraphNode::Register, this),
                                "Failed to register and try to reconnect to the meta server.", kCgnExecuteRetryNum);
  } else {
    const auto &server_url = meta_server_addr_.GetUrl();
    size_t retry = 10;
    while (!success && retry-- > 0) {
      success = Register();
      if (success) {
        break;
      }

      if (tcp_client_ != nullptr) {
        (void)tcp_client_->Disconnect(server_url);
        tcp_client_->Finalize();
        tcp_client_.reset();
      }
      if (hb_client_ != nullptr) {
        (void)hb_client_->Disconnect(server_url);
        hb_client_->Finalize();
        hb_client_.reset();
      }

      tcp_client_ = std::make_unique<rpc::TCPClient>(enable_ssl);
      MS_EXCEPTION_IF_NULL(tcp_client_);
      RETURN_IF_FALSE_WITH_LOG(tcp_client_->Initialize(), "Failed to create the TCP client.");

      hb_client_ = std::make_unique<rpc::TCPClient>(enable_ssl);
      MS_EXCEPTION_IF_NULL(hb_client_);
      RETURN_IF_FALSE_WITH_LOG(hb_client_->Initialize(), "Failed to create the heartbeat tcp client.");
    }
  }
  if (!success) {
    return false;
  }

  // Enable the heartbeat to meta server node.
  enable_hb_ = true;
  heartbeat_ = std::thread(&ComputeGraphNode::Heartbeat, this);
  return true;
}

bool ComputeGraphNode::Initialized() {
  // The cgn is initialized only when the cluster is ready, or there will be error message unexpected.
  return authenticated_ && topo_state_ == TopoState::kInitialized;
}

bool ComputeGraphNode::Finalize(bool force) {
  // Stop the heartbeat thread.
  enable_hb_ = false;
  if (heartbeat_.joinable()) {
    heartbeat_.join();
  }

  // Exit the compute graph node from the cluster topology.
  while (!force) {
    bool success = ReconnectIfNeeded(std::bind(&ComputeGraphNode::Unregister, this),
                                     "Failed to unregister and try to reconnect to the meta server.", kNoRetry);
    if (!success) {
      MS_LOG(ERROR) << "Failed to unregister from the meta server node.";
      if (recovery::IsEnableRecovery()) {
        continue;
      } else {
        break;
      }
    } else {
      MS_LOG(INFO) << "The compute graph node has been unregistered successfully.";
      break;
    }
  }

  // Release the TCP client.
  bool enable_ssl = ps::PSContext::instance()->enable_ssl();
  const auto &server_url = meta_server_addr_.GetUrl();
  if (tcp_client_ != nullptr) {
    if (!(enable_ssl && !authenticated_)) {
      (void)tcp_client_->Disconnect(server_url);
    }
    tcp_client_->Finalize();
    tcp_client_.reset();
  }

  if (hb_client_ != nullptr) {
    if (!(enable_ssl && !authenticated_)) {
      (void)hb_client_->Disconnect(server_url);
    }
    hb_client_->Finalize();
    hb_client_.reset();
  }
  return true;
}

bool ComputeGraphNode::Register() {
  MS_EXCEPTION_IF_NULL(hb_client_);
  MS_EXCEPTION_IF_NULL(tcp_client_);
  const auto &server_url = meta_server_addr_.GetUrl();
  if (!hb_client_->IsConnected(server_url)) {
    if (!hb_client_->Connect(server_url, kNoRetry)) {
      MS_LOG(WARNING) << "Failed to connect to the meta server node url: " << server_url;
      return false;
    }
  }

  if (!tcp_client_->IsConnected(server_url)) {
    if (!tcp_client_->Connect(server_url, kNoRetry)) {
      MS_LOG(WARNING) << "Failed to connect to the meta server node url: " << server_url;
      return false;
    }
  }

  RegistrationMessage reg_msg;
  reg_msg.set_node_id(node_id_);
  reg_msg.set_role(role_);

  // Set the local hostname.
  char host_name[MAX_HOSTNAME_LEN] = {0};
  if (gethostname(host_name, MAX_HOSTNAME_LEN) != 0) {
    MS_LOG(ERROR) << "Failed to get local host name.";
    return false;
  }
  reg_msg.set_host_name(std::string(host_name));

  std::string content = reg_msg.SerializeAsString();
  auto message = CreateMessage(server_url, MessageName::kRegistration, content);
  MS_EXCEPTION_IF_NULL(message);

  MessageBase *response = hb_client_->ReceiveSync(std::move(message));
  if (response == nullptr) {
    return false;
  }
  auto body = response->body;
  delete response;
  response = nullptr;

  RegistrationRespMessage reg_resp_msg;
  (void)reg_resp_msg.ParseFromArray(body.c_str(), SizeToInt(body.length()));

  if (reg_resp_msg.success()) {
    authenticated_ = true;
    rank_id_ = reg_resp_msg.rank_id();
    MS_LOG(INFO) << "The compute graph node: " << node_id_ << " has been registered successfully.";
    return true;
  } else {
    MS_LOG(INFO) << "Failed to register the compute graph node: " << node_id_;
    return false;
  }
}

bool ComputeGraphNode::Unregister() {
  MS_EXCEPTION_IF_NULL(hb_client_);

  UnregistrationMessage unreg_msg;
  unreg_msg.set_node_id(node_id_);

  std::string content = unreg_msg.SerializeAsString();
  auto message = CreateMessage(meta_server_addr_.GetUrl(), MessageName::kUnregistration, content);
  MS_EXCEPTION_IF_NULL(message);

  const uint32_t timeout = 6;
  MessageBase *response = hb_client_->ReceiveSync(std::move(message), timeout);
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
  try {
    MS_EXCEPTION_IF_NULL(hb_client_);

    MS_LOG(INFO) << "The heartbeat thread is started.";
    uint32_t interval = 3;
    uint32_t timeout = 10;

    while (enable_hb_) {
      if (topo_state_ == TopoState::kInitializing && ElapsedTime(start_time_) > kTopoInitTimeout) {
        MS_LOG(EXCEPTION) << "Building networking for " << role_ << " failed.";
      }
      HeartbeatMessage hb_msg;
      hb_msg.set_node_id(node_id_);

      const auto &server_url = meta_server_addr_.GetUrl();
      std::string content = hb_msg.SerializeAsString();
      auto message = CreateMessage(server_url, MessageName::kHeartbeat, content);
      MS_EXCEPTION_IF_NULL(message);

      MessageBase *response = hb_client_->ReceiveSync(std::move(message), timeout);
      if (response == nullptr) {
        MS_LOG(ERROR)
          << "Failed to send heartbeat message to meta server node and try to reconnect to the meta server.";
        if (!Reconnect()) {
          if (!recovery::IsEnableRecovery() && topo_state_ != TopoState::kInitializing) {
            topo_state_ = TopoState::kFailed;
            if (abnormal_callback_ != nullptr) {
              (*abnormal_callback_)();
            }
            MS_LOG(EXCEPTION) << "Failed to connect to the meta server. Maybe it has exited. Please check log.";
          } else {
            MS_LOG(ERROR) << "Failed to connect to the meta server. Maybe it has exited. Please check log.";
          }
        }
      } else {
        auto &body = response->body;
        HeartbeatRespMessage resp_msg;
        (void)resp_msg.ParseFromArray(body.c_str(), SizeToInt(body.length()));
        topo_state_ = static_cast<TopoState>(resp_msg.topo_state());
        auto nodes_num = resp_msg.nodes_num();
        auto abnormal_nodes_num = resp_msg.abnormal_nodes_num();
        if (abnormal_nodes_num > 0 && !recovery::IsEnableRecovery()) {
          topo_state_ = TopoState::kFailed;
          if (abnormal_callback_ != nullptr) {
            (*abnormal_callback_)();
          }
          delete response;
          MS_LOG(EXCEPTION) << "The state of the cluster is error, total nodes num: " << nodes_num
                            << ", abnormal nodes num: " << abnormal_nodes_num;
        }
        delete response;
      }

      (void)sleep(interval);
    }

    MS_LOG(INFO) << "The heartbeat thread is finished.";
  } catch (const std::exception &e) {
    MsException::Instance().SetException();
  }
  return true;
}

bool ComputeGraphNode::ReconnectIfNeeded(const std::function<bool(void)> &func, const std::string &error,
                                         size_t retry) {
  bool success = false;

  while (!success && retry > 0) {
    success = func();
    if (!success) {
      // Retry to reconnect to the meta server.
      MS_LOG(WARNING) << error;
      (void)sleep(kExecuteInterval);
      (void)Reconnect();
    }
    --retry;
  }
  return success;
}

bool ComputeGraphNode::Reconnect() {
  MS_ERROR_IF_NULL_W_RET_VAL(tcp_client_, false);
  MS_ERROR_IF_NULL_W_RET_VAL(hb_client_, false);

  auto server_url = meta_server_addr_.GetUrl();
  // Disconnect from meta server node firstly.
  while (tcp_client_->IsConnected(server_url)) {
    (void)tcp_client_->Disconnect(server_url);
  }
  while (hb_client_->IsConnected(server_url)) {
    (void)hb_client_->Disconnect(server_url);
  }

  // Reconnect to the meta server node.
  if (!tcp_client_->IsConnected(server_url)) {
    (void)tcp_client_->Connect(server_url, kNoRetry);
  }
  if (!tcp_client_->IsConnected(server_url)) {
    return false;
  }
  if (!hb_client_->IsConnected(server_url)) {
    (void)hb_client_->Connect(server_url, kNoRetry);
  }
  return hb_client_->IsConnected(server_url);
}

bool ComputeGraphNode::SendMessageToMSN(const std::string msg_name, const std::string &msg_body, bool sync) {
  MS_EXCEPTION_IF_NULL(tcp_client_);

  auto message = CreateMessage(meta_server_addr_.GetUrl(), msg_name, msg_body);
  MS_EXCEPTION_IF_NULL(message);

  if (sync) {
    auto retval = tcp_client_->SendSync(std::move(message));
    if (retval) {
      return true;
    } else {
      return false;
    }
  } else {
    (void)tcp_client_->SendSync(std::move(message));
    return true;
  }
}

std::shared_ptr<std::string> ComputeGraphNode::RetrieveMessageFromMSN(const std::string &msg_name, uint32_t timeout) {
  return RetrieveMessageFromMSN(msg_name, msg_name);
}

bool ComputeGraphNode::PutMetadata(const std::string &name, const std::string &value, bool sync) {
  MetadataMessage metadata;
  metadata.set_name(name);
  metadata.set_value(value);
  return SendMessageToMSN(std::to_string(static_cast<int>(MessageName::kWriteMetadata)), metadata.SerializeAsString(),
                          sync);
}

bool ComputeGraphNode::PutMetadata(const std::string &name, const void *value, const size_t &size) {
  MetadataMessage metadata;
  metadata.set_name(name);
  metadata.set_value(value, size);
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
    (void)metadata.ParseFromArray(retval->body.c_str(), SizeToInt(retval->body.length()));
    return metadata.value();
  }
  return "";
}

bool ComputeGraphNode::DeleteMetadata(const std::string &name, uint32_t timeout) {
  MetadataMessage metadata;
  metadata.set_name(name);

  auto message =
    CreateMessage(meta_server_addr_.GetUrl(), std::to_string(static_cast<int>(MessageName::kDeleteMetadata)),
                  metadata.SerializeAsString());
  MS_EXCEPTION_IF_NULL(message);

  MS_EXCEPTION_IF_NULL(tcp_client_);
  auto retval = tcp_client_->ReceiveSync(std::move(message), timeout);
  if (retval != rpc::NULL_MSG && (retval->name == std::to_string(static_cast<int>(MessageName::kValidMetadata)))) {
    return true;
  } else {
    return false;
  }
}

// The transaction of the exchange process is as follows:
// step 1: RANK[0]       - Start the exchange process (set EXCHANGE_META_${name} flag);
// step 2: RANK[1-(N-1)] - Start the exchange process (check EXCHANGE_META_${name} flag);
// step 3: RANK[0-(N-1)] - Do the exchange (exchange the metadata through meta server node);
// step 4: RANK[0-(N-1)] - Finish the exchange process (set EXCHANGE_META_${name}_DONE_RANK_${RANK_ID});
// step 5: RANK[0]       - Exit the exchange process (check all the EXCHANGE_META_${name}_DONE_RANK_${RANK_ID} flag &
//                                                    delete all the EXCHANGE_META_${name}_DONE_RANK_${RANK_ID} flag &
//                                                    delete all the metadata in results &
//                                                    delete EXCHANGE_META_${name} flag);
// step 6: RANK[1-(N-1)] - Exit the exchange process (check EXCHANGE_META_${name} flag deleted);
bool ComputeGraphNode::ExchangeMetadata(const std::string &biz, const size_t &rank_size,
                                        const std::vector<std::string> &names_prefix,
                                        const std::vector<std::string> &values,
                                        std::map<std::string, std::string> *results, uint32_t timeout) {
  std::unique_lock<std::shared_mutex> lock(exchange_meta_mutex_);
  MS_ERROR_IF_NULL_W_RET_VAL(results, false);
  MS_LOG(INFO) << "Start to exchange metadata for the biz: " << biz;
  if (names_prefix.size() != values.size()) {
    return false;
  }
  if (timeout == 0) {
    return false;
  }
  bool success = false;

  // step 1 set the start flag.
  std::string meta_name = kStartExchangeMetaPrefix + biz;
  if (rank_id_ == 0) {
    EXECUTE_WITH_TIMEOUT(PutMetadata(meta_name, kMetaFlagValue), kExecuteInterval,
                         "Failed to set the metadata exchange flag " + meta_name + ".", success, timeout);
  }
  // step 2 check the start flag.
  EXECUTE_WITH_EXPECTED(GetMetadata(meta_name), kMetaFlagValue, kExecuteInterval,
                        "Failed to check the metadata exchange flag " << meta_name << ".", timeout);
  // step 3 exchange the metadata.
  for (size_t i = 0; i < names_prefix.size(); ++i) {
    auto name = names_prefix[i] + std::to_string(rank_id_);
    auto value = values[i];
    EXECUTE_WITH_TIMEOUT(PutMetadata(name, value), kExecuteInterval,
                         "Failed to put metadata name: " + name + ", value: " + value + ".", success, timeout);
  }
  for (size_t i = 0; i < rank_size; ++i) {
    for (size_t j = 0; j < names_prefix.size(); ++j) {
      auto other_name = names_prefix[j] + std::to_string(i);
      while (true) {
        auto other_value = GetMetadata(other_name);
        if (other_value.length() > 0) {
          (*results)[other_name] = other_value;
          break;
        } else {
          MS_LOG(WARNING) << "Failed to get metadata " << other_name << " from rank " << i;
          (void)sleep(kExecuteInterval);
        }
      }
    }
  }
  // step 4 set the exchange done flag.
  auto done = kExchangeMetaDonePrefix + std::to_string(rank_id_);
  EXECUTE_WITH_TIMEOUT(PutMetadata(done, kMetaFlagValue), kExecuteInterval,
                       "Failed to set the metadata exchange done flag " + done + ".", success, timeout);
  // step 5 check all node done and then clear the metadata in meta server and remove the start flag finally.
  if (rank_id_ == 0) {
    for (size_t i = 0; i < rank_size; ++i) {
      auto other_done = kExchangeMetaDonePrefix + std::to_string(i);
      EXECUTE_WITH_EXPECTED(
        GetMetadata(other_done), kMetaFlagValue, kExecuteInterval,
        "Failed to check the metadata exchange done flag " << other_done << " for rank " << i << ".", timeout);
    }
    for (size_t i = 0; i < rank_size; ++i) {
      auto other_done = kExchangeMetaDonePrefix + std::to_string(i);
      EXECUTE_WITH_TIMEOUT(DeleteMetadata(other_done), kExecuteInterval,
                           "Failed to delete the metadata exchange done flag " + other_done + ".", success, timeout);
    }
    for (auto iter = results->begin(); iter != results->end(); ++iter) {
      auto delete_name = iter->first;
      EXECUTE_WITH_TIMEOUT(DeleteMetadata(delete_name), kExecuteInterval,
                           "Failed to delete the metadata: " + delete_name + ".", success, timeout);
    }
    EXECUTE_WITH_TIMEOUT(DeleteMetadata(meta_name), kExecuteInterval,
                         "Failed to delete the metadata flag: " + meta_name + ".", success, timeout);
  }

  // step 6 check the exchange finish flag.
  EXECUTE_WITH_EXPECTED(GetMetadata(meta_name), kMetaDeleteFlagValue, kExecuteInterval,
                        "Failed to check the metadata exchange flag " << meta_name << ".", timeout);
  MS_LOG(INFO) << "The metadata exchange for the biz: " << biz << " has been completed";
  return true;
}

std::vector<std::string> ComputeGraphNode::GetHostNames(const std::string &role) {
  auto retval = RetrieveMessageFromMSN(std::to_string(static_cast<int>(MessageName::kGetHostNames)), role);
  if (retval != nullptr) {
    nlohmann::json hostnames = nlohmann::json::parse(*retval);
    return hostnames.at(kHostNames).get<std::vector<std::string>>();
  } else {
    return std::vector<std::string>();
  }
}

void ComputeGraphNode::set_abnormal_callback(std::shared_ptr<std::function<void(void)>> abnormal_callback) {
  abnormal_callback_ = abnormal_callback;
}

std::shared_ptr<std::string> ComputeGraphNode::RetrieveMessageFromMSN(const std::string &msg_name,
                                                                      const std::string &msg_body, uint32_t timeout) {
  MS_EXCEPTION_IF_NULL(tcp_client_);

  auto message = CreateMessage(meta_server_addr_.GetUrl(), msg_name, msg_body);
  MS_EXCEPTION_IF_NULL(message);

  auto retval = tcp_client_->ReceiveSync(std::move(message), timeout);
  if (retval != rpc::NULL_MSG) {
    return std::make_shared<std::string>(retval->body);
  }
  return nullptr;
}
}  // namespace topology
}  // namespace cluster
}  // namespace distributed
}  // namespace mindspore
