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
#include "include/cluster/topology/compute_graph_node.h"
#include <memory>
#include <vector>
#include <map>
#include <string>
#include <utility>
#include <random>
#include <nlohmann/json.hpp>
#include "utils/log_adapter.h"
#include "utils/ms_exception.h"
#include "include/cluster/topology/common.h"
#include "include/cluster/topology/constants.h"
#include "proto/topology.pb.h"
#include "include/cluster/topology/ps_context.h"
#include "cluster/rpc/tcp/constants.h"
#include "utils/convert_utils_base.h"

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
    success = ReconnectWithTimeoutWindow(std::bind(&ComputeGraphNode::Register, this),
                                         "Failed to register and try to reconnect to the meta server.", topo_timeout_);
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
      if (enable_recovery_) {
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

void ComputeGraphNode::StopHeartBeatThread() {
  MS_LOG(INFO) << "Start waiting for heart beat thread to end.";
  enable_hb_ = false;
  if (heartbeat_.joinable()) {
    heartbeat_.join();
  }
}

bool ComputeGraphNode::Register() {
  MS_EXCEPTION_IF_NULL(hb_client_);
  MS_EXCEPTION_IF_NULL(tcp_client_);
  const auto &server_url = meta_server_addr_.GetUrl();
  MS_LOG(INFO) << "Start connecting heartbeat client.";
  if (!hb_client_->IsConnected(server_url)) {
    if (!hb_client_->Connect(server_url, kNoRetry)) {
      MS_LOG(WARNING) << "Failed to connect to the meta server node url: " << server_url;
      return false;
    }
  }

  MS_LOG(INFO) << "Start connecting business client.";
  if (!tcp_client_->IsConnected(server_url)) {
    if (!tcp_client_->Connect(server_url, kNoRetry)) {
      MS_LOG(WARNING) << "Failed to connect to the meta server node url: " << server_url;
      return false;
    }
  }

  RegistrationMessage reg_msg;
  reg_msg.set_node_id(node_id_);
  reg_msg.set_role(role_);
  reg_msg.set_device_id(device_id_);

  // Set the local hostname.
  char host_name[MAX_HOSTNAME_LEN] = {0};
  if (gethostname(host_name, MAX_HOSTNAME_LEN) != 0) {
    MS_LOG(ERROR) << "Failed to get local host name.";
    return false;
  }
  reg_msg.set_host_name(std::string(host_name));

  // Set client ip address.
  client_ip_ = hb_client_->GetClientIPByDstUrl(server_url);
  reg_msg.set_host_ip(client_ip_);

  std::string content = reg_msg.SerializeAsString();
  auto message = CreateMessage(server_url, MessageName::kRegistration, content);
  MS_EXCEPTION_IF_NULL(message);

  MS_VLOG(VL_DISTRIBUTED_TRACE) << "Start register.";
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
    MS_LOG(EXCEPTION) << "Failed to register the compute graph node: " << node_id_
                      << ". Reason: " << reg_resp_msg.error_reason();
  }
}

bool ComputeGraphNode::Unregister() {
  MS_EXCEPTION_IF_NULL(hb_client_);
  MS_EXCEPTION_IF_NULL(tcp_client_);

  UnregistrationMessage unreg_msg;
  unreg_msg.set_node_id(node_id_);

  std::string content = unreg_msg.SerializeAsString();
  auto message = CreateMessage(meta_server_addr_.GetUrl(), MessageName::kUnregistration, content);
  MS_EXCEPTION_IF_NULL(message);
  // 10000ms
  const uint32_t timeout = 10 * 1000;
  MessageBase *response = nullptr;
  if (disable_heartbeat_) {
    response = tcp_client_->ReceiveSync(std::move(message), timeout);
  } else {
    response = hb_client_->ReceiveSync(std::move(message), timeout);
  }
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
  std::random_device rd;
  std::mt19937 gen(rd());
  int random_time_lower =
    common::GetEnv("MS_RETRY_INTERVAL_LOWER").empty() ? 3 : std::stoi(common::GetEnv("MS_RETRY_INTERVAL_LOWER"));
  int random_time_upper =
    common::GetEnv("MS_RETRY_INTERVAL_UPPER").empty() ? 5 : std::stoi(common::GetEnv("MS_RETRY_INTERVAL_UPPER"));
  std::uniform_int_distribution<> distrib(random_time_lower, random_time_upper);
  MS_LOG(INFO) << "Interval of heartbeat lower and upper are " << random_time_lower << " and " << random_time_upper;
  try {
    MS_EXCEPTION_IF_NULL(hb_client_);

    std::string env_hb_retry_timeout = common::GetEnv("MS_HEARTBEAT_RETRY_TIMEOUT");
    int int_hb_retry_timeout = env_hb_retry_timeout.empty() ? kExecuteRetryTimeout : std::stoi(env_hb_retry_timeout);
    size_t hb_retry_timeout = (int_hb_retry_timeout < 0) ? UINT64_MAX : int_hb_retry_timeout;
    MS_LOG(INFO) << "The heartbeat thread is started. Retry timeout after heartbeat failure is " << hb_retry_timeout
                 << " seconds.";
    while (enable_hb_) {
      HeartbeatMessage hb_msg;
      hb_msg.set_node_id(node_id_);

      const auto &server_url = meta_server_addr_.GetUrl();
      std::string content = hb_msg.SerializeAsString();
      auto message = CreateMessage(server_url, MessageName::kHeartbeat, content);
      MS_EXCEPTION_IF_NULL(message);
      MS_VLOG(VL_DISTRIBUTED_TRACE) << "Start heart beat.";
      MessageBase *response = hb_client_->ReceiveSync(std::move(message));
      if (response == nullptr) {
        MS_LOG(WARNING)
          << "Failed to send heartbeat message to meta server node and try to reconnect to the meta server.";
        if (!ReconnectWithTimeout(hb_retry_timeout)) {
          if (!enable_recovery_ && topo_state_ != TopoState::kInitializing) {
            topo_state_ = TopoState::kFailed;
            if (abnormal_callback_ != nullptr) {
              (*abnormal_callback_)();
            }
            MS_LOG(EXCEPTION) << "It is possible that an exception has occurred in the TCP connection between the "
                                 "scheduler and the worker. Please check the scheduler process status via scheduler's "
                                 "log and verify that the host network is functioning properly.";
          } else {
            MS_LOG(ERROR) << "Failed to connect to the meta server. Maybe it has exited. Please check scheduler's log.";
          }
        }
      } else {
        auto &body = response->body;
        HeartbeatRespMessage resp_msg;
        (void)resp_msg.ParseFromArray(body.c_str(), SizeToInt(body.length()));
        topo_state_ = static_cast<TopoState>(resp_msg.topo_state());
        if (topo_state_ == TopoState::kInitialized && disable_heartbeat_) {
          MS_LOG(WARNING)
            << "After cluster is initialized, heartbeat client will be disconnected if MS_DISABLE_HEARTBEAT is set to "
               "1. In this scenario, MS_HEARTBEAT_RETRY_TIMEOUT does not take effect.";
          (void)hb_client_->Disconnect(meta_server_addr_.GetUrl());
          break;
        }

        auto nodes_num = resp_msg.nodes_num();
        auto abnormal_nodes_num = resp_msg.abnormal_nodes_num();
        if (abnormal_nodes_num > 0 && !enable_recovery_) {
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
      MS_VLOG(VL_DISTRIBUTED_TRACE) << "End heart beat.";

      uint32_t interval = IntToUint(distrib(gen));
      MS_LOG(DEBUG) << "Heart beat interval " << interval;
      SleepBasedOnScale(interval);
    }
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
      SleepBasedOnScale(kExecuteInterval);
      (void)Reconnect();
    }
    --retry;
  }
  return success;
}

bool ComputeGraphNode::ReconnectWithTimeoutWindow(const std::function<bool(void)> &func, const std::string &error,
                                                  size_t time_out) {
  size_t time_out_in_milli = time_out * 1000;
  size_t start_tick = LongToSize(CURRENT_TIMESTAMP_MILLI.count());
  bool success = false;
  while (!success && LongToSize(CURRENT_TIMESTAMP_MILLI.count()) - start_tick <= time_out_in_milli) {
    success = func();
    if (!success) {
      // Retry to reconnect to the meta server.
      MS_LOG(WARNING) << error;
      SleepBasedOnScale(kExecuteInterval);
      (void)Reconnect();
    }
  }
  return success;
}

bool ComputeGraphNode::ReconnectWithTimeout(size_t timeout) {
  const size_t timeout_ms = timeout * 1000;
  const auto start_time = CURRENT_TIMESTAMP_MILLI;
  size_t retry_count = 0;
  bool success = false;

  const auto deadline = start_time + std::chrono::milliseconds(timeout_ms);

  while (!success && CURRENT_TIMESTAMP_MILLI < deadline) {
    success = Reconnect();
    if (!success) {
      MS_LOG(WARNING) << "Retry to reconnect to the meta server... Retry count: " << ++retry_count;
      SleepBasedOnScale(kExecuteInterval);
    }
  }

  if (success) {
    MS_LOG(WARNING) << "Successfully reconnected to the meta server.";
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
    MS_LOG(INFO) << "Start reconnecting business client.";
    (void)tcp_client_->Connect(server_url, kNoRetry);
  }
  if (!tcp_client_->IsConnected(server_url)) {
    return false;
  }
  if (!hb_client_->IsConnected(server_url)) {
    MS_LOG(INFO) << "Start reconnecting heartbeat client.";
    (void)hb_client_->Connect(server_url, kNoRetry);
  }
  return hb_client_->IsConnected(server_url);
}

void ComputeGraphNode::set_abnormal_callback(std::shared_ptr<std::function<void(void)>> abnormal_callback) {
  abnormal_callback_ = abnormal_callback;
}
}  // namespace topology
}  // namespace cluster
}  // namespace distributed
}  // namespace mindspore
