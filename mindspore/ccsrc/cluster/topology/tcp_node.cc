/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#include "include/cluster/rpc/tcp_node.h"
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
bool TcpNodeBase::ReConnectWithTimeout(const std::function<bool(void)> &func, const std::string &error,
                                       size_t time_out) {
  size_t start_tick = LongToSize(CURRENT_TIMESTAMP_MILLI.count());
  while (LongToSize(CURRENT_TIMESTAMP_MILLI.count()) - start_tick <= time_out) {
    bool success = func();
    if (!success) {
      // Retry to reconnect to the meta server.
      MS_LOG(WARNING) << error;
      // seconds interval for large-scale cluster.
      unsigned int interval = 1;
      // milliseconds interval for small-scale cluster.
      uint32_t interval_ms = 10;
      SleepBasedOnScale(interval, interval_ms);
      (void)ReConnect();
    } else {
      return success;
    }
  }
  MS_LOG(EXCEPTION) << "The client socket has timed out after " << time_out << "ms while trying to connect to "
                    << address_id_;
}

bool TcpNodeBase::ReConnect() {
  MS_ERROR_IF_NULL_W_RET_VAL(tcp_client_, false);
  auto server_url = meta_server_addr_.GetUrl();
  // Disconnect from meta server node firstly.
  while (tcp_client_->IsConnected(server_url)) {
    (void)tcp_client_->Disconnect(server_url);
  }

  // ReConnect to the meta server node.
  if (!tcp_client_->IsConnected(server_url)) {
    MS_LOG(INFO) << "Start reconnecting business client.";
    (void)tcp_client_->Connect(server_url, kNoRetry);
  }
  return tcp_client_->IsConnected(server_url);
}

bool TcpNodeBase::RegisterTcpClient() {
  MS_EXCEPTION_IF_NULL(tcp_client_);
  const auto &server_url = meta_server_addr_.GetUrl();
  MS_LOG(INFO) << "Start connecting business client. the meta server node url: " << server_url;
  if (!tcp_client_->IsConnected(server_url)) {
    if (!tcp_client_->Connect(server_url, kNoRetry)) {
      MS_LOG(WARNING) << "Failed to connect to the meta server node url: " << server_url;
      return false;
    }
  }

  RegistrationMessage reg_msg;
  reg_msg.set_node_id(node_id_);
  reg_msg.set_role(kEnvRoleOfClient);
  reg_msg.set_device_id(0);

  // Set the local hostname.
  char host_name[MAX_HOSTNAME_LEN] = {0};
  if (gethostname(host_name, MAX_HOSTNAME_LEN) != 0) {
    MS_LOG(ERROR) << "Failed to get local host name.";
    return false;
  }
  reg_msg.set_host_name(std::string(host_name));

  // Set client ip address.
  auto client_ip = tcp_client_->GetClientIPByDstUrl(server_url);
  reg_msg.set_host_ip(client_ip);

  std::string content = reg_msg.SerializeAsString();
  auto message = CreateMessage(server_url, MessageName::kRegistration, content);
  MS_EXCEPTION_IF_NULL(message);
  MS_VLOG(VL_DISTRIBUTED_TRACE) << "Start register.";
  MessageBase *response = tcp_client_->ReceiveSync(std::move(message));
  if (response == nullptr) {
    return false;
  }
  auto body = response->body;
  delete response;
  response = nullptr;

  RegistrationRespMessage reg_resp_msg;
  (void)reg_resp_msg.ParseFromArray(body.c_str(), SizeToInt(body.length()));

  if (reg_resp_msg.success()) {
    MS_LOG(INFO) << "The tcp node: " << node_id_ << " has been registered successfully.";
    return true;
  } else {
    MS_LOG(EXCEPTION) << "Failed to register the tcp node: " << node_id_ << ". Reason: " << reg_resp_msg.error_reason();
  }

  return true;
}

bool TcpNodeBase::Finalize(bool force) {
  // Release the TCP client.
  bool enable_ssl = ps::PSContext::instance()->enable_ssl();
  const auto &server_url = meta_server_addr_.GetUrl();
  if (tcp_client_ != nullptr) {
    if (!enable_ssl) {
      (void)tcp_client_->Disconnect(server_url);
    }
    tcp_client_->Finalize();
    tcp_client_.reset();
  }
  return true;
}

TcpNodeBase::~TcpNodeBase() {
  if (!finalized_) {
    try {
      (void)Finalize(true);
    } catch (std::exception &) {
      MS_LOG(ERROR) << "Failed to finalize TcpNodeBase.";
    }
  }
}

bool TcpNodeBase::Initialized() {
  // The cgn is initialized only when the cluster is ready, or there will be error message unexpected.
  return topo_state_ == TopoState::kInitialized;
}

bool TcpNodeBase::Initialize() {
  MS_LOG(WARNING) << "Begin to Initialize TcpNodeBase.";
  if (address_id_.empty()) {
    MS_LOG(INFO) << "TcpNodeBase address is None.";
    return false;
  }
  // Init the address of meta server node.
  RETURN_IF_FALSE_WITH_LOG(ParseIPPort(address_id_, &meta_server_addr_), "Failed to Parse IP or Port.");
  // Init the TCP client.
  bool enable_ssl = ps::PSContext::instance()->enable_ssl();
  const auto &server_url = meta_server_addr_.GetUrl();
  tcp_client_ = std::make_unique<rpc::TCPClient>(enable_ssl);
  MS_EXCEPTION_IF_NULL(tcp_client_);
  RETURN_IF_FALSE_WITH_LOG(tcp_client_->Initialize(), "Failed to create the TCP client.");

  // Register itself to meta server node.
  bool success = false;
  if (!enable_ssl) {
    success = ReConnectWithTimeout(std::bind(&TcpNodeBase::RegisterTcpClient, this),
                                   "Failed to register and try to reconnect to the meta server.", timeout_);
  } else {
    size_t start_tick = LongToSize(CURRENT_TIMESTAMP_MILLI.count());
    while (!success && LongToSize(CURRENT_TIMESTAMP_MILLI.count()) - start_tick <= timeout_) {
      success = RegisterTcpClient();
      if (success) {
        break;
      }

      if (tcp_client_ != nullptr) {
        (void)tcp_client_->Disconnect(server_url);
        tcp_client_->Finalize();
        tcp_client_.reset();
      }
      tcp_client_ = std::make_unique<rpc::TCPClient>(enable_ssl);
      MS_EXCEPTION_IF_NULL(tcp_client_);
      RETURN_IF_FALSE_WITH_LOG(tcp_client_->Initialize(), "Failed to create the TCP client.");
      // seconds interval for large-scale cluster.
      unsigned int interval = 1;
      // milliseconds interval for small-scale cluster.
      uint32_t interval_ms = 10;
      SleepBasedOnScale(interval, interval_ms);
    }
  }
  if (!success) {
    return false;
  }
  client_ip_ = tcp_client_->GetClientIPByDstUrl(server_url);
  MS_LOG(WARNING) << "Finish to Initialize TcpNodeBase.";
  return true;
}

bool TcpNodeBase::SendMessageToMSN(const std::string msg_name, const std::string &msg_body, bool sync) {
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

bool TcpNodeBase::PutMetadata(const std::string &name, const void *value, const size_t &size) {
  std::lock_guard<std::mutex> lock(mutex_);
  MetadataMessage metadata;
  metadata.set_name(name);
  metadata.set_value(value, size);
  return SendMessageToMSN(std::to_string(static_cast<int>(MessageName::kWriteMetadata)), metadata.SerializeAsString());
}

std::string TcpNodeBase::GetMetadata(const std::string &name, uint32_t) {
  std::lock_guard<std::mutex> lock(mutex_);
  MetadataMessage metadata;
  metadata.set_name(name);

  auto message = CreateMessage(meta_server_addr_.GetUrl(), std::to_string(static_cast<int>(MessageName::kReadMetadata)),
                               metadata.SerializeAsString());
  MS_EXCEPTION_IF_NULL(message);

  MS_EXCEPTION_IF_NULL(tcp_client_);
  auto retval = tcp_client_->ReceiveSync(std::move(message));
  if (retval != rpc::NULL_MSG && (retval->name == std::to_string(static_cast<int>(MessageName::kValidMetadata)))) {
    (void)metadata.ParseFromArray(retval->body.c_str(), SizeToInt(retval->body.length()));
    return metadata.value();
  }
  return "";
}

int64_t TcpNodeBase::AddMetadata(const std::string &name, int64_t value) {
  std::lock_guard<std::mutex> lock(mutex_);
  MetadataMessage metadata;
  metadata.set_name(name);
  metadata.set_value(std::to_string(value));
  auto message = CreateMessage(meta_server_addr_.GetUrl(), std::to_string(static_cast<int>(MessageName::kAddMetadata)),
                               metadata.SerializeAsString());
  MS_EXCEPTION_IF_NULL(message);

  MS_EXCEPTION_IF_NULL(tcp_client_);
  auto retval = tcp_client_->ReceiveSync(std::move(message));
  if (retval != rpc::NULL_MSG && (retval->name == std::to_string(static_cast<int>(MessageName::kValidMetadata)))) {
    (void)metadata.ParseFromArray(retval->body.c_str(), SizeToInt(retval->body.length()));
    int64_t addVal;
    try {
      addVal = std::stoll(metadata.value());
    } catch (const std::exception &e) {
      MS_LOG(EXCEPTION) << "Add output value is illegal, output is " << metadata.value();
    }
    return addVal;
  }
  MS_LOG(EXCEPTION) << "TCP addition operation failed!";
}

std::string TcpNodeBase::ReTryGetMetadata(const std::string &name, uint32_t timeout) {
  std::lock_guard<std::mutex> lock(mutex_);
  size_t time_out_in_milli = timeout;
  size_t start_tick = LongToSize(CURRENT_TIMESTAMP_MILLI.count());
  size_t end_tick = LongToSize(CURRENT_TIMESTAMP_MILLI.count());
  while (end_tick - start_tick <= time_out_in_milli) {
    MetadataMessage metadata;
    metadata.set_name(name);
    auto message =
      CreateMessage(meta_server_addr_.GetUrl(), std::to_string(static_cast<int>(MessageName::kReadMetadata)),
                    metadata.SerializeAsString());
    MS_EXCEPTION_IF_NULL(message);

    MS_EXCEPTION_IF_NULL(tcp_client_);
    auto wait_time = time_out_in_milli - end_tick + start_tick;
    bool is_send_fail = false;
    auto retval = tcp_client_->ReceiveSync(std::move(message), wait_time, &is_send_fail);
    if (is_send_fail) {
      MS_LOG(EXCEPTION) << "TCPClient send msg to master failed.";
    }
    if (retval != rpc::NULL_MSG && (retval->name == std::to_string(static_cast<int>(MessageName::kValidMetadata)))) {
      (void)metadata.ParseFromArray(retval->body.c_str(), SizeToInt(retval->body.length()));
      return metadata.value();
    }
    uint32_t kIntervalMs = 10;
    std::this_thread::sleep_for(std::chrono::milliseconds(kIntervalMs));
    end_tick = LongToSize(CURRENT_TIMESTAMP_MILLI.count());
    MS_LOG(DEBUG) << "Try tor Get after kExecuteInterval=" << kExecuteInterval;
  }
  MS_LOG(EXCEPTION) << "TCP get operation failed.";
}

bool TcpNodeBase::DeleteMetadata(const std::string &name, uint32_t) {
  std::lock_guard<std::mutex> lock(mutex_);
  MetadataMessage metadata;
  metadata.set_name(name);

  auto message =
    CreateMessage(meta_server_addr_.GetUrl(), std::to_string(static_cast<int>(MessageName::kDeleteMetadata)),
                  metadata.SerializeAsString());
  MS_EXCEPTION_IF_NULL(message);

  MS_EXCEPTION_IF_NULL(tcp_client_);
  auto retval = tcp_client_->ReceiveSync(std::move(message));
  if (retval != rpc::NULL_MSG && (retval->name == std::to_string(static_cast<int>(MessageName::kValidMetadata)))) {
    return true;
  } else {
    return false;
  }
}

bool TcpNodeBase::PutMetadata(const std::string &name, const std::string &value, bool sync) {
  std::lock_guard<std::mutex> lock(mutex_);
  MetadataMessage metadata;
  metadata.set_name(name);
  metadata.set_value(value);
  return SendMessageToMSN(std::to_string(static_cast<int>(MessageName::kWriteMetadata)), metadata.SerializeAsString(),
                          sync);
}

std::shared_ptr<std::string> TcpNodeBase::RetrieveMessageFromMSN(const std::string &msg_name,
                                                                 const std::string &msg_body, uint32_t) {
  MS_EXCEPTION_IF_NULL(tcp_client_);

  auto message = CreateMessage(meta_server_addr_.GetUrl(), msg_name, msg_body);
  MS_EXCEPTION_IF_NULL(message);

  auto retval = tcp_client_->ReceiveSync(std::move(message));
  if (retval != rpc::NULL_MSG) {
    return std::make_shared<std::string>(retval->body);
  }
  return nullptr;
}

std::shared_ptr<std::string> TcpNodeBase::RetrieveMessageFromMSN(const std::string &msg_name, uint32_t timeout) {
  return RetrieveMessageFromMSN(msg_name, msg_name);
}

std::vector<std::string> TcpNodeBase::GetHostNames(const std::string &role) {
  auto retval = RetrieveMessageFromMSN(std::to_string(static_cast<int>(MessageName::kGetHostNames)), role);
  if (retval != nullptr) {
    MS_LOG(INFO) << "Worker gets host names " << *retval;
    nlohmann::json hostnames;
    size_t retry_num = 60;
    do {
      try {
        if (retval != nullptr) {
          hostnames = nlohmann::json::parse(*retval);
        } else {
          MS_LOG(ERROR) << "Get hostnames from sched failed, receive empty message.";
        }
        break;
      } catch (const std::exception &e) {
        MS_LOG(ERROR) << "Worker failed to parse hostname json " << e.what() << ". Retry number: " << retry_num;
        retval = RetrieveMessageFromMSN(std::to_string(static_cast<int>(MessageName::kGetHostNames)), role);
        retry_num--;
        (void)sleep(kExecuteInterval);
      }
    } while (retry_num != 0);
    MS_LOG(DEBUG) << "Successfully get hostnames from scheduler: " << hostnames.dump();
    return hostnames.at(kHostNames).get<std::vector<std::string>>();
  } else {
    return std::vector<std::string>();
  }
}
}  // namespace topology
}  // namespace cluster
}  // namespace distributed
}  // namespace mindspore
