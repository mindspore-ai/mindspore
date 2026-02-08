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
#include "include/cluster/rpc/tcp_store.h"
#include <memory>
#include <string>
#include "utils/ms_utils.h"
#include "include/utils/utils.h"
#if defined(__linux__) && defined(WITH_BACKEND)
#include "cluster/rpc/core/comm_util.h"
#include "cluster/topology/meta_server_node.h"
#endif

namespace mindspore {
namespace distributed {
namespace cluster {
TCPStoreClient::TCPStoreClient(const std::string &ip, int64_t port, bool is_master, int64_t timeout, int64_t world_size,
                               bool wait_for_workers) {
#if defined(__linux__) && defined(WITH_BACKEND)
  static bool dry_run = common::IsCompileSimulation();
  if (MS_UNLIKELY(dry_run)) {
    return;
  }
  size_t start_tick = LongToSize(CURRENT_TIMESTAMP_MILLI.count());
  ip_ = ip;
  port_ = port;
  world_size_ = world_size;
  is_master_ = is_master;
  timeout_ = timeout;
  if (ip.empty()) {
    MS_LOG(EXCEPTION) << "TCPStore does not support empty IP address.";
  }

  auto node_id = ps::core::CommUtil::GenerateVariableUUID();

  std::string address = ip + ":" + std::to_string(port);
  if (is_master) {
    MS_LOG(DEBUG) << "For TCPStoreClient, start address " << address;
    server_node_ =
      std::make_shared<topology::MetaServerNode>(node_id, kEnvRoleOfServer, world_size_, timeout_, address);
    MS_EXCEPTION_IF_NULL(server_node_);
    server_node_->disable_heartbeat();
    MS_EXCEPTION_IF_CHECK_FAIL(server_node_->Initialize(), "Failed to initialize the master node.");
  }
  client_node_ = std::make_shared<topology::TcpNodeBase>(node_id, kEnvRoleOfWorker, address, timeout_);
  MS_EXCEPTION_IF_NULL(client_node_);
  MS_EXCEPTION_IF_CHECK_FAIL(client_node_->Initialize(), "Failed to initialize the client node.");

  while (is_master && wait_for_workers &&
         LongToSize(CURRENT_TIMESTAMP_MILLI.count()) - start_tick <= static_cast<uint32_t>(timeout_)) {
    bool success = server_node_->Initialized();
    if (!success) {
      // seconds interval for large-scale cluster.
      unsigned int interval = 1;
      // milliseconds interval for small-scale cluster.
      uint32_t interval_ms = 10;
      SleepBasedOnScale(interval, interval_ms);
    } else {
      return;
    }
  }
  if (is_master && wait_for_workers) {
    MS_LOG(WARNING) << "Time out after " << timeout_ << "ms waiting for client.";
  }
#else
  MS_LOG(EXCEPTION) << "The TCPStore is only supported on linux platform.";
#endif
}

TCPStoreClient::~TCPStoreClient() {
#if defined(__linux__) && defined(WITH_BACKEND)
  if (client_node_) {
    client_node_->Finalize(true);
  }
  if (server_node_) {
    server_node_->Finalize(true);
  }
#else
  MS_LOG(EXCEPTION) << "The TCPStore is only supported on linux platform.";
#endif
}

py::bytes TCPStoreClient::GetKey(const std::string &key) {
#if defined(__linux__) && defined(WITH_BACKEND)
  static bool dry_run = common::IsCompileSimulation();
  if (MS_UNLIKELY(dry_run)) {
    MS_LOG(EXCEPTION) << "For TCPStoreClient::GetKey, the output shape depends on the actual execution, "
                      << "and it will affect the accuracy of memory in dryrun mode.";
  }
  MS_EXCEPTION_IF_NULL(client_node_);
  auto data = client_node_->ReTryGetMetadata(key, timeout_);
  return py::bytes(reinterpret_cast<const char *>(data.data()), data.size());
#else
  MS_LOG(EXCEPTION) << "The TCPStore is only supported on linux platform.";
#endif
}

void TCPStoreClient::SetKey(const std::string &key, const std::string &value) {
#if defined(__linux__) && defined(WITH_BACKEND)
  static bool dry_run = common::IsCompileSimulation();
  if (MS_UNLIKELY(dry_run)) {
    return;
  }
  MS_EXCEPTION_IF_NULL(client_node_);
  (void)client_node_->PutMetadata(key, value, value.size());
#else
  MS_LOG(EXCEPTION) << "The TCPStore is only supported on linux platform.";
#endif
}

int64_t TCPStoreClient::AddKey(const std::string &key, int64_t amount) {
#if defined(__linux__) && defined(WITH_BACKEND)
  static bool dry_run = common::IsCompileSimulation();
  if (MS_UNLIKELY(dry_run)) {
    return amount;
  }
  MS_EXCEPTION_IF_NULL(client_node_);
  return client_node_->AddMetadata(key, amount);
#else
  MS_LOG(EXCEPTION) << "The TCPStore is only supported on linux platform.";
#endif
}

bool TCPStoreClient::DeleteKey(const std::string &key) {
#if defined(__linux__) && defined(WITH_BACKEND)
  static bool dry_run = common::IsCompileSimulation();
  if (MS_UNLIKELY(dry_run)) {
    return true;
  }
  MS_EXCEPTION_IF_NULL(client_node_);
  return client_node_->DeleteMetadata(key);
#else
  MS_LOG(EXCEPTION) << "The TCPStore is only supported on linux platform.";
#endif
}

}  // namespace cluster
}  // namespace distributed
}  // namespace mindspore
