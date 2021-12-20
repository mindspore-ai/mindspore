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

#include <mutex>
#include <vector>
#include <string>
#include <memory>
#include "distributed/cluster/cluster_context.h"
#include "distributed/collective/collective_manager.h"
#include "utils/ms_context.h"
#include "ps/ps_context.h"
#include "debug/common.h"

namespace mindspore {
namespace distributed {
namespace cluster {
ClusterContext::ClusterContext()
    : inited_(false),
      finalized_(true),
      node_num_each_role_({}),
      scheduler_host_(kLocalHost),
      scheduler_port_(kDefaultSchedPort),
      node_(nullptr),
      node_role_(""),
      cluster_config_(nullptr) {}

ClusterContext::~ClusterContext() {
  if (!finalized_) {
    Finalize();
  }
  finalized_ = true;
}

std::shared_ptr<ClusterContext> ClusterContext::instance() {
  static std::shared_ptr<ClusterContext> cluster_instance = nullptr;
  if (cluster_instance == nullptr) {
    cluster_instance.reset(new (std::nothrow) ClusterContext());
    MS_EXCEPTION_IF_NULL(cluster_instance);
  }
  return cluster_instance;
}

bool ClusterContext::Initialize() {
  if (inited_) {
    MS_LOG(INFO) << "The cluster has been initialized.";
    return true;
  }

  // MindSpore cluster does not support PyNative mode.
  if (MsContext::GetInstance()->get_param<int>(MS_CTX_EXECUTION_MODE) == kPynativeMode) {
    MS_LOG(EXCEPTION) << "PyNative mode is not supported in MindSpore cluster.";
    return false;
  }

  // Step 1: Initialize cluster configuration.
  InitClusterConfig();

  // Step 2: Build network for this cluster. Every process will block in this method until networking is done.
  if (!BuildCluster()) {
    MS_EXCEPTION_IF_NULL(node_);
    if (!node_->Stop()) {
      MS_LOG(ERROR) << "Failed to stop node after the failure of BuildCluster";
      return false;
    }
    MS_LOG(ERROR) << "Building networking for " << node_role_ << " failed.";
    return false;
  }

  inited_ = true;
  finalized_ = false;
  return true;
}

bool ClusterContext::Finalize(uint32_t timeout) {
  if (finalized_) {
    return true;
  }
  // In some cases, one node calls the Finish function while other nodes don't. So timeout is acceptable.
  if (!node_->Finish(timeout)) {
    MS_LOG(WARNING) << "Finishing node " << node_role_ << " timeout.";
  }
  if (!node_->Stop()) {
    MS_LOG(ERROR) << "Failed to stop node " << node_role_;
    return false;
  }
  finalized_ = true;
  return true;
}

const std::shared_ptr<ps::core::Node> &ClusterContext::node() const { return node_; }

const std::string &ClusterContext::node_role() const { return node_role_; }

uint32_t ClusterContext::node_num(const std::string &node_role) {
  if (node_num_each_role_.count(node_role) == 0) {
    MS_LOG(EXCEPTION) << "Node role " << node_role << " is invalid.";
    return 0;
  }
  MS_LOG(INFO) << "Number of role " << node_role << " is " << node_num_each_role_[node_role];
  return node_num_each_role_[node_role];
}

bool ClusterContext::initialized() const { return inited_; }

void ClusterContext::InitClusterConfig() {
  InitNodeRole();
  InitSchedulerIp();
  InitSchedulerPort();
  ps::PSContext::instance()->set_ms_role(node_role_);
  ps::PSContext::instance()->set_worker_num(node_num_each_role_[kEnvRoleOfWorker]);
  ps::PSContext::instance()->set_server_num(node_num_each_role_[kEnvRoleOfServer]);
  ps::PSContext::instance()->set_scheduler_ip(scheduler_host_);
  ps::PSContext::instance()->set_scheduler_port(scheduler_port_);
  ps::PSContext::instance()->cluster_config().initial_worker_num = node_num_each_role_[kEnvRoleOfWorker];
  ps::PSContext::instance()->cluster_config().initial_server_num = node_num_each_role_[kEnvRoleOfServer];
  ps::PSContext::instance()->cluster_config().scheduler_host = scheduler_host_;
  ps::PSContext::instance()->cluster_config().scheduler_port = scheduler_port_;
}

bool ClusterContext::BuildCluster() {
  // Create node according to different role.
  if (node_role_ == kEnvRoleOfWorker) {
    node_ = std::make_shared<ps::core::WorkerNode>();
  } else if (node_role_ == kEnvRoleOfServer) {
    node_ = std::make_shared<ps::core::ServerNode>();
  } else if (node_role_ == kEnvRoleOfScheduler) {
    node_ = std::make_shared<ps::core::SchedulerNode>();
  } else {
    MS_LOG(EXCEPTION) << "The role " << node_role_ << " is invalid.";
    return false;
  }
  MS_EXCEPTION_IF_NULL(node_);

  RegisterEventCallback();
  if (!node_->Start()) {
    MS_LOG(ERROR) << "Building network failed.";
    return false;
  }
  MS_LOG(INFO) << "Cluster is successfully initialized.";
  return true;
}

void ClusterContext::InitNodeRole() {
  node_role_ = common::GetEnv(kEnvRole);
  if (kValidRoleName.count(node_role_) == 0) {
    MS_LOG(EXCEPTION) << "Role name " << node_role_ << " is invalid.";
    return;
  }

  if (common::GetEnv(kEnvWorkerNum).empty()) {
    node_num_each_role_[kEnvRoleOfWorker] = 0;
  } else {
    TRY_AND_CATCH_WITH_EXCEPTION(
      (node_num_each_role_[kEnvRoleOfWorker] = IntToUint(std::stoi(common::GetEnv(kEnvWorkerNum)))),
      "The environment variable MS_WORKER_NUM is invalid.");
  }

  if (common::GetEnv(kEnvServerNum).empty()) {
    node_num_each_role_[kEnvRoleOfServer] = 0;
  } else {
    TRY_AND_CATCH_WITH_EXCEPTION(
      (node_num_each_role_[kEnvRoleOfServer] = IntToUint(std::stoi(common::GetEnv(kEnvServerNum)))),
      "The environment variable MS_SERVER_NUM is invalid.");
  }
}

void ClusterContext::InitSchedulerIp() { scheduler_host_ = common::GetEnv(kEnvSchedulerHost); }

void ClusterContext::InitSchedulerPort() {
  TRY_AND_CATCH_WITH_EXCEPTION((scheduler_port_ = static_cast<uint16_t>(std::stoi(common::GetEnv(kEnvSchedulerPort)))),
                               "The environment variable MS_SCHED_PORT is invalid.");
  if (scheduler_port_ > kMaxPort) {
    MS_LOG(EXCEPTION) << "The port: " << scheduler_port_ << " is invalid.";
  }
}

void ClusterContext::RegisterEventCallback() {
  auto abstract_node = std::dynamic_pointer_cast<ps::core::AbstractNode>(node_);
  if (abstract_node != nullptr) {
    abstract_node->RegisterEventCallback(ps::core::ClusterEvent::SCHEDULER_TIMEOUT, [this]() {
      std::unique_lock<std::mutex> lock(finish_mutex_);
      MS_LOG(ERROR) << "Event SCHEDULER_TIMEOUT is captured.";
      try {
        MS_LOG(INFO) << "Start finalize cluster...";
        if (!Finalize()) {
          MS_LOG(EXCEPTION) << "Failed to finalize cluster.";
        }
        MS_LOG(INFO) << "Successfully finalize cluster.";

        MS_LOG(INFO) << "Start finalize collective communication...";
        if (!collective::CollectiveManager::instance()->Finalize()) {
          MS_LOG(EXCEPTION) << "Failed to finalize collective communication.";
        }
        MS_LOG(INFO) << "Successfully finalize collective communication.";

        MS_LOG(EXCEPTION)
          << "Event SCHEDULER_TIMEOUT is captured. This is because scheduler node is finalized or crashed.";
      } catch (std::exception &) {
        MsException::Instance().SetException();
      }
    });

    abstract_node->RegisterEventCallback(ps::core::ClusterEvent::NODE_TIMEOUT, [this]() {
      std::unique_lock<std::mutex> lock(finish_mutex_);
      MS_LOG(ERROR) << "Event NODE_TIMEOUT is captured.";
      try {
        MS_LOG(INFO) << "Start finalize cluster...";
        if (!Finalize()) {
          MS_LOG(EXCEPTION) << "Failed to finalize cluster.";
        }
        MS_LOG(INFO) << "Successfully finalize cluster.";

        MS_LOG(INFO) << "Start finalize collective communication...";
        if (!collective::CollectiveManager::instance()->Finalize()) {
          MS_LOG(EXCEPTION) << "Failed to finalize collective communication.";
        }
        MS_LOG(INFO) << "Successfully finalize collective communication.";

        MS_LOG(EXCEPTION) << "Event NODE_TIMEOUT is captured. This is because some nodes are finalized or crashed.";
      } catch (std::exception &) {
        MsException::Instance().SetException();
      }
    });
  }
}
}  // namespace cluster
}  // namespace distributed
}  // namespace mindspore
