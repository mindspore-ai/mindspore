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

#include <vector>
#include "distributed/cluster/cluster_context.h"
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
      node_role_(kEnvRoleOfWorker),
      cluster_config_(nullptr) {}

ClusterContext::~ClusterContext() {
  if (!finalized_) {
    Finalize();
  }
}

std::shared_ptr<ClusterContext> ClusterContext::instance() {
  static std::shared_ptr<ClusterContext> cluster_instance = nullptr;
  if (cluster_instance == nullptr) {
    cluster_instance.reset(new (std::nothrow) ClusterContext());
    MS_EXCEPTION_IF_NULL(cluster_instance);
  }
  return cluster_instance;
}

void ClusterContext::Initialize() {
  if (inited_) {
    MS_LOG(INFO) << "The cluster has been initialized.";
    return;
  }

  // Step 1: Initialize cluster configuration.
  InitClusterConfig();

  // Step 2: Build network for this cluster. Every process will block in this method until networking is done.
  if (!BuildCluster()) {
    MS_LOG(EXCEPTION) << "Building networking for " << node_role_ << " failed.";
    return;
  }

  inited_ = true;
  finalized_ = false;
}

void ClusterContext::Finalize() {
  if (finalized_) {
    return;
  }
  // In some cases, one node calls the Finish function while other nodes don't. So timeout is acceptable.
  if (!node_->Finish()) {
    MS_LOG(WARNING) << "Finishing node " << node_role_ << " timeout.";
  }
  if (!node_->Stop()) {
    MS_LOG(ERROR) << "Failed to stop node " << node_role_;
    return;
  }
  finalized_ = true;
  wait_finish_cond_.notify_all();
}

std::string ClusterContext::node_role() const { return node_role_; }

void ClusterContext::InitClusterConfig() {
  InitNodeRole();
  InitSchedulerIp();
  InitSchedulerPort();
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
    MS_LOG(EXCEPTION) << "Building network failed.";
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

void ClusterContext::InitSchedulerIp() {
  scheduler_host_ = common::GetEnv(kEnvSchedulerHost);
  if (scheduler_host_ != kLocalHost) {
    MS_LOG(EXCEPTION) << "Scheduler IP should be 127.0.0.1";
  }
}

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
      MS_LOG(ERROR) << "Event SCHEDULER_TIMEOUT is captured.";
      Finalize();
      try {
        MS_LOG(EXCEPTION)
          << "Event SCHEDULER_TIMEOUT is captured. This is because scheduler node is finalized or crashed.";
      } catch (std::exception &) {
        MsException::Instance().SetException();
      }
    });

    abstract_node->RegisterEventCallback(ps::core::ClusterEvent::NODE_TIMEOUT, [this]() {
      MS_LOG(ERROR) << "Event NODE_TIMEOUT is captured.";
      Finalize();
      try {
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
