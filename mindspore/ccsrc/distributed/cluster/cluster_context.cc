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
#include "distributed/cluster/topology/common.h"
#include "distributed/recovery/recovery_context.h"
#include "distributed/cluster/topology/compute_graph_node.h"
#include "distributed/cluster/topology/meta_server_node.h"
#include "distributed/collective/collective_manager.h"
#include "proto/topology.pb.h"
#include "utils/ms_context.h"
#include "ps/ps_context.h"
#include "ps/core/comm_util.h"
#include "include/common/debug/common.h"

namespace mindspore {
namespace distributed {
namespace cluster {
ClusterContext::ClusterContext()
    : inited_(false),
      finalized_(true),
      cluster_exit_with_exception_(false),
      node_num_each_role_({}),
      scheduler_host_(kLocalHost),
      scheduler_port_(kDefaultSchedPort),
      node_role_(""),
      cluster_config_(nullptr) {}

ClusterContext::~ClusterContext() {
  if (!finalized_) {
    try {
      const uint32_t timeout = 0;
      (void)Finalize(timeout);
    } catch (std::exception &) {
      MS_LOG(ERROR) << "Failed to finalize cluster context.";
    }
  }
  finalized_ = true;
}

std::shared_ptr<ClusterContext> ClusterContext::instance() {
  static std::once_flag init_flag;
  static std::shared_ptr<ClusterContext> cluster_instance = nullptr;
  std::call_once(init_flag, [&]() {
    if (cluster_instance == nullptr) {
      cluster_instance.reset(new (std::nothrow) ClusterContext());
      MS_EXCEPTION_IF_NULL(cluster_instance);
    }
  });

  return cluster_instance;
}

bool ClusterContext::Initialize() {
  if (inited_) {
    MS_LOG(INFO) << "The cluster has been initialized.";
    return true;
  }

  // Step 1: Initialize cluster configuration.
  InitClusterConfig();

  // Step 2: Build network for this cluster. Every process will block in this method until networking is done.
  if (!BuildCluster()) {
    MsException::Instance().CheckException();
    MS_LOG(ERROR) << "Building networking for " << node_role_ << " failed.";
    return false;
  }

  // Step 3: Initialize some modules for the node, e.g., actor route table proxy.
  if (!IsScheduler()) {
    // Only node which is not the scheduler needs route table proxy.
    auto cgn = std::dynamic_pointer_cast<distributed::cluster::topology::ComputeGraphNode>(node_base_);
    MS_EXCEPTION_IF_NULL(cgn);
    actor_route_table_proxy_ = std::make_shared<ActorRouteTableProxy>(cgn);
    MS_EXCEPTION_IF_NULL(actor_route_table_proxy_);
  }

  inited_ = true;
  finalized_ = false;
  return true;
}

bool ClusterContext::Finalize(uint32_t timeout) {
  if (finalized_) {
    return true;
  }
  MS_EXCEPTION_IF_NULL(node_base_);

  bool force = (timeout == 0);
  uint32_t interval = 5;
  while (!node_base_->Finalize(force)) {
    MS_LOG(WARNING) << "Retry to finalize the node...";
    (void)sleep(interval);
  }
  finalized_ = true;
  return true;
}

bool ClusterContext::IsScheduler() { return node_role_ == kEnvRoleOfScheduler; }

const std::shared_ptr<topology::NodeBase> &ClusterContext::node() const { return node_base_; }

const std::shared_ptr<topology::NodeBase> &ClusterContext::node_base() const { return node_base_; }

const std::string &ClusterContext::node_role() const { return node_role_; }

uint32_t ClusterContext::node_num(const std::string &node_role) {
  if (node_num_each_role_.count(node_role) == 0) {
    MS_LOG(EXCEPTION) << "Node role " << node_role << " is invalid.";
  }
  MS_LOG(INFO) << "Number of role " << node_role << " is " << node_num_each_role_[node_role];
  return node_num_each_role_[node_role];
}

uint32_t ClusterContext::node_num() const {
  uint32_t node_num = 0;
  for (auto iter = node_num_each_role_.begin(); iter != node_num_each_role_.end(); ++iter) {
    if (iter->first != kEnvRoleOfScheduler) {
      node_num += iter->second;
    }
  }
  return node_num;
}

bool ClusterContext::initialized() const { return inited_; }

const ActorRouteTableProxyPtr &ClusterContext::actor_route_table_proxy() const { return actor_route_table_proxy_; }

void ClusterContext::set_cluster_exit_with_exception() { cluster_exit_with_exception_ = true; }

bool ClusterContext::cluster_exit_with_exception() const { return cluster_exit_with_exception_; }

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
  // Get node_id from environment configuration or uuid generator.
  std::string node_id = common::GetEnv(kNodeId);
  if (node_id.length() == 0) {
    node_id = ps::core::CommUtil::GenerateUUID();
  }
  // Init the node according to the process role.
  size_t retry_num;
  if (node_role_ == kEnvRoleOfScheduler) {
    auto node_num = node_num_each_role_[kEnvRoleOfWorker] + node_num_each_role_[kEnvRoleOfServer];
    node_base_ = std::make_shared<topology::MetaServerNode>(node_id, node_role_, node_num);
    retry_num = topology::kMsnExecuteRetryNum;
  } else {
    node_base_ = std::make_shared<topology::ComputeGraphNode>(node_id, node_role_);
    retry_num = topology::kCgnExecuteRetryNum;
  }
  MS_EXCEPTION_IF_NULL(node_base_);
  RETURN_IF_FALSE_WITH_LOG(node_base_->Initialize(), "Failed to initialize the node.");

  // Check the state of topology construction.
  auto check_func = [this]() -> bool {
    // Check exception thrown by child threads in cgn or msn.
    MsException::Instance().CheckException();
    return this->node_base_->Initialized();
  };
  EXECUTE_WITH_RETRY(check_func, retry_num, topology::kExecuteInterval, "Topology build timed out.");

  MS_LOG(INFO) << "Cluster is successfully initialized.";

  if (node_role_ != kEnvRoleOfScheduler) {
    auto cgn = std::dynamic_pointer_cast<topology::ComputeGraphNode>(node_base_);
    MS_EXCEPTION_IF_NULL(cgn);
    std::string port_range_pb = cgn->GetMetadata(kNodePortRange);
    topology::NodePortRanges node_port_ranges;
    (void)node_port_ranges.ParseFromArray(port_range_pb.c_str(), SizeToInt(port_range_pb.size()));
    auto port_range = node_port_ranges.data().at(node_id);
    port_range_.first = port_range.min_port();
    port_range_.second = port_range.max_port();
    MS_LOG(INFO) << "Port range assigned for this node " << node_id << " is " << port_range_.first << " to "
                 << port_range_.second;
  }
  return true;
}

void ClusterContext::InitNodeRole() {
  node_role_ = common::GetEnv(kEnvRole);
  if (kValidRoleName.count(node_role_) == 0) {
    MS_LOG(EXCEPTION) << "Role name '" << node_role_ << "' is invalid. " << kDetailedFailureReason;
  }

  // If node role is valid, judge the execution mode.
  // MindSpore cluster does not support PyNative mode.
  if (MsContext::GetInstance()->get_param<int>(MS_CTX_EXECUTION_MODE) == kPynativeMode) {
    MS_LOG(EXCEPTION) << "PyNative mode is not supported in dynamic networking for now.";
  }

  if (common::GetEnv(kEnvWorkerNum).empty()) {
    if (node_role_ == kEnvRoleOfWorker) {
      MS_LOG(EXCEPTION) << "Please set env 'WORKER_NUM' to a number greater than 0.";
    }
    node_num_each_role_[kEnvRoleOfWorker] = 0;
  } else {
    TRY_AND_CATCH_WITH_EXCEPTION(
      (node_num_each_role_[kEnvRoleOfWorker] = IntToUint(std::stoi(common::GetEnv(kEnvWorkerNum)))),
      "The environment variable MS_WORKER_NUM is invalid.");
  }

  // MS_PSERVER is supported for now. It should be deprecated after we use cluster for distributed training.
  if (common::GetEnv(kEnvServerNum).empty()) {
    if (node_role_ == kEnvRoleOfServer || node_role_ == kEnvRoleOfPServer) {
      MS_LOG(EXCEPTION) << "Please set env 'SERVER_NUM' to a number greater than 0.";
    }
    node_num_each_role_[kEnvRoleOfServer] = 0;
    node_num_each_role_[kEnvRoleOfPServer] = 0;
  } else {
    TRY_AND_CATCH_WITH_EXCEPTION(
      (node_num_each_role_[kEnvRoleOfServer] = IntToUint(std::stoi(common::GetEnv(kEnvServerNum)))),
      "The environment variable MS_SERVER_NUM is invalid.");
    TRY_AND_CATCH_WITH_EXCEPTION(
      (node_num_each_role_[kEnvRoleOfPServer] = IntToUint(std::stoi(common::GetEnv(kEnvServerNum)))),
      "The environment variable MS_SERVER_NUM is invalid.");
  }
}

void ClusterContext::InitSchedulerIp() {
  scheduler_host_ = common::GetEnv(kEnvSchedulerHost);
  if (scheduler_host_.empty()) {
    MS_LOG(EXCEPTION) << kEnvSchedulerHost << " is empty. " << kEnvSchedulerHost;
  }
}

void ClusterContext::InitSchedulerPort() {
  TRY_AND_CATCH_WITH_EXCEPTION((scheduler_port_ = static_cast<uint16_t>(std::stoi(common::GetEnv(kEnvSchedulerPort)))),
                               "The environment variable MS_SCHED_PORT is invalid.");
  if (scheduler_port_ > kMaxPort) {
    MS_LOG(EXCEPTION) << "The port: " << scheduler_port_ << " is invalid.";
  }
}
}  // namespace cluster
}  // namespace distributed
}  // namespace mindspore
