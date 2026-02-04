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

#include "include/cluster/topology/cluster_context.h"

#include <mutex>
#include <vector>
#include <string>
#include <memory>

#include "include/cluster/topology/common.h"
#include "include/cluster/topology/compute_graph_node.h"
#include "cluster/topology/meta_server_node.h"
#include "include/cluster/topology/actor_route_table_proxy.h"
#include "include/cluster/rpc/tcp_store.h"
#include "include/cluster/rpc/tcp_node.h"
#include "cluster/topology/utils.h"
#include "include/runtime/hardware_abstract/collective/collective_manager.h"
#include "proto/topology.pb.h"
#include "utils/ms_context.h"
#include "utils/file_utils.h"
#include "utils/distributed_meta.h"
#include "nlohmann/json.hpp"
#include "include/cluster/topology/ps_context.h"
#include "cluster/rpc/core/comm_util.h"
#include "cluster/rpc/core/cluster_config.h"
#include "include/utils/common.h"
#include "utils/ms_exception.h"

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
      node_id_(""),
      node_role_(""),
      cluster_config_(nullptr),
      enable_cross_cluster_(false) {}

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
  } else if (VL_DISTRIBUTED >= g_ms_vlog_level_from && VL_DISTRIBUTED <= g_ms_vlog_level_to) {
    MS_VLOG(VL_DISTRIBUTED)
      << "The environment variable 'VLOG_v' is set to " << common::GetEnv("VLOG_v") << " includes " << VL_DISTRIBUTED
      << ", scheduler is now dumping workers' metadata information. In large-scale cluster scenarios, "
         "this process may take some time, but it will not affect distributed tasks.";
    auto msn = std::dynamic_pointer_cast<distributed::cluster::topology::MetaServerNode>(node_base_);
    MS_EXCEPTION_IF_NULL(msn);
    try {
      nlohmann::json servers = nlohmann::json::array();
      for (auto &node : msn->GetComputeGraphNodes()) {
        std::shared_ptr<topology::NodeInfo> &node_info = node.second;
        nlohmann::json device;
        device["role"] = node_info->role;
        device["node_id"] = node_info->node_id;
        device["device_id"] = node_info->device_id;
        device["rank_id"] = node_info->rank_id;

        auto it = std::find_if(servers.begin(), servers.end(),
                               [&](const nlohmann::json &server) { return server["host_ip"] == node_info->host_ip; });
        if (it != servers.end()) {
          it->at("device").push_back(device);
        } else {
          nlohmann::json server;
          server["device"] = nlohmann::json::array();
          server["host_ip"] = node_info->host_ip;
          server["host_name"] = node_info->host_name;
          server["device"].push_back(device);
          servers.push_back(server);
        }
      }
      MS_VLOG(VL_DISTRIBUTED) << "Output metadata for compute graph nodes as json format:\n"
                              << servers.dump(kJsonIndentation);
    } catch (const std::exception &e) {
      MS_LOG(WARNING) << "Failed to dump metadata to json format. Json error: " << e.what();
    }
  }

  inited_ = true;
  finalized_ = false;
  return true;
}

bool ClusterContext::Initialize(std::optional<std::string> url, int64_t timeout, uint32_t world_size, uint32_t node_id,
                                TCPStoreClientPtr store) {
  if (inited_) {
    MS_LOG(INFO) << "The cluster has been initialized.";
    return true;
  }

  node_id_ = std::to_string(node_id).c_str();
  common::SetEnv(kEnvRole, kEnvRoleOfWorker);
  common::SetEnv(kEnvWorkerNum, std::to_string(world_size).c_str());
  common::SetEnv(kNodeId, std::to_string(node_id).c_str());

  std::string ip;
  int64_t port;
  if (url.has_value()) {
    if (!cluster::Utils::ParseTcpUrlForIpv4(url.value(), &ip, &port)) {
      return false;
    }
    const bool is_master = (node_id == 0);
    tcp_store_client_ = std::make_shared<cluster::TCPStoreClient>(ip, port, is_master, timeout, world_size);
    MS_EXCEPTION_IF_NULL(tcp_store_client_);
    MS_LOG(INFO) << "Initialize cluster using init_method, a default TcpStore is created with ip: " << ip
                 << ", port: " << port << ", world_size: " << world_size;
  } else {
    tcp_store_client_ = store;
    MS_EXCEPTION_IF_NULL(tcp_store_client_);
    ip = tcp_store_client_->ip();
    port = tcp_store_client_->port();
    tcp_store_client_->set_timeout(timeout);
    int64_t tcp_world_size = tcp_store_client_->world_size();
    if (tcp_world_size != world_size) {
      MS_LOG(EXCEPTION) << "Initialize cluster using existing store with world_size [" << tcp_world_size
                        << "] is inconsistent with init_process_group provided world_size [" << world_size << "].";
    }
    MS_LOG(INFO) << "Initialize cluster using existing store with ip: " << ip << ", port: " << port
                 << ", world_size: " << world_size;
  }
  common::SetEnv(kEnvSchedulerHost, ip.c_str());
  common::SetEnv(kEnvSchedulerPort, std::to_string(port).c_str());
  InitClusterConfig();

  // Initializing with TcpStore makes client_node as cgn.
  node_base_ = tcp_store_client_->client_node_;
  MS_EXCEPTION_IF_NULL(node_base_);

  tcp_store_client_->client_node_->set_rank_id(node_id);
  MS_LOG(WARNING) << "This node " << node_id_ << " rank id: " << tcp_store_client_->client_node_->rank_id();

  // 2. Set this node's client ip address in this cluster.
  const std::string &client_ip_in_cluster = tcp_store_client_->client_node_->client_ip();
  MS_LOG(INFO) << "Client ip address in this cluster of this client_node is " << client_ip_in_cluster;
  (void)common::SetEnv(kEnvWorkerIp, client_ip_in_cluster.c_str());

  // 3. Set port range of this node.
  NodeRolePortAssignment port_assignment =
    (tcp_store_client_->client_node_->role() == kEnvRoleOfWorker) ? kWorkerPortAssignment : kServerPortAssignment;
  uint32_t start_port = port_assignment.start_port;
  uint32_t port_range = port_assignment.port_range;
  uint32_t max_node_num = port_assignment.max_node_num;
  port_range_.first =
    start_port + (port_range / max_node_num) * (tcp_store_client_->client_node_->rank_id() % max_node_num);
  port_range_.second = port_range_.first + (port_range / max_node_num) - 1;
  MS_LOG(INFO) << "Assigned for this worker port range is " << port_range_.first << " to " << port_range_.second;

  init_by_store_ = true;
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
    MS_LOG(INFO)
      << "This log means the cluster is successfully created. Retry to finalize the node and exit cluster...";
    (void)sleep(interval);
  }
  finalized_ = true;
  return true;
}

void ClusterContext::StopThreadsOnException() {
  if (node_role_ != kEnvRoleOfScheduler) {
    auto cgn = std::dynamic_pointer_cast<topology::ComputeGraphNode>(node_base_);
    if (cgn == nullptr) {
      return;
    }
    cgn->StopHeartBeatThread();
  }
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
  PROF_START(BuildCluster);
  // Get node_id from environment configuration or uuid generator.
  node_id_ = common::GetEnv(kNodeId);
  if (node_id_.length() == 0) {
    node_id_ = ps::core::CommUtil::GenerateUUID();
  }
  // Init the node according to the process role.
  if (node_role_ == kEnvRoleOfScheduler) {
    auto node_num = node_num_each_role_[kEnvRoleOfWorker] + node_num_each_role_[kEnvRoleOfServer];
    node_base_ = std::make_shared<topology::MetaServerNode>(node_id_, node_role_, node_num);
  } else {
    node_base_ = std::make_shared<topology::ComputeGraphNode>(node_id_, node_role_);
  }
  MS_EXCEPTION_IF_NULL(node_base_);
  // For cgn, 'Initialize' will block until it connect to msn, or time out.
  RETURN_IF_FALSE_WITH_LOG(node_base_->Initialize(), "Failed to initialize the node.");

  // Check the state of topology construction.
  auto check_func = [this]() -> bool {
    // Check exception thrown by child threads in cgn or msn.
    MsException::Instance().CheckException();
    return this->node_base_->Initialized();
  };
  size_t retry_num = GetRetryNumBasedOnScale(node_base_->topo_timeout(), kExecuteInterval);
  EXECUTE_WITH_RETRY(check_func, retry_num, kExecuteInterval, "Topology build timed out.");
  PROF_END(BuildCluster);

  MS_LOG(WARNING) << "Cluster is successfully initialized.";

  PROF_START(PostBuildCluster);
  PostProcess();
  PROF_END(PostBuildCluster);
  return true;
}

void ClusterContext::InitNodeRole() {
  node_role_ = common::GetEnv(kEnvRole);
  if (kValidRoleName.count(node_role_) == 0) {
    MS_LOG(EXCEPTION) << "Role name '" << node_role_ << "' is invalid. " << kDetailedFailureReason;
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
    MS_LOG(EXCEPTION) << kEnvSchedulerHost << " is empty.";
  }
}

void ClusterContext::InitSchedulerPort() {
  TRY_AND_CATCH_WITH_EXCEPTION((scheduler_port_ = static_cast<uint16_t>(std::stoi(common::GetEnv(kEnvSchedulerPort)))),
                               "The environment variable MS_SCHED_PORT is invalid.");
  if (scheduler_port_ > kMaxPort) {
    MS_LOG(EXCEPTION) << "The port: " << scheduler_port_ << " is invalid.";
  }
}

bool ClusterContext::IsEnableCrossCluster() {
  constexpr char kRankTableClusterList[] = "cluster_list";
  std::string rank_table_file_path = common::GetEnv("RANK_TABLE_FILE");
  if (rank_table_file_path.empty()) {
    return false;
  }
  auto realpath = FileUtils::GetRealPath(rank_table_file_path.c_str());
  if (!realpath.has_value()) {
    MS_LOG(WARNING) << "Failed to get real path.";
    return false;
  }
  std::ifstream jsonFile(realpath.value(), std::ifstream::in);
  if (!jsonFile.is_open()) {
    MS_LOG(WARNING)
      << "Failed to open rank table file. This may be because the path of rank table file is incorrect or the access"
      << " of json file is not permitted.";
    return false;
  }
  try {
    nlohmann::json rank_table_file_data = nlohmann::json::parse(jsonFile);
    if (rank_table_file_data.is_null()) {
      MS_LOG(WARNING) << "Failed to read data from rank table file.";
      return false;
    }
    return rank_table_file_data.contains(kRankTableClusterList);
  } catch (const std::exception &e) {
    MS_LOG(WARNING) << "Rank table file is incorrect. Json error: " << e.what();
    return false;
  }
}

void ClusterContext::PostProcess() {
  if (node_role_ == kEnvRoleOfScheduler) {
    return;
  }

  auto cgn = std::dynamic_pointer_cast<topology::ComputeGraphNode>(node_base_);
  MS_EXCEPTION_IF_NULL(cgn);
  MS_LOG(INFO) << "Start post processing for computing graph nodes.";

  // 1. Get new rank id from meta server node because it may be reassigned.
  auto node_id = common::GetEnv("MS_NODE_ID");
  if (node_id.empty() || !common::IsStrNumeric(node_id) || !common::GetEnv("RANK_TABLE_FILE").empty()) {
    MS_LOG(INFO) << "MS_NODE_ID set to this process is " << node_id
                 << " and it's not numeric. Or ranktable file is set. Need to get reassigned rank id from scheduler.";
    std::string final_rank_id = cgn->GetMetadata(node_role_ + node_id_);
    if (!final_rank_id.empty()) {
      cgn->set_rank_id(static_cast<uint32_t>(std::atoi(final_rank_id.c_str())));
    } else {
      MS_LOG(WARNING) << "This node could be redundant and is not successfully registered.";
    }
  }
  MS_LOG(WARNING) << "This node " << node_id_ << " rank id: " << cgn->rank_id();

  // 2. Set this node's client ip address in this cluster.
  const std::string &client_ip_in_cluster = cgn->client_ip();
  MS_LOG(INFO) << "Client ip address in this cluster of this compute graph node is " << client_ip_in_cluster;
  (void)common::SetEnv(kEnvWorkerIp, client_ip_in_cluster.c_str());

  // 3. Set port range of this node.
  NodeRolePortAssignment port_assignment =
    (cgn->role() == kEnvRoleOfWorker) ? kWorkerPortAssignment : kServerPortAssignment;
  uint32_t start_port = port_assignment.start_port;
  uint32_t port_range = port_assignment.port_range;
  uint32_t max_node_num = port_assignment.max_node_num;
  port_range_.first = start_port + (port_range / max_node_num) * (cgn->rank_id() % max_node_num);
  port_range_.second = port_range_.first + (port_range / max_node_num) - 1;
  MS_LOG(INFO) << "Assigned for this worker port range is " << port_range_.first << " to " << port_range_.second;

  // 4. Set whether enable cross cluster communication.
  if (IsEnableCrossCluster()) {
    MS_LOG(WARNING) << "This node enable the cross cluster communication.";
    enable_cross_cluster_ = true;
    DistributedMeta::GetInstance()->set_enable_cross_cluster(enable_cross_cluster_);
  }
}
}  // namespace cluster
}  // namespace distributed
}  // namespace mindspore
