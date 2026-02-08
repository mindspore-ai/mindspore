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

#ifndef MINDSPORE_CCSRC_DISTRIBUTED_CLUSTER_CLUSTER_CONTEXT_H_
#define MINDSPORE_CCSRC_DISTRIBUTED_CLUSTER_CLUSTER_CONTEXT_H_

#include <map>
#include <set>
#include <mutex>
#include <string>
#include <memory>
#include <atomic>
#include <utility>
#include <optional>
#include "include/cluster/topology/constants.h"
#include "utils/log_adapter.h"
#include "utils/ms_utils.h"
#include "include/cluster/topology/node_base.h"
#include "include/cluster/visible.h"

namespace mindspore {
namespace ps::core {
struct ClusterConfig;
}  // namespace ps::core
namespace distributed {
namespace cluster {
// The environment variable name represents the node id of a certain process(compute graph node).
constexpr char kNodeId[] = "MS_NODE_ID";
// The Indentation to format json information.
constexpr int kJsonIndentation = 4;
class ActorRouteTableProxy;
class TCPStoreClient;
using TCPStoreClientPtr = std::shared_ptr<TCPStoreClient>;
// Node role based cluster built by MindSpore communication framework.
class CLUSTER_EXPORT ClusterContext {
 public:
  ~ClusterContext();
  DISABLE_COPY_AND_ASSIGN(ClusterContext)
  static std::shared_ptr<ClusterContext> instance();

  // Initialize the cluster configuration and build network.
  bool Initialize();
  bool Initialize(std::optional<std::string> url, int64_t timeout, uint32_t world_size, uint32_t node_id,
                  TCPStoreClientPtr store);

  // Finalize the cluster and process exits. If timeout is set to UINT32_MAX, this method will block without timeout.
  bool Finalize(uint32_t timeout = kDefaultFinishTimeout);

  // Stop threads when exception happens to avoid unpredictable behavior.
  void StopThreadsOnException();

  // Return whether this node is the scheduler node.
  // In a cluster, the scheduler node is special because it's responsible for building network.
  bool IsScheduler();

  // Return node object of this process.
  const std::shared_ptr<topology::NodeBase> &node() const;

  // Return the shadow node.
  const std::shared_ptr<topology::NodeBase> &node_base() const;

  // Return node role in this cluster.
  const std::string &node_role() const;

  // Returns total number of the specified node role. This is used as the group size of this node role.
  uint32_t node_num(const std::string &node_role);

  // Returns the total number of various role nodes.
  uint32_t node_num() const;

  // Return cluster is initialized.
  bool initialized() const;

  // Return actor route proxy for AbstractNode.
  const std::shared_ptr<ActorRouteTableProxy> &actor_route_table_proxy() const;

  // Get and set whether this process exits with exception.
  void set_cluster_exit_with_exception();
  bool cluster_exit_with_exception() const;

  // Set whether enable cross cluster communication.
  void set_enable_cross_cluster(bool enable_cross_cluster) { enable_cross_cluster_ = enable_cross_cluster; }

  // Return whether enable cross cluster communication.
  bool enable_cross_cluster() const { return enable_cross_cluster_; }

  bool is_init_by_store() const { return init_by_store_; }

  // Return server range of this node.
  const std::pair<uint32_t, uint32_t> &port_range() const { return port_range_; }

 private:
  ClusterContext();

  // This initializing cluster configurations. They can be exported by environment variables, set by python API or
  // configuration file.
  void InitClusterConfig();

  // Build the cluster with other processes. This method will not return until the networking is done.
  bool BuildCluster();

  // Load the cluster configuration like worker number, server number and etc.
  void InitNodeRole();
  void InitSchedulerIp();
  void InitSchedulerPort();

  // Check whether the cluster enables the cross cluster communication.
  bool IsEnableCrossCluster();

  // After cluster is successfully built, some post process should be done. For example, port range assignment and
  // client ip set, etc.
  void PostProcess();

  // The flag that whether this cluster context instance is already initialized.
  std::atomic_bool inited_;

  // The flag that whether this cluster context instance is already finalized.
  std::atomic_bool finalized_;

  // The mutex about exiting status of this node.
  std::mutex finish_mutex_;

  // Whether the process in this cluster exits with any python exception.
  bool cluster_exit_with_exception_;

  // Node role to role number map.
  std::map<std::string, uint32_t> node_num_each_role_;

  // Scheduler information.
  std::string scheduler_host_;
  uint16_t scheduler_port_;

  // The compute graph node or meta server node according to the configuration of this process.
  std::shared_ptr<topology::NodeBase> node_base_;

  TCPStoreClientPtr tcp_store_client_;

  // Node id of this process in the cluster.
  std::string node_id_;

  // The role of this process in the cluster.
  std::string node_role_;

  // The configuration of this cluster.
  std::unique_ptr<ps::core::ClusterConfig> cluster_config_;

  // The actor route table proxy. It only created in abstract nodes because scheduler does not use proxy.
  std::shared_ptr<ActorRouteTableProxy> actor_route_table_proxy_;

  std::pair<uint32_t, uint32_t> port_range_;

  // Indicate whether enable cross cluster communication.
  bool enable_cross_cluster_;

  // Indicate whether initialize with TcpStore.
  bool init_by_store_{false};
};
}  // namespace cluster
}  // namespace distributed
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DISTRIBUTED_CLUSTER_CLUSTER_CONTEXT_H_
