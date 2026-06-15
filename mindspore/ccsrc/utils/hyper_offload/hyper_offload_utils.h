/**
 * Copyright 2025-2026 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_UTILS_HYPER_OFFLOAD_HYPER_OFFLOAD_UTILS_H_
#define MINDSPORE_CCSRC_UTILS_HYPER_OFFLOAD_HYPER_OFFLOAD_UTILS_H_

#include <cstdint>
#include <map>
#include <string>
#include <memory>
#include <utility>
#include <optional>
#include <set>
#include <vector>

#include "ir/anf.h"
#include "include/utils/anfalgo.h"
#include "include/utils/visible.h"

namespace mindspore {
namespace utils {
namespace hyper_offload {
using KernelWithIndex = common::KernelWithIndex;
using HandleNewNodeCallback = std::function<void(const AnfNodePtr &new_node, const AnfNodePtr &context_node)>;

constexpr auto kPeakFreeNodeAttr = "peak_free_data_after_node";

// Get the stable in-process address value of a node.
COMMON_EXPORT uintptr_t GetNodeAddress(const AnfNodePtr &node);

// Indicate the type of data for FreeData or AssignData node.
constexpr size_t kDeviceType = 0;
constexpr size_t kHostType = 1;

// Indicate the stream id for send and recv node.
constexpr size_t kCalculateStreamId = 0;
constexpr size_t kTransferStreamId = 1;

// Attribute key for hyper offload nodes. HyperOffload nodes for the same data should have the same value.
constexpr auto kHyperOffloadId = "hyper_offload_data_id";
// Attribute key for unique id for pair of send and recv node.
constexpr auto kHyperOffloadEventId = "hyper_offload_event_id";
// Attribute key for stream id for send and recv node.
constexpr auto kHyperOffloadEventStreamId = "hyper_offload_event_stream_id";
// Attribute key for event type for event node.
constexpr auto kHyperOffloadEventNodeType = "hyper_offload_event_node_type";
// User data keys for backend runtime metadata consumed by common hyper offload utils.
constexpr auto kHyperOffloadRefOutputInputMap = "hyper_offload_ref_output_input_map";
constexpr auto kHyperOffloadInlineBranchName = "hyper_offload_inline_branch_name";
constexpr auto kHyperOffloadSummaryOutputs = "hyper_offload_summary_outputs";
constexpr auto kHyperOffloadOutputMemSizes = "hyper_offload_output_mem_sizes";
constexpr auto kHyperOffloadKernelModOutputSizes = "hyper_offload_kernel_mod_output_sizes";
constexpr auto kHyperOffloadWorkspaceSize = "hyper_offload_workspace_size";
constexpr auto kHyperOffloadOutputCount = "hyper_offload_output_count";

const PrimitivePtr kFreeDataPrimitive = std::make_shared<Primitive>("FreeData");
const PrimitivePtr kAssignDataPrimitive = std::make_shared<Primitive>("AssignData");

enum EventNodeType {
  kSendRecvBeforeD2H,
  kSendRecvAfterD2H,
  kSendRecvAfterH2D,
  kFreeAndSwitchData,
};

/**
 * @brief Record one data node and all of its users in execution order.
 */
struct UserInfo {
  KernelWithIndex node;
  std::vector<KernelWithIndex> users;
};
using UserInfoList = std::vector<UserInfo>;

/**
 * @brief Describe replacement range for one offloaded data node.
 */
struct ReplaceInfo {
  KernelWithIndex replace_start_node;
  std::vector<KernelWithIndex> replace_rest_nodes;
  KernelWithIndex free_data_after_node = KernelWithIndex();
};
using ReplaceInfoList = std::vector<ReplaceInfo>;

/**
 * @brief Offload decision result for a single data node.
 */
struct OffloadInfo {
  KernelWithIndex data_node;
  ReplaceInfoList replace_info_list;
  size_t optimized_release_distance = 0;
};
using OffloadInfoList = std::vector<OffloadInfo>;

/**
 * @brief HyperOffload compile-time and runtime configuration singleton.
 */
class COMMON_EXPORT HyperOffloadConfig {
 public:
  // Get the singleton instance of HyperOffloadConfig.
  static HyperOffloadConfig &GetInstance();

  HyperOffloadConfig(const HyperOffloadConfig &) = delete;
  void operator=(const HyperOffloadConfig &) = delete;

  bool enable_hyper_offload() const { return enable_hyper_offload_; }
  bool enable_hyper_offload_peak_strategy() const { return enable_hyper_offload_peak_strategy_; }
  size_t select_distance() const { return select_distance_; }
  size_t select_num() const { return select_num_; }
  size_t prefetch_distance() const { return prefetch_distance_; }
  size_t release_distance() const { return release_distance_; }
  size_t max_device_memory() const { return max_device_memory_; }
  double score_weight_size() const { return score_weight_size_; }
  double score_weight_cover_start() const { return score_weight_cover_start_; }
  double score_weight_cover_end() const { return score_weight_cover_end_; }
  double score_weight_neighbor_impact() const { return score_weight_neighbor_impact_; }
  const std::set<int64_t> &enabled_stages() const { return enabled_stages_; }
  const std::string &offload_percentage() const { return offload_percentage_; }
  double d2h_bandwidth_gbps() const { return d2h_bandwidth_gbps_; }
  double h2d_bandwidth_gbps() const { return h2d_bandwidth_gbps_; }
  double default_kernel_time_us() const { return default_kernel_time_us_; }
  bool enable_adaptive_release_distance() const { return enable_adaptive_release_distance_; }
  bool enable_adaptive_prefetch_distance() const { return enable_adaptive_prefetch_distance_; }

 private:
  HyperOffloadConfig();

  static constexpr double kDefaultD2HBandwidthGbps = 20.0;
  static constexpr double kDefaultH2DBandwidthGbps = 55.0;
  static constexpr double kDefaultKernelTimeUs = 400.0;

  bool enable_hyper_offload_{false};
  bool enable_hyper_offload_peak_strategy_{false};
  size_t select_distance_{0};
  size_t select_num_{0};
  size_t prefetch_distance_{0};
  size_t release_distance_{0};
  size_t max_device_memory_{0};
  double score_weight_size_{1.0};
  double score_weight_cover_start_{1.0};
  double score_weight_cover_end_{1.0};
  double score_weight_neighbor_impact_{1.0};
  std::set<int64_t> enabled_stages_;
  std::string offload_percentage_;
  double d2h_bandwidth_gbps_{kDefaultD2HBandwidthGbps};
  double h2d_bandwidth_gbps_{kDefaultH2DBandwidthGbps};
  double default_kernel_time_us_{kDefaultKernelTimeUs};
  bool enable_adaptive_release_distance_{false};
  bool enable_adaptive_prefetch_distance_{false};
};

// Check if a node is a D2H (Device to Host) node.
COMMON_EXPORT bool IsD2HNode(const AnfNodePtr &node);
// Check if a node is an H2D (Host to Device) node.
COMMON_EXPORT bool IsH2DNode(const AnfNodePtr &node);
// Check if a node is a View or Inplace node.
COMMON_EXPORT bool IsViewOrInplaceNode(const AnfNodePtr &node);
// Check if a node is a View node.
COMMON_EXPORT bool IsViewNode(const AnfNodePtr &node);
// Check if a node is a Replace node.
COMMON_EXPORT bool IsInplaceNode(const AnfNodePtr &node);
// Get the original data node by tracing back through view nodes.
COMMON_EXPORT KernelWithIndex GetRealDataNodeWithIndex(const AnfNodePtr &node);
// Check if a user info indicates a valid offload candidate.
COMMON_EXPORT bool IsValidOffloadNode(const UserInfo &user_info);
// Collect all users for a list of nodes.
COMMON_EXPORT UserInfoList CollectAllNodeUsers(const CNodePtrList &nodes);
// Check if a pipeline stage is enabled.
COMMON_EXPORT bool IsStageEnabled(int64_t stage_id, const std::set<int64_t> &enabled_stages);
// Return ref output indexes whose input is the given CNode raw input index.
COMMON_EXPORT std::vector<size_t> GetRefOutputIndexesForInput(const KernelWithIndex &user_node);
// Check whether the given CNode raw input index participates in an output-input ref pair.
COMMON_EXPORT bool IsRefInputUser(const KernelWithIndex &user_node);
// Check whether the given output participates in an output-input ref pair.
COMMON_EXPORT bool IsRefOutput(const KernelWithIndex &node_output);
// Get the execution-order range of the inline control-flow branch that a node belongs to.
COMMON_EXPORT std::optional<std::pair<size_t, size_t>> GetInlineBranchIndexRange(const FuncGraphPtr &graph,
                                                                                 const CNodePtrList &exec_order,
                                                                                 const AnfNodePtr &node);
// Get the attach slot immediately before the inline branch that a node belongs to.
COMMON_EXPORT std::optional<size_t> GetInlineBranchStartAttachIndex(const FuncGraphPtr &graph,
                                                                    const CNodePtrList &exec_order,
                                                                    const AnfNodePtr &node);
// Build a ToHost (D2H) node.
COMMON_EXPORT CNodePtr BuildToHostNode(const FuncGraphPtr &graph, const AnfNodePtr &data_node,
                                       const HandleNewNodeCallback &callback = nullptr);
// Build an Inplace ToHost node.
COMMON_EXPORT CNodePtr BuildInplaceToHostNode(const FuncGraphPtr &graph, const AnfNodePtr &data_node,
                                              const AnfNodePtr &update_node, const AnfNodePtr &depend_node,
                                              const HandleNewNodeCallback &callback = nullptr);
// Build a ToDevice (H2D) node.
COMMON_EXPORT CNodePtr BuildToDeviceNode(const FuncGraphPtr &graph, const AnfNodePtr &data_node,
                                         const HandleNewNodeCallback &callback = nullptr);
// Find a parameter by its reference name.
COMMON_EXPORT std::optional<ParameterPtr> FindParameterByRefName(const FuncGraphPtr &graph, const std::string &name);
// Get the remote parameter associated with a node.
COMMON_EXPORT std::optional<ParameterPtr> GetRemoteParameter(const FuncGraphPtr &graph, const AnfNodePtr &node);
// Update the input of a node at a specific index.
COMMON_EXPORT void UpdateNodeInput(const CNodePtr &node, const size_t input_index, const AnfNodePtr &new_input_node);
COMMON_EXPORT HandleNewNodeCallback CreateContextCallback(HandleNewNodeCallback original_callback,
                                                          const AnfNodePtr &context_node);
// Build a map of user index range (start, end) for each node output in the execution order.
COMMON_EXPORT std::map<KernelWithIndex, std::pair<size_t, size_t>> BuildUserIndexRangeMap(const CNodePtrList &nodes);

/**
 * @brief Create send/recv event node pairs and attach event metadata.
 */
class COMMON_EXPORT EventNodeFactory {
 public:
  explicit EventNodeFactory(const FuncGraphPtr &fg) : fg_(fg), event_id_(0) {}
  std::pair<CNodePtr, CNodePtr> CreateSendRecvPair(size_t send_stream_id, size_t recv_stream_id, EventNodeType type,
                                                   const AnfNodePtr &data_node);

 private:
  FuncGraphPtr fg_;
  uint32_t event_id_;
};

// Build FreeData node with kDeviceType or kHostType.
COMMON_EXPORT AnfNodePtr BuildFreeDataNode(const FuncGraphPtr &fg, const AnfNodePtr &node, size_t type);
// Build AssignData node with kDeviceType or kHostType.
COMMON_EXPORT AnfNodePtr BuildAssignDataNode(const FuncGraphPtr &fg, const AnfNodePtr &node,
                                             const AnfNodePtr &data_node, size_t type);
// Get the HyperOffloadId from a node.
COMMON_EXPORT std::optional<size_t> GetHyperOffloadId(const AnfNodePtr &node);
// Get the node type (kDeviceType or kHostType) for FreeData or AssignData node.
COMMON_EXPORT std::optional<size_t> GetFreeOrAssignNodeType(const AnfNodePtr &node);
// Get the stream id for Send or Recv node.
COMMON_EXPORT std::optional<size_t> GetSendOrRecvStreamId(const AnfNodePtr &node);
// Get the EventNodeType from a node.
COMMON_EXPORT std::optional<EventNodeType> GetEventNodeType(const AnfNodePtr &node);
// Get prefetch_distance from node attribute, return nullopt if not found.
COMMON_EXPORT std::optional<size_t> GetNodePrefetchDistance(const AnfNodePtr &node);
// Get release_distance from node attribute, return nullopt if not found.
COMMON_EXPORT std::optional<size_t> GetNodeReleaseDistance(const AnfNodePtr &node);
// Get enable_offload from node attribute, return nullopt if not found.
COMMON_EXPORT std::optional<bool> GetNodeEnableOffload(const AnfNodePtr &node);
COMMON_EXPORT size_t FindLastUserIndex(const CNodePtrList &exec_order, const AnfNodePtr &data_node,
                                       size_t current_node_index);
COMMON_EXPORT size_t CalculateAdaptiveReleaseDistance(const AnfNodePtr &data_node, const CNodePtrList &exec_order,
                                                      size_t data_node_index, size_t last_user_index);
COMMON_EXPORT size_t CalculateAdaptivePrefetchDistance(const AnfNodePtr &data_node, const CNodePtrList &exec_order,
                                                       size_t first_use_index);

}  // namespace hyper_offload
}  // namespace utils
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_UTILS_HYPER_OFFLOAD_HYPER_OFFLOAD_UTILS_H_
