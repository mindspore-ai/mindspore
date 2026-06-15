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

#include "utils/hyper_offload/hyper_offload_utils.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "abstract/abstract_value.h"
#include "include/utils/tensor_py.h"
#include "include/utils/utils.h"
#include "primitive/auto_generate/gen_ops_primitive_c.h"
#include "primitive/framework_ops.h"
#include "primitive/other_ops.h"
#include "utils/anf_utils.h"
#include "utils/compile_config.h"

namespace mindspore {
namespace utils {
namespace hyper_offload {
namespace {
constexpr std::string_view kMsDevHyperOffloadEnv = "MS_DEV_HYPER_OFFLOAD";
constexpr std::string_view kMsDevHyperOffloadPeakStrategyEnv = "MS_DEV_HYPER_OFFLOAD_PEAK_STRATEGY";
constexpr std::string_view kSelectDistanceConfig = "HYPER_OFFLOAD_SELECT_DISTANCE";
constexpr std::string_view kSelectNumConfig = "HYPER_OFFLOAD_SELECT_NUM";
constexpr std::string_view kPrefetchDistanceConfig = "HYPER_OFFLOAD_PREFETCH_DISTANCE";
constexpr std::string_view kReleaseDistanceConfig = "HYPER_OFFLOAD_RELEASE_DISTANCE";
constexpr std::string_view kMaxDeviceMemoryConfig = "HYPER_OFFLOAD_MAX_DEVICE_MEMORY";
constexpr std::string_view kScoreWeightSizeConfig = "HYPER_OFFLOAD_SCORE_WEIGHT_SIZE";
constexpr std::string_view kScoreWeightCoverStartConfig = "HYPER_OFFLOAD_SCORE_WEIGHT_COVER_START";
constexpr std::string_view kScoreWeightCoverEndConfig = "HYPER_OFFLOAD_SCORE_WEIGHT_COVER_END";
constexpr std::string_view kScoreWeightNeighborImpactConfig = "HYPER_OFFLOAD_SCORE_WEIGHT_NEIGHBOR_IMPACT";
constexpr std::string_view kEnabledStagesConfig = "MS_DEV_HYPER_OFFLOAD_FOR_STAGE";
constexpr std::string_view kOffloadPercentageConfig = "MS_DEV_HYPER_OFFLOAD_FILTER_PERCENTAGE";
constexpr std::string_view kD2HBandwidthConfig = "HYPER_OFFLOAD_D2H_BANDWIDTH_GBPS";
constexpr std::string_view kH2DBandwidthConfig = "HYPER_OFFLOAD_H2D_BANDWIDTH_GBPS";
constexpr std::string_view kDefaultKernelTimeConfig = "HYPER_OFFLOAD_DEFAULT_KERNEL_TIME_US";
constexpr std::string_view kEnableAdaptiveReleaseDistanceEnv = "MS_DEV_HYPER_OFFLOAD_ENABLE_ADAPTIVE_RELEASE_DISTANCE";
constexpr std::string_view kEnableAdaptivePrefetchDistanceEnv =
  "MS_DEV_HYPER_OFFLOAD_ENABLE_ADAPTIVE_PREFETCH_DISTANCE";

constexpr std::string_view kPrefetchDistanceAttr = "prefetch_distance";
constexpr std::string_view kReleaseDistanceAttr = "release_distance";
constexpr std::string_view kEnableOffloadAttr = "enable_offload";
constexpr auto kSwitchTrueBranchNumAttr = "TrueBranchNodeNum";
constexpr auto kSwitchFalseBranchNumAttr = "FalseBranchNodeNum";

constexpr size_t kViewDataInputIndex = 1;
constexpr size_t kPrimaryDataInputIndex = 1;
constexpr size_t kConditionSwitchFirstDataInputIndex = 2;
constexpr size_t kCopyToHostInputSize = 3;
constexpr size_t kCopyToHostExtInputSize = 5;
constexpr size_t kCopyToDeviceInputSize = 4;
constexpr size_t kSwitchBranchNameTupleSize = 2;
constexpr size_t kSwitchFalseBranchNameIndex = 0;
constexpr size_t kSwitchTrueBranchNameIndex = 1;

bool IsConditionSwitchDataInput(const AnfNodePtr &node, size_t input_index) {
  return IsPrimitiveCNode(node, prim::kPrimConditionSwitch) && input_index >= kConditionSwitchFirstDataInputIndex;
}

struct ConditionSwitchBranchNames {
  std::string false_branch_name;
  std::string true_branch_name;
};

std::optional<std::string> GetInlineBranchName(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  const auto &branch_name = node->user_data<std::string>(kHyperOffloadInlineBranchName);
  if (branch_name == nullptr || branch_name->empty()) {
    return std::nullopt;
  }
  return *branch_name;
}

std::optional<ConditionSwitchBranchNames> GetConditionSwitchBranchNames(const CNodePtr &switch_node) {
  MS_EXCEPTION_IF_NULL(switch_node);
  const auto &inline_sub_graph_names = switch_node->GetAttr(kInlineSubGraphName);
  if (inline_sub_graph_names == nullptr || !inline_sub_graph_names->isa<ValueTuple>()) {
    return std::nullopt;
  }
  const auto tuple_value = inline_sub_graph_names->cast<ValueTuplePtr>();
  if (tuple_value == nullptr) {
    return std::nullopt;
  }
  const auto &tuple_name = tuple_value->value();
  if (tuple_name.size() != kSwitchBranchNameTupleSize) {
    return std::nullopt;
  }
  ConditionSwitchBranchNames branch_names;
  branch_names.false_branch_name = GetValue<std::string>(tuple_name[kSwitchFalseBranchNameIndex]);
  branch_names.true_branch_name = GetValue<std::string>(tuple_name[kSwitchTrueBranchNameIndex]);
  return branch_names;
}

std::optional<bool> MatchConditionSwitchBranch(const ConditionSwitchBranchNames &branch_names,
                                               const std::string &branch_name) {
  if (branch_name == branch_names.true_branch_name) {
    return true;
  }
  if (branch_name == branch_names.false_branch_name) {
    return false;
  }
  return std::nullopt;
}

std::optional<std::pair<size_t, size_t>> BuildConditionSwitchBranchRange(size_t switch_index, size_t exec_order_size,
                                                                         size_t true_branch_node_num,
                                                                         size_t false_branch_node_num,
                                                                         bool is_true_branch) {
  if (true_branch_node_num == 0 || false_branch_node_num == 0 || switch_index + 1 >= exec_order_size) {
    return std::nullopt;
  }
  const size_t true_branch_start = switch_index + 1;
  if (true_branch_node_num > exec_order_size - true_branch_start) {
    return std::nullopt;
  }
  const size_t true_branch_end = true_branch_start + true_branch_node_num - 1;
  const size_t false_branch_start = true_branch_end + 1;
  if (false_branch_start >= exec_order_size || false_branch_node_num > exec_order_size - false_branch_start) {
    return std::nullopt;
  }
  const size_t false_branch_end = false_branch_start + false_branch_node_num - 1;
  return is_true_branch ? std::make_pair(true_branch_start, true_branch_end)
                        : std::make_pair(false_branch_start, false_branch_end);
}

std::optional<std::pair<size_t, size_t>> GetConditionSwitchBranchRange(const CNodePtr &switch_node, size_t switch_index,
                                                                       size_t exec_order_size, bool is_true_branch) {
  MS_EXCEPTION_IF_NULL(switch_node);
  const auto &true_branch_node_num_value = switch_node->GetAttr(kSwitchTrueBranchNumAttr);
  const auto &false_branch_node_num_value = switch_node->GetAttr(kSwitchFalseBranchNumAttr);
  if (true_branch_node_num_value == nullptr || false_branch_node_num_value == nullptr) {
    return std::nullopt;
  }
  const size_t true_branch_node_num = GetValue<size_t>(true_branch_node_num_value);
  const size_t false_branch_node_num = GetValue<size_t>(false_branch_node_num_value);
  return BuildConditionSwitchBranchRange(switch_index, exec_order_size, true_branch_node_num, false_branch_node_num,
                                         is_true_branch);
}

std::optional<std::pair<size_t, size_t>> GetInlineBranchIndexRangeBySwitch(const CNodePtrList &exec_order,
                                                                           const std::string &branch_name) {
  for (size_t i = 0; i < exec_order.size(); ++i) {
    const auto &node = exec_order[i];
    if (!IsPrimitiveCNode(node, prim::kPrimConditionSwitch)) {
      continue;
    }
    const auto branch_names = GetConditionSwitchBranchNames(node);
    if (!branch_names.has_value()) {
      continue;
    }
    const auto is_true_branch = MatchConditionSwitchBranch(*branch_names, branch_name);
    if (!is_true_branch.has_value()) {
      continue;
    }
    const auto branch_range = GetConditionSwitchBranchRange(node, i, exec_order.size(), *is_true_branch);
    if (branch_range.has_value()) {
      return branch_range;
    }
  }
  return std::nullopt;
}

std::string GetParameterDeviceStr(const AnfNodePtr &node) {
  constexpr auto kParameterDeviceUserDataName = "parameter_device";
  if (!node->isa<Parameter>()) {
    return "";
  }
  const auto &parameter = node->cast<ParameterPtr>();
  MS_EXCEPTION_IF_NULL(parameter);
  const auto value = parameter->default_param();
  if (value == nullptr) {
    return "";
  }
  const auto meta_tensor = value->cast_ptr<tensor::MetaTensor>();
  if (meta_tensor == nullptr) {
    return "";
  }
  const auto &user_data = meta_tensor->user_data<tensor::TensorPyUserData>(kParameterDeviceUserDataName);
  if (user_data == nullptr || !py::isinstance<py::str>(user_data->obj)) {
    return "";
  }
  return py::cast<std::string>(user_data->obj);
}

size_t ParseCompileConfigAsSize(std::string_view key) {
  std::string value = common::GetCompileConfig(std::string(key));
  try {
    return std::stoul(value);
  } catch (const std::invalid_argument &e) {
    MS_LOG(EXCEPTION) << "Invalid numeric value for compile config '" << key << "': " << value
                      << ". Error: " << e.what();
  } catch (const std::out_of_range &e) {
    MS_LOG(EXCEPTION) << "Out-of-range numeric value for compile config '" << key << "': " << value
                      << ". Error: " << e.what();
  }
  return 0;
}

double ParseCompileConfigAsDouble(std::string_view key, double default_val) {
  std::string value = common::GetCompileConfig(std::string(key));
  if (value.empty()) {
    return default_val;
  }
  try {
    return std::stod(value);
  } catch (...) {
    return default_val;
  }
}

std::set<int64_t> ParseEnabledStages(const std::string &env_value) {
  std::set<int64_t> stages;
  if (env_value.empty()) {
    return stages;
  }
  std::stringstream ss(env_value);
  std::string token;
  while (std::getline(ss, token, ',')) {
    if (token.empty()) {
      continue;
    }
    try {
      size_t pos = 0;
      auto stage = std::stoll(token, &pos);
      if (pos != token.size()) {
        MS_LOG(EXCEPTION) << "Invalid stage value '" << token << "' in MS_DEV_HYPER_OFFLOAD_FOR_STAGE, "
                          << "expected integer but got extra characters.";
      }
      (void)stages.insert(stage);
    } catch (const std::exception &e) {
      MS_LOG(EXCEPTION) << "Invalid stage value '" << token << "' in MS_DEV_HYPER_OFFLOAD_FOR_STAGE, "
                        << "expected integer but got: " << e.what();
    }
  }
  return stages;
}

bool IsSupportedActivationSource(const AnfNodePtr &node) {
  if (node == nullptr || !node->isa<CNode>()) {
    return false;
  }
  auto abs = node->abstract();
  if (abs == nullptr) {
    return false;
  }
  return abs->isa<abstract::AbstractTensor>() || abs->isa<abstract::AbstractTuple>();
}

void CollectSourcesForInput(const AnfNodePtr &input_node, std::vector<KernelWithIndex> *sources) {
  MS_EXCEPTION_IF_NULL(input_node);
  MS_EXCEPTION_IF_NULL(sources);
  if (IsPrimitiveCNode(input_node, prim::kPrimMakeTuple)) {
    auto make_tuple = input_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(make_tuple);
    for (size_t i = 1; i < make_tuple->inputs().size(); ++i) {
      CollectSourcesForInput(make_tuple->input(i), sources);
    }
    return;
  }
  if (IsViewNode(input_node)) {
    auto view_node = input_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_CHECK_FAIL(view_node->inputs().size() > kViewDataInputIndex, "View node input size is not correct");
    CollectSourcesForInput(view_node->input(kViewDataInputIndex), sources);
    return;
  }
  const auto &source = GetRealDataNodeWithIndex(input_node);
  if (IsSupportedActivationSource(source.first)) {
    sources->emplace_back(source);
  }
}

}  // namespace

HyperOffloadConfig &HyperOffloadConfig::GetInstance() {
  static HyperOffloadConfig instance;
  return instance;
}

uintptr_t GetNodeAddress(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  return reinterpret_cast<uintptr_t>(node.get());
}

HyperOffloadConfig::HyperOffloadConfig() {
  enable_hyper_offload_ = (common::GetEnv(std::string(kMsDevHyperOffloadEnv)) == "1");
  enable_hyper_offload_peak_strategy_ = (common::GetEnv(std::string(kMsDevHyperOffloadPeakStrategyEnv)) == "1");
  if (enable_hyper_offload_) {
    select_distance_ = ParseCompileConfigAsSize(kSelectDistanceConfig);
    select_num_ = ParseCompileConfigAsSize(kSelectNumConfig);
    prefetch_distance_ = ParseCompileConfigAsSize(kPrefetchDistanceConfig);
    release_distance_ = ParseCompileConfigAsSize(kReleaseDistanceConfig);
    enabled_stages_ = ParseEnabledStages(common::GetEnv(std::string(kEnabledStagesConfig)));
    offload_percentage_ = common::GetEnv(std::string(kOffloadPercentageConfig));
    std::string max_mem_val = common::GetCompileConfig(std::string(kMaxDeviceMemoryConfig));
    max_device_memory_ = max_mem_val.empty() ? 0 : ParseCompileConfigAsSize(kMaxDeviceMemoryConfig);
    score_weight_size_ = ParseCompileConfigAsDouble(kScoreWeightSizeConfig, 1.0);
    score_weight_cover_start_ = ParseCompileConfigAsDouble(kScoreWeightCoverStartConfig, 1.0);
    score_weight_cover_end_ = ParseCompileConfigAsDouble(kScoreWeightCoverEndConfig, 1.0);
    score_weight_neighbor_impact_ = ParseCompileConfigAsDouble(kScoreWeightNeighborImpactConfig, 1.0);

    enable_adaptive_release_distance_ = (common::GetEnv(std::string(kEnableAdaptiveReleaseDistanceEnv)) == "1");
    d2h_bandwidth_gbps_ = ParseCompileConfigAsDouble(kD2HBandwidthConfig, kDefaultD2HBandwidthGbps);
    default_kernel_time_us_ = ParseCompileConfigAsDouble(kDefaultKernelTimeConfig, kDefaultKernelTimeUs);

    enable_adaptive_prefetch_distance_ = (common::GetEnv(std::string(kEnableAdaptivePrefetchDistanceEnv)) == "1");
    h2d_bandwidth_gbps_ = ParseCompileConfigAsDouble(kH2DBandwidthConfig, kDefaultH2DBandwidthGbps);
  }
}

bool IsStageEnabled(int64_t stage_id, const std::set<int64_t> &enabled_stages) {
  if (enabled_stages.empty()) {
    return true;
  }
  return enabled_stages.find(stage_id) != enabled_stages.end();
}

std::vector<size_t> GetRefOutputIndexesForInput(const KernelWithIndex &user_node) {
  std::vector<size_t> ref_output_indexes;
  const auto &node = user_node.first;
  if (node == nullptr || user_node.second == 0 || !node->isa<CNode>()) {
    return ref_output_indexes;
  }
  const auto &cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (user_node.second >= cnode->inputs().size()) {
    return ref_output_indexes;
  }
  const size_t tensor_input_index = user_node.second - 1;
  if (tensor_input_index >= common::AnfAlgo::GetInputTensorNum(cnode)) {
    return ref_output_indexes;
  }
  const auto &ref_map = cnode->user_data<OutputInputRefMap>(kHyperOffloadRefOutputInputMap);
  if (ref_map == nullptr) {
    return ref_output_indexes;
  }
  for (const auto &ref : *ref_map) {
    if (ref.second == tensor_input_index) {
      ref_output_indexes.emplace_back(ref.first);
    }
  }
  return ref_output_indexes;
}

bool IsRefInputUser(const KernelWithIndex &user_node) { return !GetRefOutputIndexesForInput(user_node).empty(); }

bool IsRefOutput(const KernelWithIndex &node_output) {
  const auto &node = node_output.first;
  if (node == nullptr || !node->isa<CNode>()) {
    return false;
  }
  const auto &cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  const auto &ref_map = cnode->user_data<OutputInputRefMap>(kHyperOffloadRefOutputInputMap);
  if (ref_map == nullptr) {
    return false;
  }
  return ref_map->count(node_output.second) != 0;
}

std::optional<std::pair<size_t, size_t>> GetInlineBranchIndexRange(const FuncGraphPtr &graph,
                                                                   const CNodePtrList &exec_order,
                                                                   const AnfNodePtr &node) {
  if (graph == nullptr || node == nullptr || exec_order.empty()) {
    return std::nullopt;
  }
  const auto branch_name = GetInlineBranchName(node);
  if (!branch_name.has_value()) {
    return std::nullopt;
  }
  const auto switch_branch_range = GetInlineBranchIndexRangeBySwitch(exec_order, *branch_name);
  if (switch_branch_range.has_value()) {
    return switch_branch_range;
  }

  bool found = false;
  size_t branch_start = 0;
  size_t branch_end = 0;
  for (size_t i = 0; i < exec_order.size(); ++i) {
    const auto current_branch_name = GetInlineBranchName(exec_order[i]);
    if (!current_branch_name.has_value() || *current_branch_name != *branch_name) {
      continue;
    }
    if (!found) {
      branch_start = i;
      found = true;
    }
    branch_end = i;
  }
  if (!found) {
    return std::nullopt;
  }
  return std::make_pair(branch_start, branch_end);
}

std::optional<size_t> GetInlineBranchStartAttachIndex(const FuncGraphPtr &graph, const CNodePtrList &exec_order,
                                                      const AnfNodePtr &node) {
  const auto branch_range = GetInlineBranchIndexRange(graph, exec_order, node);
  if (!branch_range.has_value()) {
    return std::nullopt;
  }
  return branch_range->first == 0 ? 0 : branch_range->first - 1;
}

bool IsD2HNode(const AnfNodePtr &node) {
  return IsOneOfPrimitiveCNode(node, {prim::kPrimCopyToHost, prim::kPrimCopyToHostExt});
}

bool IsH2DNode(const AnfNodePtr &node) { return IsOneOfPrimitiveCNode(node, {prim::kPrimCopyToDevice}); }

bool IsViewOrInplaceNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (IsViewNode(node) || IsInplaceNode(node)) {
    return true;
  }
  if (!node->isa<CNode>()) {
    return false;
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (cnode->inputs().empty()) {
    return false;
  }
  const auto &prim = GetValueNode<PrimitivePtr>(cnode->input(0));
  return prim != nullptr && (prim->graph_view_prim() || prim->inplace_prim());
}

bool IsViewNode(const AnfNodePtr &node) {
  if (node == nullptr) {
    return false;
  }
  if (IsPrimitiveCNode(node, prim::kPrimTupleGetItem)) {
    return true;
  }
  return node->has_user_data("is_view");
}

bool IsInplaceNode(const AnfNodePtr &node) {
  if (node == nullptr) {
    return false;
  }
  return node->has_user_data("is_inplace");
}

KernelWithIndex GetRealDataNodeWithIndex(const AnfNodePtr &node) {
  auto res = common::AnfAlgo::VisitKernelWithReturnType(node, 0, false);
  while (true) {
    if (IsViewNode(res.first)) {
      auto cnode = res.first->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      MS_EXCEPTION_IF_CHECK_FAIL(cnode->inputs().size() > kPrimaryDataInputIndex,
                                 "View node input size is not correct");
      res = common::AnfAlgo::VisitKernelWithReturnType(cnode->input(kPrimaryDataInputIndex), 0, false);
      continue;
    }
    if (IsPrimitiveCNode(res.first, prim::kPrimConditionSwitch)) {
      auto switch_node = res.first->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(switch_node);
      const size_t data_input_index = res.second + kConditionSwitchFirstDataInputIndex;
      if (data_input_index >= switch_node->inputs().size()) {
        break;
      }
      // ConditionSwitch forwards data inputs to branch consumers without materializing them on device.
      res = common::AnfAlgo::VisitKernelWithReturnType(switch_node->input(data_input_index), 0, false);
      continue;
    }
    break;
  }
  return res;
}

bool IsValidOffloadNode(const UserInfo &user_info) {
  auto node = user_info.node.first;
  MS_EXCEPTION_IF_NULL(node);
  if (!AnfUtils::IsRealKernel(node)) {
    return false;
  }
  if (IsRefOutput(user_info.node)) {
    MS_LOG(INFO) << "Skip HyperOffload ref output node: " << node->fullname_with_scope()
                 << ", output index: " << user_info.node.second;
    return false;
  }
  if (IsOneOfPrimitiveCNode(node, {prim::kPrimMoveTo, prim::kPrimTensorMove, prim::kPrimConditionSwitch})) {
    return false;
  }
  const bool has_d2h_user =
    std::any_of(user_info.users.begin(), user_info.users.end(), [](const auto &e) { return IsD2HNode(e.first); });
  if (has_d2h_user) {
    return false;
  }
  bool has_view_user =
    std::any_of(user_info.users.begin(), user_info.users.end(), [](const auto &e) { return IsViewNode(e.first); });
  bool has_inplace_user =
    std::any_of(user_info.users.begin(), user_info.users.end(), [](const auto &e) { return IsInplaceNode(e.first); });
  // Currently skip view-inplace scenario.
  if (has_view_user && has_inplace_user) {
    MS_LOG(ERROR) << "View and inplace user found for node: " << node->DebugString();
    return false;
  }
  auto node_enable_offload = GetNodeEnableOffload(node);
  if (node_enable_offload.has_value() && !node_enable_offload.value()) {
    return false;
  }
  return true;
}

namespace {
bool ProducerAbstractOkForCollectUsers(const KernelWithIndex &producer) {
  const auto &node = producer.first;
  MS_EXCEPTION_IF_NULL(node);
  const auto &abs = node->abstract();
  if (abs == nullptr) {
    return false;
  }
  if (abs->isa<abstract::AbstractTensor>()) {
    return true;
  }
  if (!HyperOffloadConfig::GetInstance().enable_hyper_offload_peak_strategy()) {
    return false;
  }
  auto tuple_abs = abs->cast_ptr<abstract::AbstractTuple>();
  if (tuple_abs == nullptr || producer.second >= tuple_abs->elements().size()) {
    return false;
  }
  // Keep tensor tuple-output producers visible only to peak strategy for now, to avoid changing
  // distance-based candidate collection through this shared utility path.
  const auto &output_abs = tuple_abs->elements()[producer.second];
  return output_abs != nullptr && output_abs->isa<abstract::AbstractTensor>();
}
}  // namespace

std::map<KernelWithIndex, std::pair<size_t, size_t>> BuildUserIndexRangeMap(const CNodePtrList &exec_order) {
  std::map<KernelWithIndex, std::pair<size_t, size_t>> user_index_ranges;
  for (size_t i = 0; i < exec_order.size(); ++i) {
    for (const auto &input : exec_order[i]->inputs()) {
      std::vector<KernelWithIndex> sources;
      CollectSourcesForInput(input, &sources);
      for (const auto &source : sources) {
        if (auto [it, inserted] = user_index_ranges.try_emplace(source, i, i); !inserted) {
          it->second.second = i;
        }
      }
    }
  }
  return user_index_ranges;
}

UserInfoList CollectAllNodeUsers(const CNodePtrList &nodes) {
  UserInfoList user_info_list;
  std::map<KernelWithIndex, size_t> source_to_index;
  MS_LOG(INFO) << "Collecting node users from " << nodes.size() << " nodes";
  for (const CNodePtr &node : nodes) {
    for (size_t i = 1; i < node->inputs().size(); ++i) {
      if (IsConditionSwitchDataInput(node, i)) {
        continue;
      }
      std::vector<KernelWithIndex> sources;
      CollectSourcesForInput(node->input(i), &sources);
      for (const auto &source : sources) {
        if (!ProducerAbstractOkForCollectUsers(source)) {
          continue;
        }
        auto iter = source_to_index.find(source);
        if (iter == source_to_index.end()) {
          source_to_index[source] = user_info_list.size();
          user_info_list.emplace_back(UserInfo{source, {KernelWithIndex{node, i}}});
        } else {
          user_info_list[iter->second].users.emplace_back(KernelWithIndex{node, i});
        }
      }
    }
  }
  MS_LOG(INFO) << "Collected " << user_info_list.size() << " node user entries";
  return user_info_list;
}

void CopyHyperOffloadFrontendAttrs(const AnfNodePtr &from, const AnfNodePtr &to) {
  MS_EXCEPTION_IF_NULL(from);
  MS_EXCEPTION_IF_NULL(to);
  if (!from->isa<CNode>() || !to->isa<CNode>()) {
    return;
  }
  auto from_cnode = from->cast<CNodePtr>();
  auto to_cnode = to->cast<CNodePtr>();
  const std::array kHyperOffloadFrontendAttrNames{kPrefetchDistanceAttr, kReleaseDistanceAttr, kEnableOffloadAttr};
  for (const auto attr_name : kHyperOffloadFrontendAttrNames) {
    const std::string name(attr_name);
    if (from_cnode->HasAttr(name)) {
      to_cnode->AddAttr(name, from_cnode->GetAttr(name));
    }
  }
}

CNodePtr BuildToHostNode(const FuncGraphPtr &graph, const AnfNodePtr &data_node,
                         const HandleNewNodeCallback &callback) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(data_node);
  const std::string &prim_name = prim::kPrimCopyToHost->name();
  auto prim = std::make_shared<Primitive>(prim_name);
  AnfNodePtrList inputs{NewValueNode(prim), data_node};
  auto sync_node = NewValueNode(MakeValue(false));
  sync_node->set_abstract(std::make_shared<abstract::AbstractScalar>(false));
  MS_EXCEPTION_IF_CHECK_FAIL(kCopyToHostInputSize == inputs.size() + 1, "CopyToHost input size check failed.");
  (void)inputs.emplace_back(sync_node);
  auto ret_node = graph->NewCNode(inputs);
  ret_node->set_abstract(data_node->abstract());
  if (data_node->isa<CNode>()) {
    CopyHyperOffloadFrontendAttrs(data_node, ret_node);
  }
  if (callback != nullptr) {
    callback(ret_node, data_node);
  }
  return ret_node;
}

CNodePtr BuildInplaceToHostNode(const FuncGraphPtr &graph, const AnfNodePtr &data_node, const AnfNodePtr &update_node,
                                const AnfNodePtr &depend_node, const HandleNewNodeCallback &callback) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(data_node);
  MS_EXCEPTION_IF_NULL(update_node);
  MS_EXCEPTION_IF_NULL(depend_node);
  const std::string &prim_name = prim::kPrimCopyToHostExt->name();
  auto prim = std::make_shared<Primitive>(prim_name);
  AnfNodePtrList inputs{NewValueNode(prim), data_node, update_node, depend_node};
  auto sync_node = NewValueNode(MakeValue(false));
  sync_node->set_abstract(std::make_shared<abstract::AbstractScalar>(false));
  MS_EXCEPTION_IF_CHECK_FAIL(kCopyToHostExtInputSize == inputs.size() + 1, "CopyToHostExt input size check failed.");
  (void)inputs.emplace_back(sync_node);
  auto ret_node = graph->NewCNode(inputs);
  ret_node->set_abstract(data_node->abstract());
  if (depend_node->isa<CNode>()) {
    CopyHyperOffloadFrontendAttrs(depend_node, ret_node);
  }
  if (callback != nullptr) {
    callback(ret_node, depend_node);
  }
  return ret_node;
}

CNodePtr BuildToDeviceNode(const FuncGraphPtr &graph, const AnfNodePtr &data_node,
                           const HandleNewNodeCallback &callback) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(data_node);
  const std::string &prim_name = prim::kPrimCopyToDevice->name();
  auto prim = std::make_shared<Primitive>(prim_name);
  auto depend_node = NewValueNode(MakeValue(false));
  depend_node->set_abstract(std::make_shared<abstract::AbstractScalar>(false));
  AnfNodePtrList inputs{NewValueNode(prim), data_node, depend_node, depend_node};
  MS_EXCEPTION_IF_CHECK_FAIL(inputs.size() == kCopyToDeviceInputSize, "CopyToDevice input size check failed.");
  auto ret_node = graph->NewCNode(inputs);
  ret_node->set_abstract(data_node->abstract());
  if (data_node->isa<CNode>()) {
    CopyHyperOffloadFrontendAttrs(data_node, ret_node);
  }
  if (callback != nullptr) {
    callback(ret_node, data_node);
  }
  return ret_node;
}

HandleNewNodeCallback CreateContextCallback(HandleNewNodeCallback original_callback, const AnfNodePtr &context_node) {
  if (original_callback == nullptr) {
    return nullptr;
  }
  return [original_callback, context_node](const AnfNodePtr &new_node, const AnfNodePtr &) {
    original_callback(new_node, context_node);
  };
}

std::optional<ParameterPtr> FindParameterByRefName(const FuncGraphPtr &graph, const std::string &name) {
  MS_EXCEPTION_IF_NULL(graph);
  for (const auto &input : graph->parameters()) {
    if (!input->isa<Parameter>()) {
      continue;
    }
    auto param = input->cast<ParameterPtr>();
    if (param->name() == name) {
      return param;
    }
  }
  return std::nullopt;
}

std::optional<ParameterPtr> GetRemoteParameter(const FuncGraphPtr &graph, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto abstract = node->abstract();
  if (abstract == nullptr) {
    return std::nullopt;
  }
  if (!abstract->isa<abstract::AbstractRefTensor>()) {
    return std::nullopt;
  }
  auto abstract_ref = abstract->cast<abstract::AbstractRefPtr>();
  const auto &ref_value = abstract_ref->ref_key_value()->cast<StringImmPtr>();
  MS_EXCEPTION_IF_NULL(ref_value);
  const auto &ref_name = ref_value->value();
  const auto &ref_node_res = FindParameterByRefName(graph, ref_name);
  if (!ref_node_res.has_value()) {
    return std::nullopt;
  }
  const auto &parameter = ref_node_res.value();
  const auto &device_str = GetParameterDeviceStr(parameter);
  if (device_str != kToRemote) {
    return std::nullopt;
  }
  return parameter;
}

void UpdateNodeInput(const CNodePtr &node, const size_t input_index, const AnfNodePtr &new_input_node) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(new_input_node);
  MS_EXCEPTION_IF_CHECK_FAIL(input_index < node->inputs().size(), "Input index out of range when updating node input.");
  node->set_input(input_index, new_input_node);
}

std::pair<CNodePtr, CNodePtr> EventNodeFactory::CreateSendRecvPair(size_t send_stream_id, size_t recv_stream_id,
                                                                   EventNodeType type, const AnfNodePtr &data_node) {
  MS_EXCEPTION_IF_NULL(fg_);
  MS_EXCEPTION_IF_NULL(data_node);
  MS_EXCEPTION_IF_CHECK_FAIL(send_stream_id != recv_stream_id, "Send and recv stream id should not be the same");
  auto send_prim = std::make_shared<Primitive>(prim::kPrimSend->name());
  auto recv_prim = std::make_shared<Primitive>(prim::kPrimReceive->name());
  auto send_node = fg_->NewCNode({NewValueNode(send_prim), data_node});
  auto recv_node = fg_->NewCNode({NewValueNode(recv_prim), data_node});

  MS_LOG(INFO) << "CreateSendRecvPair event_id_: " << event_id_;

  // Add pair send, recv node with the same event id.
  send_node->set_user_data(kHyperOffloadEventId, std::make_shared<uint32_t>(event_id_));
  recv_node->set_user_data(kHyperOffloadEventId, std::make_shared<uint32_t>(event_id_));
  ++event_id_;

  // Add corresponding stream_id to pair send and recv node.
  send_node->set_user_data(kHyperOffloadEventStreamId, std::make_shared<size_t>(send_stream_id));
  recv_node->set_user_data(kHyperOffloadEventStreamId, std::make_shared<size_t>(recv_stream_id));
  // Add event type to pair send and recv node.
  send_node->set_user_data(kHyperOffloadEventNodeType, std::make_shared<EventNodeType>(type));
  recv_node->set_user_data(kHyperOffloadEventNodeType, std::make_shared<EventNodeType>(type));
  return std::pair<CNodePtr, CNodePtr>(send_node, recv_node);
}

AnfNodePtr BuildFreeDataNode(const FuncGraphPtr &fg, const AnfNodePtr &node, size_t type) {
  MS_EXCEPTION_IF_NULL(fg);
  MS_EXCEPTION_IF_NULL(node);
  auto free_node = fg->NewCNode({NewValueNode(kFreeDataPrimitive), node, NewValueNode(MakeValue(type))});
  free_node->set_user_data(kHyperOffloadEventNodeType, std::make_shared<EventNodeType>(kFreeAndSwitchData));
  return free_node;
}

AnfNodePtr BuildAssignDataNode(const FuncGraphPtr &fg, const AnfNodePtr &node, const AnfNodePtr &data_node,
                               size_t type) {
  MS_EXCEPTION_IF_NULL(fg);
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(data_node);
  auto assign_node = fg->NewCNode({NewValueNode(kAssignDataPrimitive), node, data_node, NewValueNode(MakeValue(type))});
  assign_node->set_user_data(kHyperOffloadEventNodeType, std::make_shared<EventNodeType>(kFreeAndSwitchData));
  return assign_node;
}

std::optional<size_t> GetHyperOffloadId(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto user_data = node->user_data<size_t>(kHyperOffloadId);
  if (user_data != nullptr) {
    return *user_data;
  }
  return std::nullopt;
}

std::optional<size_t> GetFreeOrAssignNodeType(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (IsPrimitiveCNode(node, kFreeDataPrimitive)) {
    auto cnode = node->cast<CNodePtr>();
    constexpr size_t free_data_input_size = 3;
    MS_EXCEPTION_IF_CHECK_FAIL(cnode->inputs().size() == free_data_input_size,
                               "FreeData node input size is not correct");
    constexpr size_t type_input_index = 2;
    auto value_node = cnode->input(type_input_index)->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(value_node);
    return GetValue<size_t>(value_node->value());
  }
  if (IsPrimitiveCNode(node, kAssignDataPrimitive)) {
    auto cnode = node->cast<CNodePtr>();
    constexpr size_t assign_data_input_size = 4;
    MS_EXCEPTION_IF_CHECK_FAIL(cnode->inputs().size() == assign_data_input_size,
                               "AssignData node input size is not correct");
    constexpr size_t type_input_index = 3;
    auto value_node = cnode->input(type_input_index)->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(value_node);
    return GetValue<size_t>(value_node->value());
  }
  return std::nullopt;
}

std::optional<size_t> GetSendOrRecvStreamId(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return std::nullopt;
  }
  auto user_data = node->user_data<size_t>(kHyperOffloadEventStreamId);
  if (user_data != nullptr) {
    return *user_data;
  }
  return std::nullopt;
}

std::optional<EventNodeType> GetEventNodeType(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto user_data = node->user_data<EventNodeType>(kHyperOffloadEventNodeType);
  if (user_data != nullptr) {
    return *user_data;
  }
  return std::nullopt;
}

namespace {
template <typename T>
std::optional<T> GetNodeAttrValue(const AnfNodePtr &node, std::string_view attr_name) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return std::nullopt;
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto attr = cnode->GetAttr(std::string(attr_name));
  if (attr == nullptr) {
    return std::nullopt;
  }
  return GetValue<T>(attr);
}
}  // namespace

std::optional<size_t> GetNodePrefetchDistance(const AnfNodePtr &node) {
  auto value = GetNodeAttrValue<size_t>(node, kPrefetchDistanceAttr);
  return value;
}

std::optional<size_t> GetNodeReleaseDistance(const AnfNodePtr &node) {
  auto value = GetNodeAttrValue<size_t>(node, kReleaseDistanceAttr);
  return value;
}

std::optional<bool> GetNodeEnableOffload(const AnfNodePtr &node) {
  return GetNodeAttrValue<bool>(node, kEnableOffloadAttr);
}

size_t FindLastUserIndex(const CNodePtrList &exec_order, const AnfNodePtr &data_node, size_t current_node_index) {
  if (exec_order.empty()) {
    return current_node_index;
  }
  if (current_node_index >= exec_order.size()) {
    return exec_order.size() - 1;
  }
  const auto &[target_node, target_index] = GetRealDataNodeWithIndex(data_node);
  for (size_t i = exec_order.size() - 1; i > current_node_index; --i) {
    const auto &node = exec_order[i];
    for (size_t input_index = 0; input_index < node->inputs().size(); ++input_index) {
      if (IsConditionSwitchDataInput(node, input_index)) {
        continue;
      }
      const auto &input = node->input(input_index);
      // Get original data node to handle view-inplace scenario
      const auto &[cur_node, cur_index] = GetRealDataNodeWithIndex(input);
      MS_EXCEPTION_IF_NULL(cur_node);
      if (!cur_node->isa<CNode>()) {
        continue;
      }
      if (cur_node == target_node && cur_index == target_index) {
        return i;
      }
    }
  }
  return current_node_index;
}

size_t GetOperatorSizeBytes(const AnfNodePtr &data_node) {
  const auto &output_mem_sizes = data_node->user_data<std::vector<size_t>>(kHyperOffloadOutputMemSizes);
  if (output_mem_sizes == nullptr || output_mem_sizes->empty()) {
    return 0;
  }
  return (*output_mem_sizes)[0];
}

size_t CalculateDistanceFromCopyTime(double copy_time_us, double kernel_time_us, size_t start_index,
                                     size_t exec_order_size) {
  double cumulative_time_us = 0.0;
  size_t distance = 0;
  for (size_t i = start_index; i < exec_order_size; ++i) {
    cumulative_time_us += kernel_time_us;
    ++distance;
    if (cumulative_time_us >= copy_time_us) {
      break;
    }
  }
  return distance;
}

size_t CalculateAdaptiveReleaseDistance(const AnfNodePtr &data_node, const CNodePtrList &exec_order,
                                        size_t data_node_index, size_t last_user_index) {
  MS_EXCEPTION_IF_NULL(data_node);
  const auto &config = HyperOffloadConfig::GetInstance();

  const size_t operator_size_bytes = GetOperatorSizeBytes(data_node);
  const double d2h_bandwidth_gbps = config.d2h_bandwidth_gbps();
  const double d2h_copy_time_us = static_cast<double>(operator_size_bytes) / (d2h_bandwidth_gbps * 1e3);
  const double kernel_time_us = config.default_kernel_time_us();

  return CalculateDistanceFromCopyTime(d2h_copy_time_us, kernel_time_us, last_user_index + 1, exec_order.size());
}

size_t CalculateBackwardDistanceFromCopyTime(double copy_time_us, double kernel_time_us, size_t first_use_index) {
  if (first_use_index == 0) {
    return 0;
  }
  double cumulative_time_us = 0.0;
  size_t distance = 0;
  for (size_t i = first_use_index - 1;; --i) {
    cumulative_time_us += kernel_time_us;
    ++distance;
    if (cumulative_time_us >= copy_time_us || i == 0) {
      break;
    }
  }
  return distance;
}

size_t CalculateAdaptivePrefetchDistance(const AnfNodePtr &data_node, const CNodePtrList &exec_order,
                                         size_t first_use_index) {
  MS_EXCEPTION_IF_NULL(data_node);
  const auto &config = HyperOffloadConfig::GetInstance();

  const size_t operator_size_bytes = GetOperatorSizeBytes(data_node);
  const double h2d_bandwidth_gbps = config.h2d_bandwidth_gbps();
  const double h2d_copy_time_us = static_cast<double>(operator_size_bytes) / (h2d_bandwidth_gbps * 1e3);
  const double kernel_time_us = config.default_kernel_time_us();

  return CalculateBackwardDistanceFromCopyTime(h2d_copy_time_us, kernel_time_us, first_use_index);
}
}  // namespace hyper_offload
}  // namespace utils
}  // namespace mindspore
