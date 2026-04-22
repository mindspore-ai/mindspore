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

#include "utils/hyper_offload/hyper_offload.h"

#include <algorithm>
#include <functional>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "include/utils/anfalgo.h"
#include "include/utils/comm_manager.h"
#include "include/utils/parallel_context.h"
#include "include/utils/utils.h"
#include "primitive/framework_ops.h"
#include "primitive/other_ops.h"
#include "utils/hyper_offload/distance_base_hyper_offload_strategy.h"
#include "utils/hyper_offload/hyper_offload_event.h"
#include "utils/hyper_offload/hyper_offload_mem_estimate.h"
#include "utils/hyper_offload/hyper_offload_operation.h"
#include "utils/hyper_offload/memory_peak_based_hyper_offload_strategy.h"
#include "utils/log_adapter.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace utils {
namespace hyper_offload {
namespace {
constexpr size_t kPrimaryDataInputIndex = 1;

std::optional<size_t> FindNodeIndex(const CNodePtrList &exec_order, const AnfNodePtr &target_node) {
  auto iter = std::find(exec_order.begin(), exec_order.end(), target_node);
  if (iter == exec_order.end()) {
    return std::nullopt;
  }
  return static_cast<size_t>(std::distance(exec_order.begin(), iter));
}

constexpr char HCCL_WORLD_GROUP[] = "hccl_world_group";
constexpr char NCCL_WORLD_GROUP[] = "nccl_world_group";
std::string GetWorldGroup() {
  const auto &context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  const auto &backend = context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  if (backend == kAscendDevice) {
    return HCCL_WORLD_GROUP;
  }
  if (backend == kGPUDevice) {
    return NCCL_WORLD_GROUP;
  }
  MS_LOG(EXCEPTION) << "Invalid backend: " << backend;
}

int64_t GetDeviceNum(const std::string &world_group) {
  const auto &parallel_context = parallel::ParallelContext::GetInstance();
  MS_EXCEPTION_IF_NULL(parallel_context);
  if (parallel_context->device_num_is_set()) {
    return parallel_context->device_num();
  }
  uint32_t world_rank_size = 0;
  if (!CommManager::GetInstance().GetRankSize(world_group, &world_rank_size)) {
    MS_LOG(EXCEPTION) << "Get rank size failed";
  }
  MS_LOG(INFO) << "Get device num from communication model, the device num is " << world_rank_size;
  return UintToInt(world_rank_size);
}

void ValidatePipelineParams(int64_t global_rank, int64_t device_num, int64_t stage_num) {
  if (device_num < 1) {
    MS_LOG(EXCEPTION) << "For 'PipelineSplit', device_num must be positive, but got: " << device_num;
  }
  if (global_rank < 0) {
    MS_LOG(EXCEPTION) << "For 'PipelineSplit', global_rank must be nonnegative, but got: " << global_rank;
  }
  if (stage_num == 0) {
    MS_LOG(EXCEPTION) << "Stage_num is zero";
  }
  if (device_num % stage_num != 0) {
    MS_LOG(EXCEPTION) << "Device_num must be divisible by stage_num, got device_num: " << device_num
                      << ", stage_num: " << stage_num;
  }
}

int64_t GetPipelineStageId() {
  auto global_rank = UintToInt(GetRank());
  const auto &parallel_context = parallel::ParallelContext::GetInstance();
  MS_EXCEPTION_IF_NULL(parallel_context);
  auto stage_num = parallel_context->pipeline_stage_split_num();
  auto device_num = GetDeviceNum(GetWorldGroup());
  ValidatePipelineParams(global_rank, device_num, stage_num);
  auto per_stage_rank_num = device_num / stage_num;
  return global_rank / per_stage_rank_num;
}
}  // namespace

HyperOffloadOptimizer::HyperOffloadOptimizer(const HyperOffloadInput &input)
    : graph_(input.graph),
      exec_order_(input.exec_order),
      callback_(input.callback),
      enable_offload_parameter_(input.enable_offload_parameter),
      enable_offload_activation_(input.enable_offload_activation) {
  MS_LOG(INFO) << "HyperOffloadOptimizer initialized, enable_parameter: " << enable_offload_parameter_
               << ", enable_activation: " << enable_offload_activation_;
}

void HyperOffloadOptimizer::AddSingleParameterOperations(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(graph_);
  MS_EXCEPTION_IF_NULL(node);
  auto node_enable_offload = GetNodeEnableOffload(node);
  if (node_enable_offload.has_value() && !node_enable_offload.value()) {
    return;
  }
  if (IsPrimitiveCNode(node, prim::kPrimConditionSwitch)) {
    return;
  }
  const auto node_index_opt = FindNodeIndex(exec_order_, node);
  if (!node_index_opt.has_value()) {
    return;
  }
  const size_t node_index = node_index_opt.value();
  CNodePtrList to_remote_nodes{};
  for (size_t i = 0; i < node->inputs().size(); ++i) {
    const auto &input = node->input(i);
    if (IsRefInputUser({node, i})) {
      MS_LOG(INFO) << "Skip HyperOffload parameter input replacement for ref input, node: "
                   << node->fullname_with_scope() << ", input index: " << i;
      continue;
    }
    auto real_input = common::AnfAlgo::VisitKernelWithReturnType(input, 0, false, {prim::kPrimLoad}).first;
    if (IsPrimitiveCNode(real_input, prim::kPrimLoad)) {
      auto load_node = real_input->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(load_node);
      MS_EXCEPTION_IF_CHECK_FAIL(load_node->inputs().size() > kPrimaryDataInputIndex, "Load node input size invalid.");
      auto data_node = load_node->input(kPrimaryDataInputIndex);
      if (!GetRemoteParameter(graph_, data_node).has_value()) {
        continue;
      }
      auto to_device_node = BuildToDeviceNode(graph_, data_node, CreateContextCallback(callback_, node));
      to_device_node->set_abstract(real_input->abstract());
      UpdateNodeInput(node, i, to_device_node);
      operations_.AddOperationBefore(node_index, to_device_node);
      continue;
    }
    const auto &param_res = GetRemoteParameter(graph_, input);
    if (!param_res.has_value()) {
      continue;
    }
    auto to_device_node = BuildToDeviceNode(graph_, input, CreateContextCallback(callback_, node));
    auto input_abstract = dyn_cast<abstract::AbstractRefTensor>(input->abstract());
    MS_EXCEPTION_IF_CHECK_FAIL(input_abstract != nullptr, "Remote parameter abstract should be AbstractRefTensor.");
    to_device_node->set_abstract(input_abstract->CloneAsTensor());
    UpdateNodeInput(node, i, to_device_node);
    operations_.AddOperationBefore(node_index, to_device_node);
    auto to_remote_node = BuildInplaceToHostNode(graph_, to_device_node, param_res.value(), node, callback_);
    to_remote_nodes.push_back(to_remote_node);
  }

  for (const auto &to_remote_node : to_remote_nodes) {
    operations_.AddOperationAfter(node_index, to_remote_node);
  }
}

namespace {
// Peak strategy may need to rewrite graph outputs through wrapper nodes such as MakeTuple/Depend/Load.
bool KernelWithIndexSemanticEqual(const KernelWithIndex &lhs, const KernelWithIndex &rhs) {
  const auto real_lhs = common::AnfAlgo::FetchRealNodeSkipMonadControl(lhs);
  const auto real_rhs = common::AnfAlgo::FetchRealNodeSkipMonadControl(rhs);
  return real_lhs.first == real_rhs.first && real_lhs.second == real_rhs.second;
}

bool NodeResolvesToSingleSemanticOutput(const AnfNodePtr &node, const KernelWithIndex &target) {
  MS_EXCEPTION_IF_NULL(node);
  const auto outputs = common::AnfAlgo::GetAllOutputWithIndex(node);
  return outputs.size() == 1 && KernelWithIndexSemanticEqual(outputs[0], target);
}

bool ReplaceGraphOutputSubtree(const AnfNodePtr &node, const KernelWithIndex &target, const AnfNodePtr &replacement) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(replacement);
  auto cnode = node->cast_ptr<CNode>();
  if (cnode == nullptr) {
    return false;
  }

  auto try_replace_input = [&](size_t input_index) -> bool {
    if (input_index >= cnode->inputs().size()) {
      return false;
    }
    const auto &input = cnode->input(input_index);
    if (input == nullptr) {
      return false;
    }
    if (NodeResolvesToSingleSemanticOutput(input, target)) {
      cnode->set_input(input_index, replacement);
      return true;
    }
    return ReplaceGraphOutputSubtree(input, target, replacement);
  };

  if (IsPrimitiveCNode(node, prim::kPrimMakeTuple)) {
    bool changed = false;
    for (size_t i = 1; i < cnode->inputs().size(); ++i) {
      changed = try_replace_input(i) || changed;
    }
    return changed;
  }
  if (IsOneOfPrimitiveCNode(node, {prim::kPrimDepend, prim::kPrimLoad})) {
    return try_replace_input(kRealInputIndexInDepend);
  }
  if (IsPrimitiveCNode(node, prim::kPrimUpdateState)) {
    return try_replace_input(kUpdateStateStateInput);
  }
  return false;
}

AnfNodePtr BuildActivationDataNodeForOffload(const FuncGraphPtr &graph, const HandleNewNodeCallback &callback,
                                             const KernelWithIndex &data_node_info) {
  MS_EXCEPTION_IF_NULL(graph);
  auto data_node = data_node_info.first;
  MS_EXCEPTION_IF_NULL(data_node);
  auto data_node_abstract = data_node->abstract();
  MS_EXCEPTION_IF_NULL(data_node_abstract);
  if (data_node_abstract->isa<abstract::AbstractTensor>()) {
    return data_node;
  }
  if (!data_node_abstract->isa<abstract::AbstractTuple>()) {
    MS_LOG(EXCEPTION) << "Found non-except abstract for node " << data_node->fullname_with_scope()
                      << " with abstract: " << data_node_abstract->ToString();
  }
  auto index_node = NewValueNode(MakeValue(static_cast<int64_t>(data_node_info.second)));
  index_node->set_abstract(std::make_shared<abstract::AbstractScalar>(static_cast<int64_t>(data_node_info.second)));
  auto tuple_item_node = graph->NewCNode(
    {NewValueNode(std::make_shared<Primitive>(prim::kPrimTupleGetItem->name())), data_node, index_node});
  auto tuple_abstract = data_node_abstract->cast_ptr<abstract::AbstractTuple>();
  MS_EXCEPTION_IF_NULL(tuple_abstract);
  MS_EXCEPTION_IF_CHECK_FAIL(data_node_info.second < tuple_abstract->elements().size(),
                             "Tuple output index is out of range.");
  tuple_item_node->set_abstract(tuple_abstract->elements()[data_node_info.second]);
  if (callback != nullptr) {
    callback(tuple_item_node, data_node_info.first);
  }
  return tuple_item_node;
}

CNodePtr AddActivationD2HOperation(const FuncGraphPtr &graph, const HandleNewNodeCallback &callback,
                                   const CNodePtrList &exec_order, HyperOffloadOperations *operations,
                                   const OffloadInfo &offload_info, const AnfNodePtr &data_node,
                                   bool is_peak_strategy) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(operations);
  auto to_remote_node = BuildToHostNode(graph, data_node, callback);
  const auto &data_node_info = offload_info.data_node;
  const AnfNodePtr &d2h_after_node = (is_peak_strategy && !offload_info.replace_info_list.empty())
                                       ? offload_info.replace_info_list[0].replace_start_node.first
                                       : data_node_info.first;
  const auto d2h_after_index_opt = FindNodeIndex(exec_order, d2h_after_node);
  MS_EXCEPTION_IF_CHECK_FAIL(d2h_after_index_opt.has_value(), "Fail to find D2H insertion node in execution order.");
  operations->AddOperationAfter(d2h_after_index_opt.value(), to_remote_node);
  return to_remote_node;
}

void AddActivationReplacementOperations(const FuncGraphPtr &graph, const HandleNewNodeCallback &callback,
                                        const CNodePtrList &exec_order, HyperOffloadOperations *operations,
                                        const ReplaceInfoList &replace_info_list, const CNodePtr &to_remote_node) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(operations);
  MS_EXCEPTION_IF_NULL(to_remote_node);
  for (const auto &replace_info : replace_info_list) {
    const auto &change_nodes_infos = replace_info.replace_rest_nodes;
    MS_EXCEPTION_IF_CHECK_FAIL(!change_nodes_infos.empty(), "change_nodes should not be empty");
    auto first_change_node = change_nodes_infos[0].first;
    MS_EXCEPTION_IF_NULL(first_change_node);
    auto to_device_node = BuildToDeviceNode(graph, to_remote_node, CreateContextCallback(callback, first_change_node));
    for (const auto &change_node_info : change_nodes_infos) {
      MS_EXCEPTION_IF_NULL(change_node_info.first);
      auto change_node = change_node_info.first->cast_ptr<CNode>();
      MS_EXCEPTION_IF_NULL(change_node);
      MS_EXCEPTION_IF_CHECK_FAIL(change_node_info.second < change_node->inputs().size(),
                                 "Input index out of range when replacing activation user.");
      change_node->set_input(change_node_info.second, to_device_node);
    }
    const auto change_node_index_opt = FindNodeIndex(exec_order, first_change_node);
    if (change_node_index_opt.has_value()) {
      operations->AddOperationBefore(change_node_index_opt.value(), to_device_node);
    }
  }
}

ReplaceInfoList FilterReplaceInfosWithRefInputs(const ReplaceInfoList &replace_info_list) {
  ReplaceInfoList filtered_replace_info_list;
  for (const auto &replace_info : replace_info_list) {
    if (std::any_of(replace_info.replace_rest_nodes.begin(), replace_info.replace_rest_nodes.end(),
                    [](const auto &replace_node) { return IsRefInputUser(replace_node); })) {
      MS_LOG(INFO) << "Skip HyperOffload replace range because it would replace a ref input.";
      continue;
    }
    filtered_replace_info_list.emplace_back(replace_info);
  }
  return filtered_replace_info_list;
}

bool IsActivationGraphOutput(const std::vector<KernelWithIndex> &outputs, const KernelWithIndex &data_node_info,
                             const AnfNodePtr &data_node, bool is_peak_strategy) {
  if (is_peak_strategy) {
    return std::any_of(outputs.begin(), outputs.end(), [&data_node_info](const auto &output) {
      return KernelWithIndexSemanticEqual(output, data_node_info);
    });
  }
  return std::any_of(outputs.begin(), outputs.end(),
                     [&data_node](const auto &output) { return output.first == data_node; });
}

void AppendActivationOutputToExecOrder(const CNodePtrList &exec_order, HyperOffloadOperations *operations,
                                       const CNodePtr &to_device_node) {
  MS_EXCEPTION_IF_NULL(operations);
  MS_EXCEPTION_IF_NULL(to_device_node);
  if (exec_order.empty()) {
    return;
  }
  operations->AddOperationAfter(exec_order.size() - 1, to_device_node);
}

void AddPeakActivationOutputOperation(const FuncGraphPtr &graph, const HandleNewNodeCallback &callback,
                                      const CNodePtrList &exec_order, HyperOffloadOperations *operations,
                                      const KernelWithIndex &data_node_info, const CNodePtr &to_remote_node) {
  MS_EXCEPTION_IF_NULL(graph);
  const auto &output = graph->output();
  MS_EXCEPTION_IF_NULL(output);
  auto to_device_node = BuildToDeviceNode(graph, to_remote_node, CreateContextCallback(callback, output));
  bool output_replaced = false;
  if (NodeResolvesToSingleSemanticOutput(output, data_node_info)) {
    const auto output_cnode = output->cast_ptr<CNode>();
    if (output_cnode != nullptr && IsOneOfPrimitiveCNode(output, {prim::kPrimDepend, prim::kPrimLoad})) {
      output_replaced = ReplaceGraphOutputSubtree(output, data_node_info, to_device_node);
    } else if (output_cnode != nullptr && IsPrimitiveCNode(output, prim::kPrimUpdateState)) {
      output_replaced = ReplaceGraphOutputSubtree(output, data_node_info, to_device_node);
    } else {
      graph->set_output(to_device_node);
      output_replaced = true;
    }
  } else {
    output_replaced = ReplaceGraphOutputSubtree(output, data_node_info, to_device_node);
  }
  MS_EXCEPTION_IF_CHECK_FAIL(output_replaced, "Fail to remap graph output for activation offload.");
  AppendActivationOutputToExecOrder(exec_order, operations, to_device_node);
}

void AddDistanceActivationOutputOperation(const FuncGraphPtr &graph, const HandleNewNodeCallback &callback,
                                          const CNodePtrList &exec_order, HyperOffloadOperations *operations,
                                          const AnfNodePtr &data_node, const CNodePtr &to_remote_node) {
  MS_EXCEPTION_IF_NULL(graph);
  auto graph_output = graph->output();
  MS_EXCEPTION_IF_NULL(graph_output);
  auto output = graph_output->cast<CNodePtr>();
  MS_EXCEPTION_IF_CHECK_FAIL(IsPrimitiveCNode(output, prim::kPrimMakeTuple), "output not make tuple");
  int index = -1;
  for (size_t i = 1; i < output->inputs().size(); ++i) {
    if (output->input(i) == data_node) {
      index = SizeToInt(i);
      break;
    }
  }
  MS_EXCEPTION_IF_CHECK_FAIL(index > 0, "Fail to find index");
  auto to_device_node = BuildToDeviceNode(graph, to_remote_node, CreateContextCallback(callback, output));
  UpdateNodeInput(output, IntToSize(index), to_device_node);
  AppendActivationOutputToExecOrder(exec_order, operations, to_device_node);
}
}  // namespace

void HyperOffloadOptimizer::AddSingleActivationOperations(const OffloadInfo &offload_info,
                                                          const std::vector<KernelWithIndex> &outputs) {
  MS_EXCEPTION_IF_NULL(graph_);
  const auto &data_node_info = offload_info.data_node;
  const bool is_peak_strategy = HyperOffloadConfig::GetInstance().enable_hyper_offload_peak_strategy();
  const auto &replace_info_list = FilterReplaceInfosWithRefInputs(offload_info.replace_info_list);
  if (replace_info_list.empty()) {
    MS_LOG(INFO) << "Skip HyperOffload activation candidate because all replace ranges touch ref inputs.";
    return;
  }
  auto data_node = BuildActivationDataNodeForOffload(graph_, callback_, data_node_info);
  auto filtered_offload_info = OffloadInfo{data_node_info, replace_info_list};
  auto to_remote_node = AddActivationD2HOperation(graph_, callback_, exec_order_, &operations_, filtered_offload_info,
                                                  data_node, is_peak_strategy);
  AddActivationReplacementOperations(graph_, callback_, exec_order_, &operations_, replace_info_list, to_remote_node);
  if (!IsActivationGraphOutput(outputs, data_node_info, data_node, is_peak_strategy)) {
    return;
  }
  if (is_peak_strategy) {
    AddPeakActivationOutputOperation(graph_, callback_, exec_order_, &operations_, data_node_info, to_remote_node);
    return;
  }
  AddDistanceActivationOutputOperation(graph_, callback_, exec_order_, &operations_, data_node, to_remote_node);
}

void HyperOffloadOptimizer::AddEventNodes() {
  MS_LOG(INFO) << "Adding event nodes for synchronization";
  HyperOffloadEventInjector event_injector(graph_, exec_order_, &operations_);
  event_injector.Run();
  MS_LOG(INFO) << "Event nodes added successfully";
}

bool HyperOffloadOptimizer::SkipHyperOffloadOptimizer() {
  if (!HyperOffloadConfig::GetInstance().enable_hyper_offload()) {
    return true;
  }
  if (!enable_offload_parameter_ && !enable_offload_activation_) {
    return true;
  }
  if (graph_ == nullptr || exec_order_.empty()) {
    return true;
  }
  return false;
}

HyperOffloadPlan HyperOffloadOptimizer::Run() {
  MS_LOG(INFO) << "HyperOffloadOptimizer::Run started";
  HyperOffloadPlan plan;
  plan.graph = graph_;
  plan.exec_order = exec_order_;
  if (SkipHyperOffloadOptimizer()) {
    MS_LOG(INFO) << "SkipHyperOffloadOptimizer triggered, skip hyper offload optimization";
    return plan;
  }

  operations_.Init(plan.exec_order.size());
  if (enable_offload_activation_) {
    GenerateActivationOperations();
  }
  if (enable_offload_parameter_) {
    GenerateParameterOperations();
  }
  AdjustHyperOffloadNodePosition();
  AddEventNodes();

  plan.exec_order = BuildExecutionOrder();
  plan.hyper_offload_operations = operations_;
  AssignHyperOffloadIds(plan.exec_order);
  plan.hyper_offload_operations.Print(exec_order_);
  MS_LOG(INFO) << "HyperOffloadOptimizer::Run completed, final execution order size: " << plan.exec_order.size();
  return plan;
}

void HyperOffloadOptimizer::GenerateParameterOperations() {
  MS_EXCEPTION_IF_NULL(graph_);
  MS_LOG(DEBUG) << "Generating parameter operations for " << exec_order_.size() << " nodes";
  for (const auto &node : exec_order_) {
    AddSingleParameterOperations(node);
  }
}

void HyperOffloadOptimizer::GenerateActivationOperations() {
  MS_EXCEPTION_IF_NULL(graph_);
  const auto &parallel_context = parallel::ParallelContext::GetInstance();
  MS_EXCEPTION_IF_NULL(parallel_context);
  auto stage_num = parallel_context->pipeline_stage_split_num();
  if (stage_num > 1) {
    auto stage_id = GetPipelineStageId();
    const auto &config = HyperOffloadConfig::GetInstance();
    if (!IsStageEnabled(stage_id, config.enabled_stages())) {
      MS_LOG(INFO) << "Skip hyper offload activation optimizer for pipeline stage " << stage_id;
      return;
    }
  }

  UserInfoList user_info_list = CollectAllNodeUsers(exec_order_);

  HyperOffloadStrategyPtr strategy = nullptr;
  bool use_filter = false;
  const bool use_peak_strategy = HyperOffloadConfig::GetInstance().enable_hyper_offload_peak_strategy();
  if (use_peak_strategy) {
    std::map<AnfNodePtr, size_t> node_to_index_for_span;
    for (size_t ei = 0; ei < exec_order_.size(); ++ei) {
      node_to_index_for_span[exec_order_[ei]] = ei;
    }
    AugmentUserInfoListForOffloadSpan(&user_info_list, exec_order_, node_to_index_for_span, exec_order_.size());
    MS_LOG(INFO) << "Using MemoryPeakBasedHyperOffloadStrategy with max_device_memory="
                 << HyperOffloadConfig::GetInstance().max_device_memory();
    strategy = std::make_unique<MemoryPeakBasedHyperOffloadStrategy>();
  } else {
    MS_LOG(INFO) << "Using DistanceBaseHyperOffloadStrategy";
    strategy = std::make_unique<DistanceBaseHyperOffloadStrategy>();
    use_filter = true;
  }

  auto offload_info_list = strategy->Run(exec_order_, user_info_list, graph_);

  if (use_filter) {
    OffloadInfoFilterByNumber filter_by_number;
    offload_info_list = filter_by_number.Filter(offload_info_list);
    OffloadInfoFilterByPercentage filter_by_percentage;
    offload_info_list = filter_by_percentage.Filter(offload_info_list);
  }

  if (offload_info_list.empty()) {
    MS_LOG(INFO) << "No activation offload candidates found";
    return;
  }

  const auto &outputs = common::AnfAlgo::GetAllOutputWithIndex(graph_->output());
  MS_LOG(INFO) << "Processing " << offload_info_list.size() << " activation offload candidates";
  for (const auto &offload_info : offload_info_list) {
    AddSingleActivationOperations(offload_info, outputs);
  }
}

CNodePtrList HyperOffloadOptimizer::BuildExecutionOrder() {
  CNodePtrList new_execution_order;
  const auto &pre_operations = operations_.GetPreOperations().operations;
  const size_t appended_operation_count =
    std::accumulate(operations_.GetOperations().begin(), operations_.GetOperations().end(), static_cast<size_t>(0),
                    [](size_t acc, const NodeOperations &op_slot) { return acc + op_slot.operations.size(); });
  const size_t extra_operation_count = pre_operations.size() + appended_operation_count;
  new_execution_order.reserve(exec_order_.size() + extra_operation_count);
  std::copy(pre_operations.begin(), pre_operations.end(), std::back_inserter(new_execution_order));

  MS_EXCEPTION_IF_CHECK_FAIL(operations_.GetOperations().size() == exec_order_.size(),
                             "The size of operations is not equal to the size of original order.");

  size_t added_ops_count = 0;
  for (size_t i = 0; i < exec_order_.size(); ++i) {
    new_execution_order.push_back(exec_order_[i]);
    const auto &cur_operations = operations_.GetOperations()[i].operations;
    added_ops_count += cur_operations.size();
    std::copy(cur_operations.begin(), cur_operations.end(), std::back_inserter(new_execution_order));
  }
  MS_LOG(DEBUG) << "Built execution order: " << exec_order_.size() << " original + " << added_ops_count
                << " added = " << new_execution_order.size() << " total";
  return new_execution_order;
}

void HyperOffloadOptimizer::AssignHyperOffloadIds(const CNodePtrList &exec_order) {
  std::unordered_map<AnfNodePtr, std::string> node_to_key_cache;
  std::unordered_map<size_t, std::string> id_to_key_map;

  std::function<std::string(const AnfNodePtr &)> get_key = [&](const AnfNodePtr &node) -> std::string {
    if (node == nullptr) {
      return "";
    }
    if (node_to_key_cache.count(node)) {
      return node_to_key_cache[node];
    }

    std::string key;
    if (IsPrimitiveCNode(node, prim::kPrimTupleGetItem)) {
      auto cnode = node->cast<CNodePtr>();
      MS_EXCEPTION_IF_CHECK_FAIL(cnode->inputs().size() > 1, "Node inputs size check failed.");
      auto index = common::AnfAlgo::GetTupleGetItemOutIndex(cnode);
      key = get_key(cnode->input(1)) + "_idx_" + std::to_string(index);
    } else if (IsD2HNode(node) || IsH2DNode(node) ||
               IsOneOfPrimitiveCNode(node,
                                     {kFreeDataPrimitive, kAssignDataPrimitive, prim::kPrimSend, prim::kPrimReceive})) {
      auto cnode = node->cast<CNodePtr>();
      MS_EXCEPTION_IF_CHECK_FAIL(cnode->inputs().size() > 1, "Node inputs size check failed.");
      key = get_key(cnode->input(1));
    } else {
      key = node->fullname_with_scope();
    }

    node_to_key_cache[node] = key;
    return key;
  };

  for (const auto &node : exec_order) {
    auto key = get_key(node);
    if (key.empty()) {
      continue;
    }
    size_t id = std::hash<std::string>{}(key);
    while (id_to_key_map.find(id) != id_to_key_map.end() && id_to_key_map[id] != key) {
      id++;
    }
    id_to_key_map[id] = key;
    node->set_user_data(kHyperOffloadId, std::make_shared<size_t>(id));
  }
}

void HyperOffloadOptimizer::AdjustHyperOffloadNodePosition() {
  const bool use_peak_strategy = HyperOffloadConfig::GetInstance().enable_hyper_offload_peak_strategy();
  if (use_peak_strategy) {
    MS_LOG(DEBUG) << "Peak strategy enabled with max_device_memory="
                  << HyperOffloadConfig::GetInstance().max_device_memory();
  }
  const auto &h2d_nodes = operations_.GetAllH2DNodes();
  MS_LOG(DEBUG) << "Adjusting " << h2d_nodes.size() << " H2D node positions";
  for (const auto &h2d_node : h2d_nodes) {
    MS_EXCEPTION_IF_CHECK_FAIL(h2d_node->inputs().size() > kPrimaryDataInputIndex, "H2D node input size invalid.");
    auto input_node = h2d_node->input(kPrimaryDataInputIndex);
    AnfNodePtr data_node = input_node;
    CNodePtr d2h_node = nullptr;
    if (IsD2HNode(input_node)) {
      d2h_node = input_node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(d2h_node);
      MS_EXCEPTION_IF_CHECK_FAIL(d2h_node->inputs().size() > kPrimaryDataInputIndex, "D2H node input size invalid.");
      data_node = d2h_node->input(kPrimaryDataInputIndex);
    }
    const size_t prefetch_req =
      GetNodePrefetchDistance(data_node).value_or(HyperOffloadConfig::GetInstance().prefetch_distance());
    std::optional<size_t> min_after_exec_idx = (use_peak_strategy && d2h_node != nullptr)
                                                 ? std::make_optional(operations_.GetNodeAttachIndex(d2h_node))
                                                 : std::nullopt;
    const auto branch_start_attach_idx = GetInlineBranchStartAttachIndex(graph_, exec_order_, h2d_node);
    if (branch_start_attach_idx.has_value()) {
      min_after_exec_idx = min_after_exec_idx.has_value()
                             ? std::make_optional(std::max(*min_after_exec_idx, *branch_start_attach_idx))
                             : branch_start_attach_idx;
    }
    operations_.MoveAhead(h2d_node, prefetch_req, nullptr, min_after_exec_idx);
  }
}
}  // namespace hyper_offload
}  // namespace utils
}  // namespace mindspore
