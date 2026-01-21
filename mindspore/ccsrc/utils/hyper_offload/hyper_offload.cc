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

#include <atomic>
#include <unordered_map>
#include "include/utils/anfalgo.h"
#include "primitive/framework_ops.h"
#include "primitive/other_ops.h"
#include "utils/hyper_offload/hyper_offload_operation.h"
#include "utils/hyper_offload/hyper_offload_event.h"

namespace mindspore {
namespace utils {
namespace hyper_offload {
HyperOffloadOptimizer::HyperOffloadOptimizer(const HyperOffloadInput &input)
    : graph_(input.graph),
      exec_order_(input.exec_order),
      callback_(input.callback),
      enable_offload_parameter_(input.enable_offload_parameter),
      enable_offload_activation_(input.enable_offload_activation) {}

void HyperOffloadOptimizer::AddSingleParameterOperations(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(graph_);
  MS_EXCEPTION_IF_NULL(node);
  if (IsPrimitiveCNode(node, prim::kPrimConditionSwitch)) {
    return;
  }
  auto node_iter = std::find(exec_order_.begin(), exec_order_.end(), node);
  if (node_iter == exec_order_.end()) {
    return;
  }
  size_t node_index = std::distance(exec_order_.begin(), node_iter);
  CNodePtrList to_remote_nodes{};
  for (size_t i = 0; i < node->inputs().size(); ++i) {
    const auto &input = node->input(i);
    auto real_input = common::AnfAlgo::VisitKernelWithReturnType(input, 0, false, {prim::kPrimLoad}).first;
    if (IsPrimitiveCNode(real_input, prim::kPrimLoad)) {
      constexpr size_t data_index = 1;
      auto data_node = real_input->cast<CNodePtr>()->input(data_index);
      if (!GetRemoteParameter(graph_, data_node).has_value()) {
        continue;
      }
      auto to_device_node = BuildToDeviceNode(graph_, data_node, callback_);
      to_device_node->set_abstract(real_input->abstract());
      UpdateNodeInput(node, i, to_device_node);
      operations_.AddOperationBefore(node_index, to_device_node);
      continue;
    }
    const auto &param_res = GetRemoteParameter(graph_, input);
    if (!param_res.has_value()) {
      continue;
    }
    auto to_device_node = BuildToDeviceNode(graph_, input, callback_);
    to_device_node->set_abstract(dyn_cast<abstract::AbstractRefTensor>(input->abstract())->CloneAsTensor());
    UpdateNodeInput(node, i, to_device_node);
    operations_.AddOperationBefore(node_index, to_device_node);
    auto to_remote_node = BuildInplaceToHostNode(graph_, to_device_node, param_res.value(), node, callback_);
    to_remote_nodes.push_back(to_remote_node);
  }

  for (const auto &to_remote_node : to_remote_nodes) {
    operations_.AddOperationAfter(node_index, to_remote_node);
  }
  return;
}

void HyperOffloadOptimizer::AddSingleActivationOperations(const OffloadInfo &offload_info,
                                                          const std::vector<KernelWithIndex> &outputs) {
  MS_EXCEPTION_IF_NULL(graph_);
  const auto &data_node_info = offload_info.data_node;
  auto data_node = data_node_info.first;
  auto data_node_abstract = data_node->abstract();
  MS_EXCEPTION_IF_NULL(data_node_abstract);

  if (!data_node_abstract->isa<abstract::AbstractTensor>()) {
    if (!data_node_abstract->isa<abstract::AbstractTuple>()) {
      MS_LOG(EXCEPTION) << "Found non-except abstract for node " << data_node->fullname_with_scope()
                        << " with abstract: " << data_node_abstract->ToString();
    }
    auto index_node = NewValueNode(MakeValue(int64_t(data_node_info.second)));
    index_node->set_abstract(std::make_shared<abstract::AbstractScalar>(int64_t(data_node_info.second)));
    data_node = graph_->NewCNode(
      {NewValueNode(std::make_shared<Primitive>(prim::kPrimTupleGetItem->name())), data_node, index_node});
    data_node->set_abstract(data_node_abstract->cast<abstract::AbstractTuplePtr>()->elements()[data_node_info.second]);
    // Also notify callback about the tuplegetitem node
    if (callback_ != nullptr) {
      callback_(data_node->cast<CNodePtr>(), data_node_info.first);
    }
  }

  auto to_remote_node = BuildToHostNode(graph_, data_node, callback_);

  auto data_iter = std::find(exec_order_.begin(), exec_order_.end(), data_node_info.first);
  MS_EXCEPTION_IF_CHECK_FAIL(data_iter != exec_order_.end(), "Fail to find node in execution order.");
  size_t data_node_index = std::distance(exec_order_.begin(), data_iter);
  operations_.AddOperationAfter(data_node_index, to_remote_node);

  for (const auto &cur_replace_info : offload_info.replace_info_list) {
    auto to_device_node = BuildToDeviceNode(graph_, to_remote_node, callback_);
    const auto &change_nodes_infos = cur_replace_info.replace_rest_nodes;
    MS_EXCEPTION_IF_CHECK_FAIL(!change_nodes_infos.empty(), "change_nodes should not be empty");
    for (const auto &change_node_info : change_nodes_infos) {
      auto change_node = change_node_info.first->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(change_node);
      int index = change_node_info.second;
      MS_EXCEPTION_IF_CHECK_FAIL(index >= 0, "fail to found index.");
      UpdateNodeInput(change_node, index, to_device_node);
    }
    auto change_node_iter = std::find(exec_order_.begin(), exec_order_.end(), change_nodes_infos[0].first);
    if (change_node_iter != exec_order_.end()) {
      size_t change_node_index = std::distance(exec_order_.begin(), change_node_iter);
      operations_.AddOperationBefore(change_node_index, to_device_node);
    }
  }

  bool is_output =
    std::any_of(outputs.begin(), outputs.end(), [&data_node](const auto &e) { return e.first == data_node; });
  if (is_output) {
    auto output = graph_->output()->cast<CNodePtr>();
    MS_EXCEPTION_IF_CHECK_FAIL(IsPrimitiveCNode(output, prim::kPrimMakeTuple), "output not make tuple");
    int index = -1;
    for (size_t i = 1; i < output->inputs().size(); ++i) {
      if (output->input(i) == data_node) {
        index = SizeToInt(i);
        break;
      }
    }
    MS_EXCEPTION_IF_CHECK_FAIL(index > 0, "Fail to find index");
    auto to_device_node = BuildToDeviceNode(graph_, to_remote_node, callback_);
    UpdateNodeInput(output, IntToSize(index), to_device_node);
    if (!exec_order_.empty()) {
      size_t last_index = exec_order_.size() - 1;
      operations_.AddOperationAfter(last_index, to_device_node);
    }
  }
  return;
}

void HyperOffloadOptimizer::AddEventNodes() {
  HyperOffloadEventInjector event_injector(graph_, exec_order_, &operations_);
  event_injector.Run();
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
  HyperOffloadPlan plan;
  plan.graph = graph_;
  plan.exec_order = exec_order_;
  if (SkipHyperOffloadOptimizer()) {
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
  MS_LOG(INFO) << plan.hyper_offload_operations.ToString(exec_order_);
  return plan;
}

void HyperOffloadOptimizer::GenerateParameterOperations() {
  MS_EXCEPTION_IF_NULL(graph_);
  for (const auto &node : exec_order_) {
    AddSingleParameterOperations(node);
  }
}

void HyperOffloadOptimizer::GenerateActivationOperations() {
  MS_EXCEPTION_IF_NULL(graph_);

  const auto &user_info_list = CollectAllNodeUsers(exec_order_);
  DistanceBaseHyperOffloadStrategy strategy;
  auto offload_info_list = strategy.Run(exec_order_, user_info_list);
  OffloadInfoFilterByNumber filter;
  offload_info_list = filter.Filter(offload_info_list);

  if (offload_info_list.empty()) {
    return;
  }

  const auto &outputs = common::AnfAlgo::GetAllOutputWithIndex(graph_->output());
  for (const auto &offload_info : offload_info_list) {
    AddSingleActivationOperations(offload_info, outputs);
  }
}

CNodePtrList HyperOffloadOptimizer::BuildExecutionOrder() {
  CNodePtrList new_execution_order;
  const auto &pre_operations = operations_.GetPreOperations().operations;
  std::copy(pre_operations.begin(), pre_operations.end(), std::back_inserter(new_execution_order));

  MS_EXCEPTION_IF_CHECK_FAIL(operations_.GetOperations().size() == exec_order_.size(),
                             "The size of operations is not equal to the size of original order.");

  for (size_t i = 0; i < exec_order_.size(); ++i) {
    new_execution_order.push_back(exec_order_[i]);
    const auto &cur_operations = operations_.GetOperations()[i].operations;
    std::copy(cur_operations.begin(), cur_operations.end(), std::back_inserter(new_execution_order));
  }
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
  const auto &h2d_nodes = operations_.GetAllH2DNodes();
  const size_t prefetch_distance = HyperOffloadConfig::GetInstance().prefetch_distance();
  for (const auto &h2d_node : h2d_nodes) {
    operations_.MoveAhead(h2d_node, prefetch_distance);
  }
}
}  // namespace hyper_offload
}  // namespace utils
}  // namespace mindspore
