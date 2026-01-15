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

#include <unordered_map>
#include "include/utils/anfalgo.h"
#include "primitive/framework_ops.h"
#include "primitive/other_ops.h"
#include "utils/hyper_offload/hyper_offload_operation.h"

namespace mindspore {
namespace utils {
namespace hyper_offload {
namespace {
void AddSingleParameterOperations(const FuncGraphPtr &graph, const CNodePtr &node, const CNodePtrList &exec_order,
                                  const HandleNewNodeCallback &callback, HyperOffloadOperations *operations) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  if (IsPrimitiveCNode(node, prim::kPrimConditionSwitch)) {
    return;
  }
  auto node_iter = std::find(exec_order.begin(), exec_order.end(), node);
  if (node_iter == exec_order.end()) {
    return;
  }
  size_t node_index = std::distance(exec_order.begin(), node_iter);
  CNodePtrList to_remote_nodes{};
  for (size_t i = 0; i < node->inputs().size(); ++i) {
    const auto &input = node->input(i);
    auto real_input = common::AnfAlgo::VisitKernelWithReturnType(input, 0, false, {prim::kPrimLoad}).first;
    if (IsPrimitiveCNode(real_input, prim::kPrimLoad)) {
      constexpr size_t data_index = 1;
      auto data_node = real_input->cast<CNodePtr>()->input(data_index);
      if (!GetRemoteParameter(graph, data_node).has_value()) {
        continue;
      }
      auto to_device_node = BuildToDeviceNode(graph, data_node, callback);
      to_device_node->set_abstract(real_input->abstract());
      UpdateNodeInput(node, i, to_device_node);
      operations->AddOperationBefore(node_index, to_device_node);
      continue;
    }
    const auto &param_res = GetRemoteParameter(graph, input);
    if (!param_res.has_value()) {
      continue;
    }
    auto to_device_node = BuildToDeviceNode(graph, input, callback);
    to_device_node->set_abstract(dyn_cast<abstract::AbstractRefTensor>(input->abstract())->CloneAsTensor());
    UpdateNodeInput(node, i, to_device_node);
    operations->AddOperationBefore(node_index, to_device_node);
    auto to_remote_node = BuildInplaceToHostNode(graph, to_device_node, param_res.value(), node, callback);
    to_remote_nodes.push_back(to_remote_node);
  }

  for (const auto &to_remote_node : to_remote_nodes) {
    operations->AddOperationAfter(node_index, to_remote_node);
  }
  return;
}

void AddSingleActivationOperations(const OffloadInfo &offload_info, const std::vector<KernelWithIndex> &outputs,
                                   const FuncGraphPtr &graph, const CNodePtrList &exec_order,
                                   const HandleNewNodeCallback &callback, HyperOffloadOperations *operations) {
  MS_EXCEPTION_IF_NULL(graph);
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
    data_node = graph->NewCNode(
      {NewValueNode(std::make_shared<Primitive>(prim::kPrimTupleGetItem->name())), data_node, index_node});
    data_node->set_abstract(data_node_abstract->cast<abstract::AbstractTuplePtr>()->elements()[data_node_info.second]);
    // Also notify callback about the tuplegetitem node
    if (callback != nullptr) {
      callback(data_node->cast<CNodePtr>(), data_node_info.first);
    }
  }

  auto to_remote_node = BuildToHostNode(graph, data_node, callback);

  auto data_iter = std::find(exec_order.begin(), exec_order.end(), data_node_info.first);
  MS_EXCEPTION_IF_CHECK_FAIL(data_iter != exec_order.end(), "Fail to find node in execution order.");
  size_t data_node_index = std::distance(exec_order.begin(), data_iter);
  operations->AddOperationAfter(data_node_index, to_remote_node);

  for (const auto &cur_replace_info : offload_info.replace_info_list) {
    auto to_device_node = BuildToDeviceNode(graph, to_remote_node, callback);
    const auto &change_nodes_infos = cur_replace_info.replace_rest_nodes;
    MS_EXCEPTION_IF_CHECK_FAIL(!change_nodes_infos.empty(), "change_nodes should not be empty");
    for (const auto &change_node_info : change_nodes_infos) {
      auto change_node = change_node_info.first->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(change_node);
      int index = change_node_info.second;
      MS_EXCEPTION_IF_CHECK_FAIL(index >= 0, "fail to found index.");
      UpdateNodeInput(change_node, index, to_device_node);
    }
    auto change_node_iter = std::find(exec_order.begin(), exec_order.end(), change_nodes_infos[0].first);
    if (change_node_iter != exec_order.end()) {
      size_t change_node_index = std::distance(exec_order.begin(), change_node_iter);
      operations->AddOperationBefore(change_node_index, to_device_node);
    }
  }

  bool is_output =
    std::any_of(outputs.begin(), outputs.end(), [&data_node](const auto &e) { return e.first == data_node; });
  if (is_output) {
    auto output = graph->output()->cast<CNodePtr>();
    MS_EXCEPTION_IF_CHECK_FAIL(IsPrimitiveCNode(output, prim::kPrimMakeTuple), "output not make tuple");
    int index = -1;
    for (size_t i = 1; i < output->inputs().size(); ++i) {
      if (output->input(i) == data_node) {
        index = SizeToInt(i);
        break;
      }
    }
    MS_EXCEPTION_IF_CHECK_FAIL(index > 0, "Fail to find index");
    auto to_device_node = BuildToDeviceNode(graph, to_remote_node, callback);
    UpdateNodeInput(output, IntToSize(index), to_device_node);
    if (!exec_order.empty()) {
      size_t last_index = exec_order.size() - 1;
      operations->AddOperationAfter(last_index, to_device_node);
    }
  }
  return;
}
}  // namespace

HyperOffloadPlan HyperOffloadOptimizer::Run(const HyperOffloadInput &input) {
  HyperOffloadPlan plan;
  plan.graph = input.graph;
  plan.exec_order = input.exec_order;
  plan.event_plan.clear();
  if (!HyperOffloadConfig::GetInstance().enable_hyper_offload()) {
    return plan;
  }
  if (!input.enable_offload_parameter && !input.enable_offload_activation) {
    return plan;
  }
  if (plan.graph == nullptr || plan.exec_order.empty()) {
    return plan;
  }

  HyperOffloadOperations hyper_offload_operations;
  hyper_offload_operations.Init(plan.exec_order.size());
  if (input.enable_offload_activation) {
    auto act_ops = GenerateActivationOperations(plan.graph, plan.exec_order, input.callback);
    hyper_offload_operations.MergeFrom(act_ops);
  }
  if (input.enable_offload_parameter) {
    auto param_ops = GenerateParameterOperations(plan.graph, plan.exec_order, input.callback);
    hyper_offload_operations.MergeFrom(param_ops);
  }

  AdjustHyperOffloadNodePosition(&hyper_offload_operations);
  plan.exec_order = BuildExecutionOrder(plan.exec_order, hyper_offload_operations);
  plan.event_plan = GenerateEventPlan(plan.graph, plan.exec_order);
  plan.hyper_offload_operations = hyper_offload_operations;
  return plan;
}

HyperOffloadOperations HyperOffloadOptimizer::GenerateParameterOperations(const FuncGraphPtr &graph,
                                                                          const CNodePtrList &exec_order,
                                                                          const HandleNewNodeCallback &callback) {
  MS_EXCEPTION_IF_NULL(graph);
  HyperOffloadOperations params_operation;
  params_operation.Init(exec_order.size());
  for (const auto &node : exec_order) {
    AddSingleParameterOperations(graph, node, exec_order, callback, &params_operation);
  }
  return params_operation;
}

HyperOffloadOperations HyperOffloadOptimizer::GenerateActivationOperations(const FuncGraphPtr &graph,
                                                                           const CNodePtrList &exec_order,
                                                                           const HandleNewNodeCallback &callback) {
  MS_EXCEPTION_IF_NULL(graph);
  HyperOffloadOperations activation_operation;
  activation_operation.Init(exec_order.size());

  const auto &user_info_list = CollectAllNodeUsers(exec_order);
  DistanceBaseHyperOffloadStrategy strategy;
  auto offload_info_list = strategy.Run(exec_order, user_info_list);
  OffloadInfoFilterByNumber filter;
  offload_info_list = filter.Filter(offload_info_list);

  if (offload_info_list.empty()) {
    return activation_operation;
  }

  const auto &outputs = common::AnfAlgo::GetAllOutputWithIndex(graph->output());
  for (const auto &offload_info : offload_info_list) {
    AddSingleActivationOperations(offload_info, outputs, graph, exec_order, callback, &activation_operation);
  }
  return activation_operation;
}

CNodePtrList HyperOffloadOptimizer::BuildExecutionOrder(const CNodePtrList &execution_order,
                                                        const HyperOffloadOperations &hyper_offload_operations) {
  CNodePtrList new_execution_order;
  const auto &pre_operations = hyper_offload_operations.GetPreOperations().operations;
  std::copy(pre_operations.begin(), pre_operations.end(), std::back_inserter(new_execution_order));

  MS_EXCEPTION_IF_CHECK_FAIL(hyper_offload_operations.GetOperations().size() == execution_order.size(),
                             "The size of operations is not equal to the size of original order.");

  for (size_t i = 0; i < execution_order.size(); ++i) {
    new_execution_order.push_back(execution_order[i]);
    const auto &cur_operations = hyper_offload_operations.GetOperations()[i].operations;
    std::copy(cur_operations.begin(), cur_operations.end(), std::back_inserter(new_execution_order));
  }
  return new_execution_order;
}

void HyperOffloadOptimizer::AdjustHyperOffloadNodePosition(HyperOffloadOperations *hyper_offload_operations) {
  MS_EXCEPTION_IF_NULL(hyper_offload_operations);
  const auto &h2d_nodes = hyper_offload_operations->GetAllH2DNodes();
  static size_t prefetch_distance = HyperOffloadConfig::GetInstance().prefetch_distance();
  for (const auto &h2d_node : h2d_nodes) {
    hyper_offload_operations->MoveAhead(h2d_node, prefetch_distance);
  }
}

std::vector<EventPlan> HyperOffloadOptimizer::GenerateEventPlan(const FuncGraphPtr &graph,
                                                                const CNodePtrList &exec_order) {
  MS_EXCEPTION_IF_NULL(graph);
  std::vector<EventPlan> events;
  if (exec_order.empty()) {
    return events;
  }

  const auto &all_node_users = CollectAllNodeUsers(exec_order);
  static size_t release_distance = HyperOffloadConfig::GetInstance().release_distance();

  for (const auto &node : exec_order) {
    if (!IsD2HNode(node)) {
      continue;
    }
    constexpr size_t data_index = 1;
    auto data_node = node->input(data_index);
    auto data_node_last_user = FindLastRealUser(data_node, all_node_users);
    CNodePtr insert_node = nullptr;

    auto find_insert_node = [&](const AnfNodePtr &start_node) -> CNodePtr {
      auto iter = std::find(exec_order.begin(), exec_order.end(), start_node);
      if (iter == exec_order.end()) {
        return nullptr;
      }

      size_t index = std::distance(exec_order.begin(), iter);
      size_t target_index = index + release_distance;

      if (target_index >= exec_order.size()) {
        return exec_order.back();
      }
      return exec_order[target_index];
    };

    if (data_node_last_user.has_value()) {
      auto last_user_node = data_node_last_user.value();
      insert_node = find_insert_node(last_user_node);
    } else {
      insert_node = find_insert_node(node);
    }

    if (insert_node != nullptr) {
      EventPlan hint;
      hint.send_insert_after = node;
      hint.recv_insert_after = insert_node;
      events.push_back(hint);
    }
  }
  return events;
}
}  // namespace hyper_offload
}  // namespace utils
}  // namespace mindspore
