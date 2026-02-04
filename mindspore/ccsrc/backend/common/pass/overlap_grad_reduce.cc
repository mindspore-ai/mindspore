/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include "backend/common/pass/overlap_grad_reduce.h"

#include <memory>
#include <queue>
#include <vector>
#include <utility>
#include <algorithm>

#include "primitive/nn_ops.h"
#include "primitive/other_ops.h"
#include "ir/graph_utils.h"
#include "include/utils/utils.h"
#include "include/utils/parallel_context.h"
#include "include/runtime/hardware_abstract/kernel_base/graph_fusion/graph_kernel_flags.h"
#include "mindspore/core/include/utils/ms_context.h"
#include "include/utils/anfalgo.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"
#include "primitive/auto_generate/gen_ops_primitive_b.h"
#include "primitive/auto_generate/gen_ops_primitive_c.h"
#include "primitive/auto_generate/gen_ops_primitive_d.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"
#include "primitive/auto_generate/gen_ops_primitive_r.h"
#include "primitive/auto_generate/gen_ops_primitive_t.h"
#include "primitive/auto_generate/gen_ops_primitive_u.h"

namespace mindspore {
namespace opt {
namespace {
constexpr char kAccuGradsPrefix[] = "accu_grads.";
typedef struct GradReduceUser {
 public:
  GradReduceUser() : latest_dw_execute_order(0) {}
  GradReduceUser(const std::string &param_name_, const CNodePtrList &assign_add_list_,
                 const CNodePtrList &grad_reduce_list_, const AnfNodePtrList &latest_dw_compute_nodes_,
                 const AnfNodePtrList &latest_assign_add_nodes_, size_t latest_execute_order_)
      : param_name(param_name_),
        assign_add_list(assign_add_list_),
        grad_reduce_list(grad_reduce_list_),
        latest_dw_compute_nodes(latest_dw_compute_nodes_),
        latest_assign_add_nodes(latest_assign_add_nodes_),
        latest_dw_execute_order(latest_execute_order_) {}

  std::string param_name;
  CNodePtrList assign_add_list;
  CNodePtrList grad_reduce_list;
  AnfNodePtrList latest_dw_compute_nodes;
  AnfNodePtrList latest_assign_add_nodes;
  size_t latest_dw_execute_order;
} GradReduceUser;

bool IsFromParallelOptimizerRs(const AnfNodePtr &node) {
  if (!IsPrimitiveCNode(node, prim::kPrimReduceScatter)) {
    return false;
  }
  auto prim = GetCNodePrimitive(node->cast<CNodePtr>());
  MS_EXCEPTION_IF_NULL(prim);
  if (prim->instance_name().find("grad_parallel_optimizer") == std::string::npos) {
    return false;
  }
  return true;
}

bool IsRecomputeNode(const AnfNodePtr &node) {
  if (node == nullptr) {
    return false;
  }
  auto cnode = node->cast<CNodePtr>();
  return cnode != nullptr && cnode->HasAttr(kAttrDuplicated);
}

bool IsFromGradMirrorAR(const AnfNodePtr &node) {
  if (!IsPrimitiveCNode(node, prim::kPrimAllReduce)) {
    return false;
  }
  auto prim = GetCNodePrimitive(node->cast<CNodePtr>());
  MS_EXCEPTION_IF_NULL(prim);
  if (prim->instance_name().find("grad_mirror") == std::string::npos) {
    return false;
  }
  return true;
}

std::optional<std::string> GetRefKeyFromNode(const AnfNodePtr &node) {
  auto abs = node->abstract();
  if (abs == nullptr) {
    // Abstract for some depend node are not proper set, we follow its input.
    if (IsPrimitiveCNode(node, prim::kPrimDepend)) {
      return GetRefKeyFromNode(node->cast<CNodePtr>());
    }
    // Abstract should be set except UpdateState nodes.
    if (!IsPrimitiveCNode(node, prim::kPrimUpdateState)) {
      MS_LOG(WARNING) << "Abstract not set for " << node->DebugString();
    }
    return std::nullopt;
  }
  auto abs_ref = abs->cast<abstract::AbstractRefPtr>();
  if (abs_ref == nullptr) {
    return std::nullopt;
  }
  auto ref_key = abs_ref->ref_key_value()->cast<StringImmPtr>();
  if (ref_key == nullptr) {
    return std::nullopt;
  }
  return ref_key->value();
}

std::optional<std::string> GetMirrorUserIdFromCNode(const CNodePtr &cnode) {
  if (!cnode->HasPrimalAttr(kPrimalAttrMirrorUserId)) {
    return std::nullopt;
  }
  auto mirror_user_id = GetValue<std::string>(cnode->GetPrimalAttr(kPrimalAttrMirrorUserId));
  return mirror_user_id;
}

std::vector<std::pair<AnfNodePtr, int>> GetOutputNodesWithFilter(const AnfNodePtr &node,
                                                                 std::function<bool(const AnfNodePtr &)> filter) {
  auto func_graph = node->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  std::vector<std::pair<AnfNodePtr, int>> res;
  std::queue<AnfNodePtr> anf_queue;
  anf_queue.push(node);
  while (!anf_queue.empty()) {
    auto queue_end = anf_queue.front();
    anf_queue.pop();
    auto user_set = manager->node_users()[queue_end];
    for (auto &pair : user_set) {
      if (filter(pair.first)) {
        anf_queue.push(pair.first);
        continue;
      }
      res.push_back(pair);
    }
  }
  return res;
}

bool IsGraphKernelNode(const CNodePtr &cnode) {
  if (cnode == nullptr) {
    return false;
  }
  auto prim = GetCNodePrimitive(cnode);
  MS_EXCEPTION_IF_NULL(prim);
  return prim->HasAttr(kAttrFuncGraph) && prim->GetAttr(kAttrFuncGraph) != nullptr;
}

AnfNodePtrList UpperSearchWithFilter(const AnfNodePtrList &node_list, size_t input_index, FilterFunc skip_condition) {
  AnfNodePtrList node_list_cpy(node_list);
  if (node_list.empty()) {
    return AnfNodePtrList();
  }
  AnfNodePtr cur_node;
  while (!node_list_cpy.empty()) {
    auto cnode = node_list_cpy.back()->cast<CNodePtr>();
    node_list_cpy.pop_back();
    MS_EXCEPTION_IF_NULL(cnode);
    cur_node = cnode->input(input_index);
    input_index = kIndex1;
    while (skip_condition(cur_node)) {
      auto cur_cnode = cur_node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cur_cnode);
      cur_node = cur_cnode->input(kIndex1);
    }
    if (IsGraphKernelNode(cur_node->cast<CNodePtr>())) {
      node_list_cpy.push_back(cur_node);
      auto sub_graph = GetValue<FuncGraphPtr>(GetCNodePrimitive(cur_node)->GetAttr(kAttrFuncGraph));
      node_list_cpy.push_back(sub_graph->get_return());
      continue;
    }
    const auto &func_graph_inputs = cnode->func_graph()->get_inputs();
    auto iter = std::find_if(func_graph_inputs.begin(), func_graph_inputs.end(),
                             [&cur_node](const AnfNodePtr &node) { return node == cur_node; });
    if (iter != func_graph_inputs.end()) {
      input_index = LongToSize(iter - func_graph_inputs.begin() + 1);
      continue;
    }
    break;
  }
  node_list_cpy.push_back(cur_node);
  return node_list_cpy;
}

AnfNodePtrList GetDwComputeNode(const AnfNodePtrList &node_list, size_t input_index) {
  return UpperSearchWithFilter(node_list, input_index, [](const AnfNodePtr &node) {
    return IsOneOfPrimitiveCNode(node, {prim::kPrimDepend, prim::kPrimCast, prim::kPrimTupleGetItem});
  });
}

CNodePtr FindDxMatMulByDw(const AnfNodePtrList &dw_nodes) {
  if (dw_nodes.empty()) {
    return nullptr;
  }
  auto dw_node = dw_nodes.back();
  MS_EXCEPTION_IF_NULL(dw_node);
  auto dw_cnode = dw_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(dw_cnode);
  if (!IsOneOfPrimitiveCNode(
        dw_node, {prim::kPrimMatMul, prim::kPrimMatMulExt, prim::kPrimBatchMatMul, prim::kPrimBatchMatMulExt}) ||
      !dw_cnode->HasPrimalAttr(kPrimalAttrForwardUniqueId)) {
    return nullptr;
  }

  auto dw_prim = GetCNodePrimitive(dw_cnode);
  MS_EXCEPTION_IF_NULL(dw_prim);
  auto dw_forward_unique_id = GetValue<std::string>(dw_cnode->GetPrimalAttr(kPrimalAttrForwardUniqueId));

  auto common_inputs = UpperSearchWithFilter(
    dw_nodes, kIndex1, [](const AnfNodePtr &node) { return IsPrimitiveCNode(node, prim::kPrimDepend); });

  auto common_input_user_list = GetOutputNodesWithFilter(
    common_inputs.front(), [](const AnfNodePtr &node) { return IsOneOfPrimitiveCNode(node, {prim::kPrimDepend}); });
  CNodePtr dx_cnode = nullptr;
  auto is_dx_node_for_dw = [&dw_cnode, &dw_forward_unique_id](const AnfNodePtr &dx_node) {
    if (!IsPrimitiveCNode(dx_node, GetCNodePrimitive(dw_cnode)) || dw_cnode == dx_node) {
      return false;
    }
    auto dx_cnode = dx_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(dx_cnode);
    return dx_cnode->HasPrimalAttr(kPrimalAttrForwardUniqueId) &&
           GetValue<std::string>(dx_cnode->GetPrimalAttr(kPrimalAttrForwardUniqueId)) == dw_forward_unique_id;
  };
  for (const auto &common_input_user_pair : common_input_user_list) {
    auto user_node = common_input_user_pair.first;
    MS_EXCEPTION_IF_NULL(user_node);
    auto user_cnode = user_node->cast<CNodePtr>();
    if (!user_cnode) {
      continue;
    }
    if (IsGraphKernelNode(user_cnode)) {
      auto sub_graph_user_cnode = GetCNodePrimitive(user_cnode);
      MS_EXCEPTION_IF_NULL(sub_graph_user_cnode);
      auto sub_graph = GetValue<FuncGraphPtr>(sub_graph_user_cnode->GetAttr(kAttrFuncGraph));
      auto common_input_in_sub_graph = sub_graph->get_inputs().at(common_input_user_pair.second - 1);
      const auto &sub_graph_cnode_list = sub_graph->GetOrderedCnodes();
      if (std::find_if(sub_graph_cnode_list.begin(), sub_graph_cnode_list.end(),
                       [&is_dx_node_for_dw, &common_input_in_sub_graph](const CNodePtr &sub_graph_cnode) {
                         const auto &sub_graph_cnode_inputs = sub_graph_cnode->inputs();
                         return std::find(sub_graph_cnode_inputs.begin(), sub_graph_cnode_inputs.end(),
                                          common_input_in_sub_graph) != sub_graph_cnode_inputs.end() &&
                                is_dx_node_for_dw(sub_graph_cnode);
                       }) != sub_graph_cnode_list.end()) {
        return user_cnode;
      }
    } else {
      if (is_dx_node_for_dw(user_cnode)) {
        return user_cnode;
      }
    }
  }
  return nullptr;
}

// Return true if it satisfy one of condition.
// 1. The input node is AssignAdd CNode and AssignAdd->input(1) is accu_grad
// 2. The input node is a GraphKernel and it has structure Add->Assign and Assign->input(1) is accu_grad.
std::optional<std::string> ExtractAccuRefKeyFromAssignAddCNode(const CNodePtr &cnode,
                                                               AnfNodePtrList *assign_add_nodes) {
  MS_EXCEPTION_IF_NULL(cnode);
  CNodePtr assign_cnode = nullptr;
  auto prim = GetCNodePrimitive(cnode);
  MS_EXCEPTION_IF_NULL(prim);
  if (IsPrimitiveCNode(cnode, prim::kPrimAssignAdd)) {
    assign_cnode = cnode;
    assign_add_nodes->push_back(cnode);
  } else if (IsGraphKernelNode(cnode)) {
    auto sub_graph = GetValue<FuncGraphPtr>(prim->GetAttr(kAttrFuncGraph));
    const auto &sub_graph_cnode_list = sub_graph->GetOrderedCnodes();
    // Add -> Assign
    auto target = std::find_if(sub_graph_cnode_list.begin(), sub_graph_cnode_list.end(), [](const CNodePtr &cur_cnode) {
      return IsPrimitiveCNode(cur_cnode, prim::kPrimAssign) && cur_cnode->HasAttr("expand_from") &&
             GetValue<std::string>(cur_cnode->GetAttr("expand_from")) == prim::kPrimAssignAdd->name();
    });
    if (target != sub_graph_cnode_list.end()) {
      assign_cnode = *target;
      assign_add_nodes->push_back(cnode);  // GraphKernel_Node
      MS_EXCEPTION_IF_NULL(assign_cnode);
      assign_add_nodes->push_back(assign_cnode->input(kIndex2));  // Add_node
    }
  }
  if (!assign_cnode) {
    return std::optional<std::string>();
  }
  return GetRefKeyFromNode(assign_cnode->input(kIndex1));
}

std::unordered_map<std::string, std::pair<CNodePtrList, AnfNodePtrList>> ExtractAssignAddByMirrorUser(
  const CNodePtrList &execute_order_cnode_list) {
  std::unordered_map<std::string, std::pair<CNodePtrList, AnfNodePtrList>> assign_add_map;
  for (const CNodePtr &cur_cnode : execute_order_cnode_list) {
    AnfNodePtrList assign_add_nodes;
    auto ref_key = ExtractAccuRefKeyFromAssignAddCNode(cur_cnode, &assign_add_nodes);
    if (!ref_key.has_value() || ref_key.value().find(kAccuGradsPrefix) != kIndex0) {
      continue;
    }
    if (assign_add_map.find(ref_key.value()) == assign_add_map.end()) {
      assign_add_map[ref_key.value()] = std::make_pair(CNodePtrList{cur_cnode}, assign_add_nodes);
    } else {
      assign_add_map[ref_key.value()].first.push_back(cur_cnode);
      assign_add_map[ref_key.value()].second = assign_add_nodes;
    }
  }
  return assign_add_map;
}

std::unordered_map<std::string, GradReduceUser> ExtractGradReduceByMirrorUser(
  const CNodePtrList &execute_order_cnode_list) {
  std::unordered_map<std::string, GradReduceUser> grad_reduce_map;
  for (const CNodePtr &cur_cnode : execute_order_cnode_list) {
    if (!IsFromParallelOptimizerRs(cur_cnode) && !IsFromGradMirrorAR(cur_cnode)) {
      continue;
    }
    auto mirror_user = GetMirrorUserIdFromCNode(cur_cnode);
    if (!mirror_user.has_value()) {
      continue;
    }
    if (grad_reduce_map.find(mirror_user.value()) == grad_reduce_map.end()) {
      auto grad_reduce_user = GradReduceUser();
      grad_reduce_user.grad_reduce_list.push_back(cur_cnode);
      grad_reduce_user.param_name = mirror_user.value();
      grad_reduce_map[mirror_user.value()] = grad_reduce_user;
    } else {
      grad_reduce_map[mirror_user.value()].grad_reduce_list.push_back(cur_cnode);
    }
  }
  return grad_reduce_map;
}

bool ExtractGradReduceUserList(const CNodePtrList &execute_order_cnode_list, bool with_accumulation,
                               std::vector<GradReduceUser> *grad_reduce_user_list) {
  auto grad_reduce_map = ExtractGradReduceByMirrorUser(execute_order_cnode_list);
  if (with_accumulation) {
    auto assign_add_map = ExtractAssignAddByMirrorUser(execute_order_cnode_list);
    for (auto grad_reduce_user_pair : grad_reduce_map) {
      auto mirror_user_id = grad_reduce_user_pair.first;
      auto expect_accu_grad_ref_key = kAccuGradsPrefix + mirror_user_id;
      if (assign_add_map.find(expect_accu_grad_ref_key) == assign_add_map.end()) {
        MS_LOG(WARNING) << "Cannot find accu_grad '" << expect_accu_grad_ref_key << "' in assign_add_map";
        return false;
      }
      grad_reduce_map[mirror_user_id].assign_add_list = assign_add_map[expect_accu_grad_ref_key].first;
      grad_reduce_map[mirror_user_id].latest_assign_add_nodes = assign_add_map[expect_accu_grad_ref_key].second;
    }
  }
  std::transform(grad_reduce_map.begin(), grad_reduce_map.end(), std::back_inserter(*grad_reduce_user_list),
                 [](const std::pair<std::string, GradReduceUser> &pair) { return pair.second; });
  for (auto &grad_reduce_user : *grad_reduce_user_list) {
    AnfNodePtrList latest_dw_compute_nodes;
    if (grad_reduce_user.assign_add_list.empty()) {
      latest_dw_compute_nodes = GetDwComputeNode(AnfNodePtrList{grad_reduce_user.grad_reduce_list.front()}, kIndex1);
    } else {
      latest_dw_compute_nodes = GetDwComputeNode(grad_reduce_user.latest_assign_add_nodes, kIndex2);
    }
    if (latest_dw_compute_nodes.empty()) {
      MS_LOG(WARNING) << "Cannot find corresponding dw calculation for " << grad_reduce_user.param_name << ", skip it.";
      return false;
    }
    auto latest_dw_execution_order =
      std::find(execute_order_cnode_list.begin(), execute_order_cnode_list.end(), latest_dw_compute_nodes.front()) -
      execute_order_cnode_list.begin();
    grad_reduce_user.latest_dw_compute_nodes = latest_dw_compute_nodes;
    grad_reduce_user.latest_dw_execute_order = LongToSize(latest_dw_execution_order);
  }
  std::sort(grad_reduce_user_list->begin(), grad_reduce_user_list->end(),
            [](const GradReduceUser &a, const GradReduceUser &b) {
              return a.latest_dw_execute_order < b.latest_dw_execute_order;
            });
  return true;
}

void ResetAssignAddGradCommDependWithAccumulation(const KernelGraphPtr &kernel_graph,
                                                  const GradReduceUser &grad_reduce_user) {
  auto manager = kernel_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto assign_add_list = grad_reduce_user.assign_add_list;
  auto grad_reduce_list = grad_reduce_user.grad_reduce_list;
  if (assign_add_list.size() > 1) {
    AnfNodePtrList inputs(assign_add_list.begin(), assign_add_list.end());
    auto make_tuple_cnode = common::AnfAlgo::CreateMakeTupleNode(kernel_graph, inputs);
    MS_EXCEPTION_IF_NULL(make_tuple_cnode);
    auto accu_input = grad_reduce_list.front()->input(kIndex1);
    while (IsPrimitiveCNode(accu_input, prim::kPrimDepend)) {
      auto accu_cnode = accu_input->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(accu_cnode);
      accu_input = accu_cnode->input(kIndex1);
    }
    auto depend_cnode = kernel_graph->NewCNode({NewValueNode(prim::kPrimDepend), accu_input, make_tuple_cnode});
    depend_cnode->set_abstract(accu_input->abstract());
    depend_cnode->AddAttr("grad_comm_assign_add_depend", MakeValue<bool>(true));
    manager->SetEdge(grad_reduce_list.front(), kIndex1, depend_cnode);
  } else {
    common::AnfAlgo::InsertDepend(assign_add_list.front(), grad_reduce_list.front(), manager, kernel_graph,
                                  "grad_comm_assign_add_depend");
  }
}
}  // namespace

bool OverlapGradReduce::DoOverlapGradReduce(const KernelGraphPtr &kernel_graph, bool with_accumulation) {
  kernel_graph->SetExecOrderByDefault();
  const auto &execution_order = kernel_graph->execution_order();
  std::vector<GradReduceUser> grad_reduce_user_list;
  if (!ExtractGradReduceUserList(execution_order, with_accumulation, &grad_reduce_user_list)) {
    MS_LOG(WARNING) << "Failed to extract grad_reduce_user list, skip it.";
    return false;
  }
  if (grad_reduce_user_list.empty()) {
    MS_LOG(DEBUG) << "grad_reduce_user_list is empty, no need to optimize, skip it.";
    return false;
  }
  auto manager = kernel_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto last_grad_reduce_node = grad_reduce_user_list.back().grad_reduce_list.back();
  for (const auto &grad_reduce_user : grad_reduce_user_list) {
    // Reconstruct depend for grad reduce
    auto next_op_users = manager->node_users()[grad_reduce_user.grad_reduce_list.back()];
    if (with_accumulation) {
      ResetAssignAddGradCommDependWithAccumulation(kernel_graph, grad_reduce_user);
    }
    // Move all communication users to the back of the last gradient communication.
    for (const auto &next_op_user : next_op_users) {
      if (IsPrimitiveCNode(next_op_user.first, prim::kPrimDepend) && next_op_user.second == kIndex2) {
        continue;
      }
      common::AnfAlgo::InsertDepend(last_grad_reduce_node, next_op_user.first, manager, kernel_graph,
                                    "last_grad_comm_compute_depend");
    }
  }
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto is_reorder_dw_dx_in_previous_pass =
    (!ms_context->get_param<std::string>(MS_CTX_GRAD_COMM_OVERLAP).empty() ||
     parallel::ParallelContext::GetInstance()->enable_fine_grained_micro_interleaved());
  auto is_enable_graph_kernel_fusion = graphkernel::GraphKernelFlags::GetInstance()
                                         .IsEnableGraphKernel();  // When graph kernel fusion is enabled, the gradient
                                                                  // and recompute nodes may be fusion together.
  if (is_reorder_dw_dx_in_previous_pass || is_enable_graph_kernel_fusion) {
    // Insert depend according grad_reduce in order
    auto pre_grad_reduce_user = grad_reduce_user_list.front();
    for (size_t i = 1; i < grad_reduce_user_list.size(); ++i) {
      auto pre_grad_reduce_cnode = pre_grad_reduce_user.grad_reduce_list.back();
      auto cur_grad_reduce_cnode = grad_reduce_user_list.at(i).grad_reduce_list.front();
      common::AnfAlgo::InsertDepend(pre_grad_reduce_cnode, cur_grad_reduce_cnode, manager, kernel_graph,
                                    "grad_comm_in_order_depend");
      pre_grad_reduce_user = grad_reduce_user_list.at(i);
    }
    // Insert depend between grad reduce and next op
    for (size_t i = 0; i < grad_reduce_user_list.size() - 1; ++i) {
      auto cur_grad_reduce_user = grad_reduce_user_list.at(i);
      auto next_grad_reduce_user = grad_reduce_user_list.at(i + 1);
      auto cur_grad_reduce_cnode = cur_grad_reduce_user.grad_reduce_list.back();

      auto cur_grad_compute_node = cur_grad_reduce_user.latest_dw_compute_nodes.front();
      auto next_grad_compute_node = next_grad_reduce_user.latest_dw_compute_nodes.front();
      common::AnfAlgo::InsertDepend(cur_grad_compute_node, next_grad_compute_node, manager, kernel_graph,
                                    "dw_in_order_depend");
      common::AnfAlgo::InsertDepend(cur_grad_reduce_cnode, next_grad_compute_node, manager, kernel_graph,
                                    "grad_comm_next_dw_depend");
    }
  } else {
    // Insert depend node to correspond dx
    for (size_t i = 0; i < grad_reduce_user_list.size(); ++i) {
      const auto &cur_grad_reduce_user = grad_reduce_user_list.at(i);
      const auto &cur_grad_reduce_cnode = cur_grad_reduce_user.grad_reduce_list.back();
      const auto &cur_grad_compute_nodes = cur_grad_reduce_user.latest_dw_compute_nodes;
      if (!IsOneOfPrimitiveCNode(cur_grad_compute_nodes.back(), {prim::kPrimMatMul, prim::kPrimBatchMatMul,
                                                                 prim::kPrimMatMulExt, prim::kPrimBatchMatMulExt})) {
        continue;
      }
      auto grad_compute_cnode = cur_grad_compute_nodes.back()->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(grad_compute_cnode);
      if (IsRecomputeNode(grad_compute_cnode->input(kIndex2))) {
        MS_LOG(DEBUG) << "For " << cur_grad_reduce_user.param_name
                      << ", its forward input is a recompute node, skip to insert grad_comm_dx_depend.";
        continue;
      }

      // find corresponding dx computation
      auto dx_compute_node = FindDxMatMulByDw(cur_grad_compute_nodes);
      if (dx_compute_node == nullptr) {
        MS_LOG(DEBUG) << "Cannot found corresponding dx computation for "
                      << cur_grad_compute_nodes.back()->DebugString() << ", skip to insert grad_comm_dx_depend";
        continue;
      }

      common::AnfAlgo::InsertDepend(cur_grad_reduce_cnode, dx_compute_node, manager, kernel_graph,
                                    "grad_comm_dx_depend");
    }
  }
  return true;
}

bool OverlapGradReduce::Run(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  if (kernel_graph == nullptr) {
    MS_LOG(DEBUG) << "Failed convert func_graph to kernel_graph, skip it.";
    return false;
  }
  MS_LOG(DEBUG) << "Status record: start overlap grad reduce optimization. graph id: " << kernel_graph->graph_id();
  auto parallel_context = parallel::ParallelContext::GetInstance();
  MS_EXCEPTION_IF_NULL(parallel_context);

  auto stages = parallel_context->pipeline_stage_split_num();
  auto grad_accu_step = parallel_context->grad_accumulation_step();
  auto with_accumulation = stages > 1 || grad_accu_step > 1;
  auto stage_device_num = parallel_context->device_num() / stages;
  auto stage_id = parallel_context->global_rank() / stage_device_num;
  bool ret;
  if (stage_id > 0) {
    MS_LOG(DEBUG) << "Under pipeline parallelism, stage " << stage_id
                  << " no need to reorder grad reduce communication, skip it.";
    ret = false;
  } else {
    ret = DoOverlapGradReduce(kernel_graph, with_accumulation);
  }
  MS_LOG(DEBUG) << "Status record: end overlap grad reduce optimization. graph id: " << kernel_graph->graph_id();
  return ret;
}
}  // namespace opt
}  // namespace mindspore
