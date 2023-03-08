/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/optimizer/enhancer/insert_tensor_move_for_hccl_op.h"
#include <vector>
#include <set>
#include <string>
#include "include/common/utils/utils.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "frontend/optimizer/opt.h"
#include "plugin/device/ascend/optimizer/ascend_helper.h"
#include "utils/trace_base.h"
namespace mindspore {
namespace opt {
namespace {
// insert tensormove for some cnode even if not a Ref cnode
const std::set<std::string> kNeedInsertTensorMoveOpSet = {kLambNextMVOpName, kLambNextMVWithDecayOpName,
                                                          kLambUpdateWithLROpName, kGetNextOpName};

// NodeUsersMap, for node B input i use node A, it will be one item in map with key: A, and value: (B, i)
bool IsNodeOutPutUsedByOtherRealKernel(const FuncGraphPtr &graph, const AnfNodePtr &input, size_t input_idx,
                                       const CNodePtr &cur_node) {
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto &node_users = manager->node_users();
  auto iter = node_users.find(input);
  if (iter == node_users.end()) {
    MS_LOG(EXCEPTION) << "node has no output in manager." << trace::DumpSourceLines(input);
  }
  auto user_items = iter->second;
  if (user_items.size() == 1) {
    MS_LOG(INFO) << "This node only used once, no need to insert tensormove node.";
    return false;
  }
  for (const auto &node_pair : user_items) {
    auto node = node_pair.first;
    auto idx = node_pair.second;
    MS_EXCEPTION_IF_NULL(node);
    if (AnfUtils::IsRealKernel(node) && (node != cur_node || idx != SizeToInt(input_idx))) {
      MS_LOG(INFO) << "This node only used other real kernel: " << node->fullname_with_scope();
      return true;
    }
  }
  MS_LOG(INFO) << "This node used by other node, but the node is not real kernel, no need to insert tensormove node.";
  return false;
}
}  // namespace

bool InsertTensorMoveForHcclOp::NeedInsertTensorMoveForSpecialCase(const AnfNodePtr &input,
                                                                   const CNodePtr &cur_node) const {
  MS_EXCEPTION_IF_NULL(input);
  MS_EXCEPTION_IF_NULL(cur_node);
  if (IsPrimitiveCNode(cur_node, prim::kPrimReceive)) {
    return false;
  }
  // visited skip nop node.
  auto kernel_with_index = common::AnfAlgo::VisitKernelWithReturnType(input, 0, true);
  auto real_input = kernel_with_index.first;
  // when input is a parameter or is a value node
  if (real_input->isa<Parameter>() || real_input->isa<ValueNode>()) {
    return true;
  }
  // when input is a Ref cnode
  if (kernel_query_->IsTbeRef(real_input)) {
    return true;
  }
  // when input is some special cnodes: kLambNextMVOpName, kLambNextMVWithDecayOpName, kLambUpdateWithLROpName,
  // kGetNextOpName
  if (kNeedInsertTensorMoveOpSet.find(common::AnfAlgo::GetCNodeName(real_input)) != kNeedInsertTensorMoveOpSet.end()) {
    return true;
  }
  return false;
}

bool InsertTensorMoveForHcclOp::NeedInsertTensorMove(const FuncGraphPtr &graph, const AnfNodePtr &input,
                                                     size_t input_idx, const CNodePtr &cur_node) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(input);
  MS_EXCEPTION_IF_NULL(cur_node);
  if (IsNodeOutPutUsedByOtherRealKernel(graph, input, input_idx, cur_node)) {
    return true;
  }
  if (common::AnfAlgo::IsNopNode(input) || (!AnfUtils::IsRealKernel(input))) {
    auto cnode = input->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    return NeedInsertTensorMove(graph, cnode->input(kIndex1), kIndex1, cnode);
  }
  return false;
}

void InsertTensorMoveForHcclOp::InsertTensorMove(const FuncGraphPtr &graph, const CNodePtr &hccl_node) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(hccl_node);
  bool need_tensor_move_async = false;
  std::vector<AnfNodePtr> new_inputs = {hccl_node->input(0)};
  for (size_t i = 1; i < hccl_node->size(); ++i) {
    auto input = hccl_node->input(i);
    if (NeedInsertTensorMoveForSpecialCase(input, hccl_node) || NeedInsertTensorMove(graph, input, i, hccl_node)) {
      auto tensor_move = CreateTensorMoveOp(graph, input);
      if (tensor_move == nullptr) {
        MS_LOG(EXCEPTION) << "Create tensor_move op failed.";
      }
      if (input->isa<CNode>() && common::AnfAlgo::IsDynamicShape(input)) {
        MS_LOG(DEBUG) << "The tenser move op has dynamic shape attr.";
      }
      new_inputs.push_back(tensor_move);
      need_tensor_move_async = true;
    } else {
      new_inputs.push_back(input);
    }
  }

  if (need_tensor_move_async) {
    CNodePtr new_hccl_node = std::make_shared<CNode>(*hccl_node);
    new_hccl_node->set_inputs(new_inputs);
    auto manager = graph->manager();
    MS_EXCEPTION_IF_NULL(manager);
    MS_LOG(DEBUG) << "start replace new_hccl_node to old hccl_node";
    auto kernel_graph = graph->cast<KernelGraphPtr>();
    if (kernel_graph != nullptr && kernel_graph->IsInternalOutput(hccl_node)) {
      kernel_graph->ReplaceInternalOutput(hccl_node, new_hccl_node);
    }
    (void)manager->Replace(hccl_node, new_hccl_node);
    MS_LOG(DEBUG) << "end replace";
  }
}

const AnfNodePtr InsertTensorMoveForHcclOp::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                    const EquivPtr &) const {
  if (func_graph == nullptr || node == nullptr || !node->isa<CNode>()) {
    return nullptr;
  }
  if (!common::AnfAlgo::IsCommunicationOp(node)) {
    return nullptr;
  }
  InsertTensorMove(func_graph, node->cast<CNodePtr>());
  return nullptr;
}
}  // namespace opt
}  // namespace mindspore
