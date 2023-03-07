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
#include "plugin/device/ascend/optimizer/enhancer/insert_tensor_move_for_cascade.h"
#include <vector>
#include "include/common/utils/utils.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "frontend/optimizer/opt.h"
#include "plugin/device/ascend/optimizer/ascend_helper.h"
#include "utils/trace_base.h"

namespace mindspore {
namespace opt {
namespace {
constexpr auto kSingleOutput = 1;

bool IsPartOutputsOfHcclOp(const AnfNodePtr &node, const CNodePtr &cur_hccl, const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(cur_hccl);
  MS_EXCEPTION_IF_NULL(graph);
  if (!common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimTupleGetItem)) {
    return false;
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto prev_node = cnode->input(kRealInputNodeIndexInTupleGetItem);
  MS_EXCEPTION_IF_NULL(prev_node);
  if (!common::AnfAlgo::IsCommunicationOp(prev_node)) {
    return false;
  }
  auto prev_hccl_op = prev_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(prev_hccl_op);

  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto &node_users = manager->node_users();
  auto iter = node_users.find(prev_hccl_op);
  if (iter == node_users.end()) {
    MS_LOG(EXCEPTION) << "Node has no output in manager" << trace::DumpSourceLines(cur_hccl);
  }
  for (const auto &node_index : iter->second) {
    AnfNodePtr output = node_index.first;
    MS_EXCEPTION_IF_NULL(output);
    if (IsPrimitiveCNode(output, prim::kPrimTupleGetItem)) {
      bool is_contain = false;
      for (size_t i = 1; i < cur_hccl->size(); ++i) {
        if (cur_hccl->input(i) == output) {
          is_contain = true;
          break;
        }
      }
      if (!is_contain) {
        return true;
      }
    }
  }
  return false;
}
}  // namespace

AnfNodePtr InsertTensorMoveForCascade::InsertTensorMove(const FuncGraphPtr &graph, const CNodePtr &hccl_node) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(hccl_node);
  std::vector<AnfNodePtr> tensor_move_list;
  std::vector<AnfNodePtr> new_inputs = {hccl_node->input(0)};
  for (size_t i = 1; i < hccl_node->size(); ++i) {
    auto input = hccl_node->input(i);
    MS_EXCEPTION_IF_NULL(input);
    // when input is also a hccl op and just part outputs of it linking with cur_hccl_op
    if (IsPartOutputsOfHcclOp(input, hccl_node, graph)) {
      auto tensor_move = CreateTensorMoveOp(graph, input);
      if (tensor_move == nullptr) {
        MS_LOG(EXCEPTION) << "Create tensor_move op failed." << trace::DumpSourceLines(hccl_node);
      }
      if (common::AnfAlgo::IsDynamicShape(input)) {
        MS_LOG(DEBUG) << "The tenser move op has dynamic shape attr.";
      }
      auto kernel_info = std::make_shared<device::KernelInfo>();
      tensor_move->set_kernel_info(kernel_info);
      MS_EXCEPTION_IF_NULL(kernel_select_);
      kernel_select_->SelectKernel(tensor_move->cast<CNodePtr>());
      new_inputs.push_back(tensor_move);
      tensor_move_list.push_back(tensor_move);
    } else {
      new_inputs.push_back(input);
    }
  }

  if (!tensor_move_list.empty()) {
    CNodePtr new_hccl_node = std::make_shared<CNode>(*hccl_node);
    MS_EXCEPTION_IF_NULL(new_hccl_node);
    new_hccl_node->set_inputs(new_inputs);
    return new_hccl_node;
  }
  return nullptr;
}

void InsertTensorMoveForCascade::InsertOutputTensorMove(const FuncGraphPtr &graph, const CNodePtr &hccl_node) const {
  // Need to insert TensorMove if the output of FusedCommunicationOp is GraphOutput
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(hccl_node);
  MS_EXCEPTION_IF_NULL(kernel_select_);

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto task_sink = ms_context->get_param<bool>(MS_CTX_ENABLE_TASK_SINK);
  auto opt_level = ms_context->get_param<int>(MS_CTX_MEMORY_OPTIMIZE_LEVEL);
  if (!task_sink && (opt_level == kOptimizeO0)) {
    // not use somas
    return;
  }

  AnfNodePtr node = nullptr;
  auto outputs = common::AnfAlgo::GetAllOutputWithIndex(graph->output());
  for (const auto &output : outputs) {
    const auto &output_with_index = common::AnfAlgo::FetchRealNodeSkipMonadControl(output);
    if (!common::AnfAlgo::IsCommunicationOp(output_with_index.first)) {
      continue;
    }
    auto cnode = output_with_index.first->cast<CNodePtr>();
    if (cnode != hccl_node || AnfAlgo::GetOutputTensorNum(output_with_index.first) == kSingleOutput) {
      continue;
    }
    node = output_with_index.first;
    break;
  }

  if (node != nullptr) {
    auto tensor_move_list = InsertTensorMoveForGraphOutput(graph, node);
    for (auto &tensor_move : tensor_move_list) {
      kernel_select_->SelectKernel(tensor_move->cast<CNodePtr>());
    }
  }
}

const AnfNodePtr InsertTensorMoveForCascade::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                     const EquivPtr &) const {
  if (func_graph == nullptr || node == nullptr || !node->isa<CNode>()) {
    return nullptr;
  }
  auto cnode = node->cast<CNodePtr>();
  if (!common::AnfAlgo::IsCommunicationOp(node)) {
    return nullptr;
  }
  InsertOutputTensorMove(func_graph, cnode);
  return InsertTensorMove(func_graph, cnode);
}
}  // namespace opt
}  // namespace mindspore
