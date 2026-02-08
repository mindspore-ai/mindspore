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

#include "backend/ms_backend/graph_fusion/deal_with_side_effect.h"

#include <utility>
#include "ir/anf.h"
#include "ir/graph_utils.h"
#include "primitive/nn_optimizer_ops.h"
#include "include/runtime/hardware_abstract/kernel_base/graph_fusion/framework_utils.h"
#include "include/utils/anfalgo.h"
#include "include/backend/common/kernel_graph/kernel_graph.h"
#include "include/runtime/hardware_abstract/kernel_base/kernel_info.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"

namespace mindspore::graphkernel {
namespace {
void AddRefPairForNode(const AnfNodePtr &node, size_t output_idx, size_t input_idx) {
  MS_EXCEPTION_IF_NULL(node);
  if (node->kernel_info() == nullptr) {
    MS_LOG(DEBUG) << "node " << node->DebugString() << " " << node->fullname_with_scope() << " has no kernel_info";
    return;
  }
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  MS_LOG(INFO) << "AddRefMap(" << output_idx << "," << input_idx << ") for node " << node->DebugString() << " "
               << node->fullname_with_scope();
  kernel_info->AddRefMap(output_idx, input_idx);
}
}  // namespace

void DealWithSideEffect::MarkSideEffect(const FuncGraphPtr &sub_graph) {
  MS_EXCEPTION_IF_NULL(sub_graph);
  bool has_side_effect_mem = false;
  auto nodes = TopoSort(sub_graph->get_return());
  for (auto node : nodes) {
    if (node == nullptr) {
      continue;
    }
    if (IsPrimitiveCNode(node, prim::kPrimAssign)) {
      has_side_effect_mem = true;
      break;
    }
  }
  if (has_side_effect_mem) {
    sub_graph->set_attr(GRAPH_FLAG_SIDE_EFFECT_MEM, MakeValue(true));
  }
}

bool DealWithSideEffect::Run(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  bool changed = false;

  auto todos = TopoSort(func_graph->get_return());
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  for (auto node : todos) {
    if (!common::AnfAlgo::IsGraphKernel(node)) {
      continue;
    }
    auto sub_graph = common::AnfAlgo::GetCNodeFuncGraphPtr(node);
    MS_EXCEPTION_IF_NULL(sub_graph);
    // add side effect attr
    MarkSideEffect(sub_graph);
    AnfNodePtrList output_list;
    kernel::GetFuncGraphOutputNodes(sub_graph, &output_list);
    auto graph_inputs = sub_graph->parameters();
    for (size_t i = 0; i < output_list.size(); ++i) {
      auto &out = output_list[i];
      auto out_cnode = out->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(out_cnode);
      auto out_cnode_prim = GetCNodePrimitive(out_cnode);
      if (out_cnode_prim != nullptr && out_cnode_prim->name() == prim::kPrimAssign->name()) {
        auto iter = std::find(graph_inputs.begin(), graph_inputs.end(), out_cnode->input(kIndex1));
        if (iter == graph_inputs.end()) {
          MS_LOG_WITH_NODE(EXCEPTION, out_cnode) << out_cnode->fullname_with_scope() << " first input isn't parameter.";
        }
        auto input_idx = std::distance(graph_inputs.begin(), iter);
        auto origin_pair = common::AnfAlgo::GetPrevNodeOutput(node, input_idx, true);
        // record the ref_pair
        session::AnfWithOutIndex final_pair = std::make_pair(node, i);
        if (kernel_graph->IsInRefOutputMap(final_pair)) {
          MS_LOG(INTERNAL_EXCEPTION) << "Ref_pair is already in ref map, node is " << node->fullname_with_scope()
                                     << ", index is " << i;
        }
        MS_LOG(DEBUG) << "Add Ref pair, final {node ptr " << final_pair.first.get() << " , info is "
                      << final_pair.first->fullname_with_scope() << " , index is " << final_pair.second
                      << "}, origin {node ptr " << origin_pair.first.get() << ", info is "
                      << origin_pair.first->fullname_with_scope() << " : index " << origin_pair.second << "}";
        kernel_graph->AddRefCorrespondPairs(final_pair, origin_pair);
        AddRefPairForNode(node, i, static_cast<size_t>(input_idx));
        changed = true;
      }
    }
  }

  return changed;
}
}  // namespace mindspore::graphkernel
