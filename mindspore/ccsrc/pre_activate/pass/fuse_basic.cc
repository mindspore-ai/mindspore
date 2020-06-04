
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
#include "pre_activate/pass/fuse_basic.h"
#include "pre_activate/pass/fuse_composite.h"

#include <memory>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <string>

#include "operator/ops.h"
#include "utils/utils.h"
#include "utils/graph_utils.h"
#include "pre_activate/common/helper.h"
#include "session/anf_runtime_algorithm.h"
#include "vm/segment_runner.h"
#include "debug/draw.h"
#include "debug/anf_ir_dump.h"
#include "ir/func_graph_cloner.h"

namespace mindspore {
namespace opt {
namespace {
std::vector<PrimitivePtr> get_fusable_basic_ops(bool is_before_kernel_select) {
  std::vector<PrimitivePtr> fusable_basic_ops = {prim::kPrimTensorAdd, prim::kPrimMul, prim::kPrimSub,
                                                 prim::kPrimExpandDims};
  if (!is_before_kernel_select) {
    fusable_basic_ops.push_back(prim::kPrimCast);
  }
  return fusable_basic_ops;
}

IncludeType IncludeFusedBasicOpForward(const AnfNodePtr &cur_node, const CompositeInfo &info, const AnfNodePtr &node) {
  if (cur_node == node) {
    return FOLLOW;
  }
  if (!IsPrimitiveCNode(node)) {
    return EXCLUDE;
  }

  auto fusable_basic_ops = get_fusable_basic_ops(info.is_before_kernel_select);
  bool is_fusable = std::any_of(fusable_basic_ops.begin(), fusable_basic_ops.end(),
                                [&node](const PrimitivePtr &prim) { return IsPrimitiveCNode(node, prim); });

  return is_fusable ? FOLLOW : EXCLUDE;
}

std::vector<AnfNodePtr> FindFuseCNodes(const CNodePtr &cnode, bool is_before_kernel_select) {
  CompositeInfo info;
  info.is_before_kernel_select = is_before_kernel_select;
  // Search fusable nodes according input direction.
  auto include_func_forward = std::bind(IncludeFusedBasicOpForward, cnode, info, std::placeholders::_1);
  auto used_nodes = DeepLinkedGraphSearch(cnode, include_func_forward);
  if (used_nodes.size() > 1) {
    used_nodes = RemoveCircle(used_nodes, false);
  }
  TopoSortForNodeList(&used_nodes);
  return used_nodes;
}

void RemoveControlDependOut(const FuncGraphPtr &fg, AnfNodePtrList *outputs, const FuncGraphManagerPtr &mng) {
  AnfNodeSet outputs_set;
  for (auto out : *outputs) {
    outputs_set.insert(out);
  }

  AnfNodePtrList vir_outputs;
  std::unordered_map<AnfNodePtr, AnfNodePtr> eqv;
  auto fg_outputs = fg->output();
  if (IsPrimitiveCNode(fg_outputs, prim::kPrimMakeTuple)) {
    auto cnode = fg_outputs->cast<CNodePtr>();
    for (size_t i = 1; i < cnode->size(); ++i) {
      vir_outputs.push_back(cnode->input(i));
    }
  } else {
    vir_outputs.push_back(fg_outputs);
  }

  if (vir_outputs.size() != outputs->size()) {
    MS_LOG(EXCEPTION) << "The size of virtual output of the fg is not the same with the real output";
  }
  bool has_erase_outs = false;
  size_t index = -1;
  for (auto it = outputs->begin(); it != outputs->end();) {
    index++;
    auto out = *it;
    eqv[out] = vir_outputs[index];
    auto users = mng->node_users()[out];
    bool is_only_control_depend_use = true;
    std::vector<size_t> control_depend_use_index;
    std::vector<CNodePtr> control_depend_nodes;
    AnfNodePtr use_out = nullptr;
    for (auto &user : users) {
      auto use_node = user.first;
      if (outputs_set.count(use_node) == 0 && !(IsPrimitiveCNode(use_node, prim::kPrimControlDepend))) {
        is_only_control_depend_use = false;
        continue;
      }
      if (outputs_set.count(use_node) != 0) {
        use_out = use_node;
      }

      if (IsPrimitiveCNode(use_node, prim::kPrimControlDepend)) {
        control_depend_nodes.push_back(use_node->cast<CNodePtr>());
        control_depend_use_index.push_back(user.second);
      }
    }

    if (is_only_control_depend_use && !control_depend_nodes.empty()) {
      MS_EXCEPTION_IF_NULL(use_out);
      it = outputs->erase(it);
      for (size_t i = 0; i < control_depend_nodes.size(); ++i) {
        auto control_depend_node = control_depend_nodes[i];
        std::vector<AnfNodePtr> new_control_depend_inputs;
        for (size_t j = 0; j < control_depend_node->size(); ++j) {
          if (j == control_depend_use_index[i]) {
            new_control_depend_inputs.push_back(use_out);
          } else {
            new_control_depend_inputs.push_back(control_depend_node->input(j));
          }
        }
        auto new_control_depend = control_depend_node->func_graph()->NewCNode(new_control_depend_inputs);
        mng->Replace(control_depend_node, new_control_depend);
        has_erase_outs = true;
      }
    } else {
      it++;
    }
  }

  if (!has_erase_outs) {
    return;
  }

  AnfNodePtr fg_new_output;
  if (outputs->size() > 1) {
    std::vector<AnfNodePtr> output_args;
    output_args.push_back(NewValueNode(prim::kPrimMakeTuple));
    (void)std::transform(std::begin(*outputs), std::end(*outputs), std::back_inserter(output_args),
                         [&eqv](const AnfNodePtr &o) -> AnfNodePtr { return eqv[o]; });
    // Set output for AnfGraph
    fg_new_output = fg->NewCNode(output_args);
  } else {
    fg_new_output = eqv[(*outputs)[0]];
  }
  fg->set_output(fg_new_output, true);
}

void FuseBasic(const std::shared_ptr<session::KernelGraph> &kernel_graph, const std::vector<AnfNodePtr> &todos,
               std::unordered_set<AnfNodePtr> *fused_ops, bool is_before_kernel_select) {
  auto mng = kernel_graph->manager();
  for (auto iter = todos.cbegin(); iter != todos.cend(); ++iter) {
    auto node = (*iter)->cast<CNodePtr>();
    if (node == nullptr) {
      continue;
    }
    if (fused_ops->count(node)) {
      continue;
    }
    auto fusable_basic_ops = get_fusable_basic_ops(is_before_kernel_select);
    bool is_basic_op = std::any_of(fusable_basic_ops.begin(), fusable_basic_ops.end(),
                                   [&node](const PrimitivePtr &prim) { return IsPrimitiveCNode(node, prim); });
    if (!is_basic_op || !kernel_graph->nodes().contains(node)) {
      continue;
    }

    auto fuse_nodes = FindFuseCNodes(node, is_before_kernel_select);
    if (fuse_nodes.size() <= 1) {
      continue;
    }

    FuncGraphPtr fg;
    AnfNodePtrList inputs;
    AnfNodePtrList outputs;
    std::tie(fg, inputs, outputs) = compile::TransformSegmentToAnfGraph(fuse_nodes);
    RemoveControlDependOut(fg, &outputs, mng);
    auto fuse_new_node = CreateNewFuseCNode(kernel_graph, fg, inputs, outputs, is_before_kernel_select);

    ReplaceNewFuseCNode(kernel_graph, fuse_new_node, outputs);

    // Set composite flag
    std::string fuse_op_name = "";
    for (auto &fuse_node : fuse_nodes) {
      fuse_op_name += AnfAlgo::GetCNodePrimitive(fuse_node)->name() + "_";
    }
    fused_ops->insert(fuse_nodes.begin(), fuse_nodes.end());
    fg->set_attr(FUNC_GRAPH_FLAG_COMPOSITE, MakeValue(fuse_op_name));
  }
}
}  // namespace

void FuseBasic(const std::shared_ptr<session::KernelGraph> &kernel_graph, bool is_before_kernel_select) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto mng = kernel_graph->manager();
  if (mng == nullptr) {
    mng = Manage(kernel_graph, true);
    kernel_graph->set_manager(mng);
  }
  std::unordered_set<AnfNodePtr> fused_ops;
  auto todos = TopoSort(kernel_graph->get_return());
  std::reverse(todos.begin(), todos.end());
  FuseBasic(kernel_graph, todos, &fused_ops, is_before_kernel_select);
}
}  // namespace opt
}  // namespace mindspore
