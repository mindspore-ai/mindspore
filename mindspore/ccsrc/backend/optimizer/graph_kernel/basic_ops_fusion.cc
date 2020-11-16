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
#include "backend/optimizer/graph_kernel/basic_ops_fusion.h"

#include <memory>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <string>

#include "base/core_ops.h"
#include "ir/graph_utils.h"
#include "backend/optimizer/common/helper.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "vm/segment_runner.h"
#include "debug/anf_ir_dump.h"
#include "ir/func_graph_cloner.h"
#include "backend/optimizer/graph_kernel/composite_ops_fusion.h"
#include "backend/optimizer/graph_kernel/graph_kernel_helper.h"

namespace mindspore {
namespace opt {
namespace {
IncludeType IncludeFusedBasicOpForward(const AnfNodePtr &cur_node, const AnfNodePtr &node) {
  if (cur_node == node) {
    return FOLLOW;
  }
  if (!IsPrimitiveCNode(node)) {
    return EXCLUDE;
  }

  bool is_fusable = IsBasicFuseOp(node);
  return is_fusable ? FOLLOW : EXCLUDE;
}

std::vector<AnfNodePtr> FindFuseCNodes(const CNodePtr &cnode) {
  // Search fusable nodes according input direction.
  auto include_func_forward = std::bind(IncludeFusedBasicOpForward, cnode, std::placeholders::_1);
  auto used_nodes = DeepLinkedGraphSearch(cnode, include_func_forward);
  if (used_nodes.size() > 1) {
    used_nodes = RemoveCircle(used_nodes, false);
  }
  TopoSortForNodeList(&used_nodes);
  return used_nodes;
}

void SearchForDependNode(const AnfNodeSet &outputs_set, const AnfNodeIndexSet &users,
                         std::vector<CNodePtr> *control_depend_nodes, std::vector<size_t> *control_depend_use_index,
                         bool *is_only_control_depend_use, AnfNodePtr *use_out) {
  for (auto &user : users) {
    auto use_node = user.first;
    if (outputs_set.count(use_node) == 0 && !(IsPrimitiveCNode(use_node, prim::kPrimControlDepend))) {
      *is_only_control_depend_use = false;
      continue;
    }
    if (outputs_set.count(use_node) != 0) {
      *use_out = use_node;
    }
    if (IsPrimitiveCNode(use_node, prim::kPrimControlDepend)) {
      control_depend_nodes->push_back(use_node->cast<CNodePtr>());
      control_depend_use_index->push_back(user.second);
    }
  }
}

bool FindControlDependOut(AnfNodePtrList *outputs, const AnfNodePtrList &vir_outputs, const FuncGraphManagerPtr &mng,
                          std::unordered_map<AnfNodePtr, AnfNodePtr> *eqv) {
  AnfNodeSet outputs_set;
  for (auto out : *outputs) {
    outputs_set.insert(out);
  }
  bool has_erase_outs = false;
  int index = -1;
  for (auto it = outputs->begin(); it != outputs->end();) {
    index++;
    auto out = *it;
    (*eqv)[out] = vir_outputs[IntToSize(index)];
    auto users = mng->node_users()[out];
    bool is_only_control_depend_use = true;
    std::vector<size_t> control_depend_use_index;
    std::vector<CNodePtr> control_depend_nodes;
    AnfNodePtr use_out = nullptr;
    SearchForDependNode(outputs_set, users, &control_depend_nodes, &control_depend_use_index,
                        &is_only_control_depend_use, &use_out);
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
  return has_erase_outs;
}

void RemoveControlDependOut(const FuncGraphPtr &fg, AnfNodePtrList *outputs, const FuncGraphManagerPtr &mng) {
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

  if (!FindControlDependOut(outputs, vir_outputs, mng, &eqv)) {
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

bool FuseBasicOps(const FuncGraphPtr &kernel_graph, const std::vector<AnfNodePtr> &todos,
                  std::unordered_set<AnfNodePtr> *fused_ops) {
  bool changed = false;
  auto mng = kernel_graph->manager();
  for (auto iter = todos.cbegin(); iter != todos.cend(); ++iter) {
    auto node = (*iter)->cast<CNodePtr>();
    if (node == nullptr) {
      continue;
    }
    if (fused_ops->count(node)) {
      continue;
    }
    bool is_basic_op = IsBasicFuseOp(node);
    if (!is_basic_op || !kernel_graph->nodes().contains(node)) {
      continue;
    }

    auto fuse_nodes = FindFuseCNodes(node);
    if (fuse_nodes.size() <= 1) {
      continue;
    }

    changed = true;
    FuncGraphPtr fg;
    AnfNodePtrList inputs;
    AnfNodePtrList outputs;
    std::tie(fg, inputs, outputs) = compile::TransformSegmentToAnfGraph(fuse_nodes);
    RemoveControlDependOut(fg, &outputs, mng);
    ConvertNonscalarTensorToParameter(fg, &inputs);
    auto fuse_new_node = CreateNewFuseCNode(kernel_graph, fg, inputs, outputs);
    SetNewKernelInfo(fuse_new_node, fg, inputs, outputs, AnfAlgo::GetProcessor(fuse_nodes[0]));

    ReplaceNewFuseCNode(kernel_graph, fuse_new_node, outputs);

    // Set graph kernel attr
    std::string fuse_op_name = "";
    for (auto &fuse_node : fuse_nodes) {
      fuse_op_name += AnfAlgo::GetCNodePrimitive(fuse_node)->name() + "_";
    }
    fused_ops->insert(fuse_nodes.begin(), fuse_nodes.end());
    fg->set_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL, MakeValue(fuse_op_name));
  }
  std::dynamic_pointer_cast<session::KernelGraph>(kernel_graph)->SetExecOrderByDefault();
  return changed;
}
}  // namespace

bool FuseBasicOps(const FuncGraphPtr &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto mng = kernel_graph->manager();
  if (mng == nullptr) {
    mng = Manage(kernel_graph, true);
    kernel_graph->set_manager(mng);
  }
  std::unordered_set<AnfNodePtr> fused_ops;
  auto todos = TopoSort(kernel_graph->get_return());
  std::reverse(todos.begin(), todos.end());
  return FuseBasicOps(kernel_graph, todos, &fused_ops);
}

bool BasicOpsFusion::Run(const FuncGraphPtr &func_graph) { return FuseBasicOps(func_graph); }
}  // namespace opt
}  // namespace mindspore
