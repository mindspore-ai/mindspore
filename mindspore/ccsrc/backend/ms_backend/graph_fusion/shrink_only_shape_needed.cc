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
#include "backend/ms_backend/graph_fusion/shrink_only_shape_needed.h"
#include <map>
#include <vector>
#include <memory>
#include "primitive/array_ops.h"
#include "primitive/sequence_ops.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "include/utils/anfalgo.h"
#include "ir/graph_utils.h"
#include "backend/ms_backend/graph_fusion/core/graph_kernel_utils.h"
#include "backend/ms_backend/graph_fusion/core/eliminate_redundant_output.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"
#include "primitive/auto_generate/gen_ops_primitive_s.h"
#include "primitive/auto_generate/gen_ops_primitive_t.h"

namespace mindspore::graphkernel {
namespace {
bool UsedByShapeOp(const AnfNodePtr &user, size_t user_input_idx = 0) {
  MS_EXCEPTION_IF_NULL(user);
  if (IsPrimitiveCNode(user, prim::kPrimShape)) {
    return true;
  }
  if (IsPrimitiveCNode(user, prim::kPrimShapeCalc)) {
    const auto &attr = common::AnfAlgo::GetCNodePrimitiveAttr(user, kAttrOnlyDependShape);
    if (attr != nullptr) {
      auto only_depend_shape = GetValue<std::vector<bool>>(attr);
      return user_input_idx < only_depend_shape.size() && only_depend_shape[user_input_idx];
    }
  }
  return false;
}

bool OnlyUsedByShapeOp(const AnfNodePtr &node, const FuncGraphManagerPtr &mng) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(mng);
  // check if all users of node are shape op
  auto &users = mng->node_users();
  auto iter = users.find(node);
  if (iter == users.end()) {
    return false;
  }
  for (auto &[user_node, user_input_idx] : iter->second) {
    if (!UsedByShapeOp(user_node, IntToSize(user_input_idx - 1))) {
      return false;
    }
  }
  return true;
}

bool ShapeEqual(const ListSymbolPtr &shape1, const ListSymbolPtr &shape2) {
  if (shape1 == nullptr || shape2 == nullptr) {
    return false;
  }
  return shape1->EqualsTo(shape2);
}

AnfNodePtr FindSameShapeInput(const AnfNodePtr &node, const AnfNodePtr &output) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(output);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto out_shape = GkUtils::GetOutputSymbolicShape(output, 0);
  if (out_shape == nullptr) {
    return nullptr;
  }
  for (size_t i = 1; i < cnode->inputs().size(); ++i) {
    auto input_node = cnode->input(i);
    auto input_shape = GkUtils::GetOutputSymbolicShape(input_node, 0);
    if (ShapeEqual(out_shape, input_shape)) {
      return input_node;
    }
  }
  return nullptr;
}

void FindOutputReplaceMap(const AnfNodePtr &node, const FuncGraphManagerPtr &mng,
                          std::map<AnfNodePtr, AnfNodePtr> *replace_map) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(mng);
  MS_EXCEPTION_IF_NULL(replace_map);
  auto func_graph = GetCNodeFuncGraph(node);
  MS_EXCEPTION_IF_NULL(func_graph);
  auto output = func_graph->output();
  if (IsPrimitiveCNode(output, prim::kPrimMakeTuple)) {
    // multiple outputs, check user of get item node
    auto &users = mng->node_users();
    auto iter = users.find(node);
    if (iter == users.end()) {
      return;
    }
    for (auto &item : iter->second) {
      auto get_item = item.first;
      MS_EXCEPTION_IF_NULL(get_item);
      if (IsPrimitiveCNode(get_item, prim::kPrimTupleGetItem) && OnlyUsedByShapeOp(get_item, mng)) {
        auto tuple_idx = common::AnfAlgo::GetTupleGetItemOutIndex(get_item->cast<CNodePtr>());
        auto make_tuple = output->cast<CNodePtr>();
        MS_EXCEPTION_IF_NULL(make_tuple);
        (*replace_map)[get_item] = FindSameShapeInput(node, make_tuple->input(tuple_idx + 1));
      }
    }
  } else if (OnlyUsedByShapeOp(node, mng)) {
    // single output, check user of node
    (*replace_map)[node] = FindSameShapeInput(node, output);
  }
}
}  // namespace

bool ShrinkOnlyShapeNeeded::Process(const AnfNodePtr &node, const FuncGraphManagerPtr &mng) const {
  std::map<AnfNodePtr, AnfNodePtr> replace_map;
  FindOutputReplaceMap(node, mng, &replace_map);
  bool changed = false;
  for (const auto &[src, dst] : replace_map) {
    if (dst != nullptr) {
      MS_LOG(DEBUG) << "replace node [" << src->fullname_with_scope() << "] with node [" << dst->fullname_with_scope()
                    << "]";
      (void)mng->Replace(src, dst);
      changed = true;
    }
  }
  return changed;
}

bool ShrinkOnlyShapeNeeded::Run(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  if (!func_graph->dynamic_shape()) {
    return false;
  }
  auto mng = func_graph->manager();
  MS_EXCEPTION_IF_NULL(mng);
  bool changed = false;
  auto nodes = TopoSort(func_graph->get_return());
  for (auto iter = nodes.crbegin(); iter != nodes.crend(); ++iter) {
    auto node = *iter;
    if (node == nullptr || !AnfUtils::IsGraphKernel(node)) {
      continue;
    }
    MS_LOG(INFO) << "start processing node: " << node->fullname_with_scope();
    changed = Process(node, mng) || changed;
    MS_LOG(INFO) << "end processing node: " << node->fullname_with_scope();
  }
  if (changed) {
    GkUtils::UpdateFuncGraphManager(mng, func_graph);
    (void)std::make_shared<EliminateHangingOutput>()->Run(func_graph);
  }
  return changed;
}
}  // namespace mindspore::graphkernel
