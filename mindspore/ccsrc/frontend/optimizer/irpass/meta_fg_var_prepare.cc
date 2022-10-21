/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "frontend/optimizer/irpass/meta_fg_var_prepare.h"
#include <vector>
#include <algorithm>
#include <memory>

#include "utils/hash_map.h"
#include "frontend/operator/composite/composite.h"
#include "frontend/operator/ops.h"
#include "frontend/optimizer/irpass.h"
#include "frontend/optimizer/anf_visitor.h"
#include "ir/func_graph.h"

namespace mindspore {
namespace opt {
namespace irpass {
// Get meta_fg_ops registration white list
static const std::vector<MatcherPtr> &GetMetaFgOps() {
  static const std::vector<MatcherPtr> meta_fg_ops{
    std::make_shared<MetaFgMatcher<prim::GradOperation>>(),
    std::make_shared<MetaFgMatcher<prim::VmapOperation>>(),
    std::make_shared<MetaFgMatcher<prim::Shard>>(),
  };
  return meta_fg_ops;
}

static AnfNodePtr GenerateUnpackGraphNode(const AnfNodePtr &origin_node, const std::vector<AnfNodePtr> &inputs_y,
                                          const AnfNodePtr &func_node, bool is_unpack, bool sens_param) {
  MS_EXCEPTION_IF_NULL(func_node);
  FuncGraphPtr func_graph = origin_node->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  std::vector<AnfNodePtr> nodes;
  AnfNodePtr unpack_graph_node = nullptr;
  std::shared_ptr<prim::UnpackGraphPrimitive> unpack_graph;
  int64_t inputs_begin_index;
  constexpr int64_t unpack_inputs_begin_index = 2;
  if (is_unpack) {
    unpack_graph = std::make_shared<prim::UnpackGraphPrimitive>(sens_param, true);
    // {unpackcall, {GradOperation, ...}, args...} and other {unpackcall, {meta_fg_opration, ...}, args...}
    inputs_begin_index = unpack_inputs_begin_index;
  } else {
    unpack_graph = std::make_shared<prim::UnpackGraphPrimitive>(sens_param, false);
    // {{GradOperation, ...}, args...} and other {{meta_fg_opration, ...}, args...}
    inputs_begin_index = 1;
  }
  (void)nodes.emplace_back(NewValueNode(unpack_graph));
  (void)nodes.emplace_back(func_node);
  (void)std::transform(inputs_y.cbegin() + inputs_begin_index, inputs_y.cend(), std::back_inserter(nodes),
                       [](const AnfNodePtr &node) { return node; });
  unpack_graph_node = func_graph->NewCNodeBefore(origin_node, nodes);
  return unpack_graph_node;
}

// Check if node is a specific meta_fg_operation that registered in the meta_fg_ops
bool CheckMetaFgOps(const AnfNodePtr &node) {
  if (node == nullptr) {
    return false;
  }
  auto value = GetValueWithoutDoSignature(node);
  if (value == nullptr) {
    return false;
  }
  auto meta_func_graph = value->cast<MetaFuncGraphPtr>();
  if (meta_func_graph == nullptr) {
    return false;
  }

  const auto &meta_fg_ops = GetMetaFgOps();
  for (auto meta_fg_op : meta_fg_ops) {
    if (meta_fg_op->Match(meta_func_graph)) {
      return true;
    }
  }
  return false;
}

// {{GradOperation, g, w}, Ys}, {UnpackCall, {GradOperation, g, w}, Ys},
// and other {{meta_fg_opration, ...}, ...} or {UnpackCall, {meta_fg_opration, ...}, ...}
AnfNodePtr MetaFgVarPrepare::operator()(const OptimizerPtr &, const AnfNodePtr &node) {
  auto cnode = node->cast<CNodePtr>();
  if (cnode == nullptr) {
    return nullptr;
  }
  auto func_graph = cnode->func_graph();
  if (func_graph == nullptr) {
    return nullptr;
  }

  // {{...}, Ys}
  auto inputs_y = cnode->inputs();
  std::vector<AnfNodePtr> inputs_x;
  if (IsCNode(inputs_y[0])) {
    inputs_x = inputs_y[0]->cast<CNodePtr>()->inputs();
  } else if (unpack_op_->Match(inputs_y[0]) && IsCNode(inputs_y[1])) {
    inputs_x = inputs_y[1]->cast<CNodePtr>()->inputs();
  } else {
    return nullptr;
  }

  // {{...}, Xs}
  const size_t inputs_x_minimum_size = 2;
  if (inputs_x.size() < inputs_x_minimum_size) {
    return nullptr;
  }

  if (!CheckMetaFgOps(inputs_x[0])) {
    return nullptr;
  }

  auto func_node = inputs_x[1];
  if (!IsValueNode<FuncGraph>(func_node)) {
    return nullptr;
  }

  const bool is_unpack = unpack_op_->Match(inputs_y[0]);

  // For general meta_fg_opration, ‘sens_param’ is not involved, and that of GradOperation obtained specifically.
  bool sens_param = false;
  if (grad_op_->Match(inputs_x[0])) {
    auto value = GetValueWithoutDoSignature(inputs_x[0]);
    MS_EXCEPTION_IF_NULL(value);
    auto meta_func = value->cast<MetaFuncGraphPtr>();
    MS_EXCEPTION_IF_NULL(meta_func);
    auto grad_op_ptr = meta_func->cast<prim::GradOperationPtr>();
    MS_EXCEPTION_IF_NULL(grad_op_ptr);
    sens_param = grad_op_ptr->sens_param();

    // Remove the tuple/list inputs from order list for grad(UnpackGraph(..), list/tuple)(..)
    if (inputs_x.size() > inputs_x_minimum_size) {
      constexpr size_t sequence_input_pos = 2;
      auto seq_node = inputs_x[sequence_input_pos];
      auto prim = GetCNodePrimitiveWithoutDoSignature(seq_node);
      if (prim != nullptr &&
          (IsPrimitiveEquals(prim, prim::kPrimMakeTuple) || IsPrimitiveEquals(prim, prim::kPrimMakeList))) {
        auto seq_cnode = dyn_cast<CNode>(seq_node);
        MS_EXCEPTION_IF_NULL(seq_cnode);
        seq_cnode->func_graph()->EraseUnusedNodeInOrder(seq_cnode);
      }
    }
  }

  inputs_x[1] = GenerateUnpackGraphNode(node, inputs_y, func_node, is_unpack, sens_param);
  // Construct new meta_fg_opration
  auto meta_fg_op_cnode = func_graph->NewCNodeBefore(node, inputs_x);
  if (unpack_op_->Match(inputs_y[0])) {
    inputs_y[1] = meta_fg_op_cnode;
  } else {
    inputs_y[0] = meta_fg_op_cnode;
  }
  return func_graph->NewCNodeBefore(node, inputs_y);
}
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
