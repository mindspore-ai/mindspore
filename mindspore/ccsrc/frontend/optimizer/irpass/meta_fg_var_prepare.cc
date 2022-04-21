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
  };
  return meta_fg_ops;
}

static AnfNodePtr GenerateUnpackGraphNode(const AnfNodePtr &origin_node, std::vector<AnfNodePtr> inputs_y,
                                          const AnfNodePtr &func_node, bool is_unpack, bool sens_param) {
  MS_EXCEPTION_IF_NULL(func_node);
  FuncGraphPtr func_graph = origin_node->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  std::vector<AnfNodePtr> nodes;
  AnfNodePtr unpack_graph_node = nullptr;
  if (is_unpack) {
    auto unpack_graph = std::make_shared<prim::UnpackGraphPrimitive>(sens_param, true);
    nodes.push_back(NewValueNode(unpack_graph));
    nodes.push_back(func_node);
    // {unpackcall, {GradOperation, ...}, args...} and other {unpackcall, {meta_fg_opration, ...}, args...}
    const size_t inputs_begin_index = 2;
    (void)std::transform(inputs_y.begin() + inputs_begin_index, inputs_y.end(), std::back_inserter(nodes),
                         [](const AnfNodePtr &node) { return node; });
    unpack_graph_node = func_graph->NewCNodeBefore(origin_node, nodes);
  } else {
    auto unpack_graph = std::make_shared<prim::UnpackGraphPrimitive>(sens_param, false);
    nodes.push_back(NewValueNode(unpack_graph));
    nodes.push_back(func_node);
    // {{GradOperation, ...}, args...} and other {{meta_fg_opration, ...}, args...}
    const size_t inputs_begin_index = 1;
    (void)std::transform(inputs_y.begin() + inputs_begin_index, inputs_y.end(), std::back_inserter(nodes),
                         [](const AnfNodePtr &node) { return node; });
    unpack_graph_node = func_graph->NewCNodeBefore(origin_node, nodes);
  }
  return unpack_graph_node;
}

MetaFuncGraphPtr GetMetaFuncGraphOfValueNode(const AnfNodePtr &node) {
  ValuePtr value;
  if (IsValueNode<prim::DoSignaturePrimitive>(node)) {
    value = GetValueNode(node)->cast<prim::DoSignaturePrimitivePtr>()->function();
  } else {
    value = GetValueNode(node);
  }
  if (value == nullptr) {
    return nullptr;
  }
  return value->cast<MetaFuncGraphPtr>();
}

// check if node is a specific meta_fg_opration that registered in the meta_fg_ops
bool CheckMetaFgOps(const AnfNodePtr &node) {
  if (node == nullptr) {
    return false;
  }

  auto meta_func_graph_ptr = GetMetaFuncGraphOfValueNode(node);
  if (meta_func_graph_ptr == nullptr) {
    return false;
  }

  const auto &meta_fg_ops = GetMetaFgOps();
  for (auto meta_fg_op : meta_fg_ops) {
    if (meta_fg_op->Match(meta_func_graph_ptr)) {
      return true;
    }
  }
  return false;
}

// {{GradOperation, g, w}, Ys}, {UnPackCall, {GradOperation, g, w}, Ys},
// and other {{meta_fg_opration, ...}, ...} or {UnPackCall, {meta_fg_opration, ...}, ...}
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
    auto meta_func = GetMetaFuncGraphOfValueNode(inputs_x[0]);
    if (meta_func == nullptr) {
      return nullptr;
    }
    auto grad_op_ptr = meta_func->cast<prim::GradOperationPtr>();
    sens_param = grad_op_ptr->sens_param();
  }

  inputs_x[1] = GenerateUnpackGraphNode(node, inputs_y, func_node, is_unpack, sens_param);
  // construct new meta_fg_opration
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
