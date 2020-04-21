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

#include "optimizer/irpass/grad_var_prepare.h"
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <memory>

#include "operator/composite/composite.h"
#include "operator/ops.h"
#include "optimizer/irpass.h"
#include "optimizer/optimizer.h"
#include "ir/visitor.h"
#include "ir/func_graph.h"
#include "ir/func_graph_cloner.h"

namespace mindspore {
namespace opt {
namespace irpass {

static AnfNodePtr GenerateUnpackGraphNode(std::vector<AnfNodePtr> inputs_y, FuncGraphPtr func_graph,
                                          AnfNodePtr func_node, bool is_unpack, bool sens_param) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(func_node);
  std::vector<AnfNodePtr> nodes;
  AnfNodePtr unpack_graph_node = nullptr;
  if (is_unpack) {
    auto unpack_graph = std::make_shared<prim::UnpackGraphPrimitive>("unpack_graph", sens_param, true);
    nodes.push_back(NewValueNode(unpack_graph));
    nodes.push_back(func_node);
    // {unpackcall, {GradOperation, ...}, args...}
    std::transform(inputs_y.begin() + 2, inputs_y.end(), std::back_inserter(nodes),
                   [](const AnfNodePtr &node) { return node; });
    unpack_graph_node = func_graph->NewCNode(nodes);
  } else {
    auto unpack_graph = std::make_shared<prim::UnpackGraphPrimitive>("unpack_graph", sens_param, false);
    nodes.push_back(NewValueNode(unpack_graph));
    nodes.push_back(func_node);
    // {{GradOperation, ...}, args...}
    std::transform(inputs_y.begin() + 1, inputs_y.end(), std::back_inserter(nodes),
                   [](const AnfNodePtr &node) { return node; });
    unpack_graph_node = func_graph->NewCNode(nodes);
  }
  return unpack_graph_node;
}

// get metagraph of value node
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

// check if node is a specific metafuncgraph op
bool IsMetaFuncGraph(const AnfNodePtr &node, const MetaFuncGraphPtr meta_func_graph) {
  if (node != nullptr) {
    auto meta_func_graph_ptr = GetMetaFuncGraphOfValueNode(node);
    if (meta_func_graph_ptr == nullptr) {
      return false;
    }

    if (meta_func_graph_ptr->type_name() == meta_func_graph->type_name()) {
      return true;
    }
  }
  return false;
}

// {{GradOperation, g, w}, Ys}
// {UnPackCall, {GradOperation, g, w}, Ys}
AnfNodePtr GradVarPrepare::operator()(const OptimizerPtr &, const AnfNodePtr &node) {
  if (!node->isa<CNode>() || node->func_graph() == nullptr) {
    return nullptr;
  }

  // {{...}, Ys}
  auto inputs_y = node->cast<CNodePtr>()->inputs();
  std::vector<AnfNodePtr> inputs_x;
  if (IsCNode(inputs_y[0])) {
    inputs_x = inputs_y[0]->cast<CNodePtr>()->inputs();
  } else if (IsMetaFuncGraph(inputs_y[0], unpack_op_) && IsCNode(inputs_y[1])) {
    inputs_x = inputs_y[1]->cast<CNodePtr>()->inputs();
  } else {
    return nullptr;
  }

  // {{...}, Xs}
  if (inputs_x.size() < 2) {
    return nullptr;
  }

  // {GradOperation, g, w} or {GradOperation, g}
  if (!IsMetaFuncGraph(inputs_x[0], grad_op_)) {
    return nullptr;
  }

  auto meta_func = GetMetaFuncGraphOfValueNode(inputs_x[0]);
  if (meta_func == nullptr) {
    return nullptr;
  }
  auto grad_op_ptr = meta_func->cast<prim::GradOperationPtr>();
  auto func_node = inputs_x[1];
  if (!IsValueNode<FuncGraph>(func_node)) {
    return nullptr;
  }

  AnfNodePtr unpack_graph_node =
    GenerateUnpackGraphNode(inputs_y, node->cast<CNodePtr>()->func_graph(), func_node,
                            IsMetaFuncGraph(inputs_y[0], unpack_op_), grad_op_ptr->sens_param());
  // constuct new grad_opration
  inputs_x[1] = unpack_graph_node;
  auto grad_op_cnode = node->func_graph()->NewCNode(inputs_x);
  if (IsMetaFuncGraph(inputs_y[0], unpack_op_)) {
    inputs_y[1] = grad_op_cnode;
  } else {
    inputs_y[0] = grad_op_cnode;
  }
  auto cnode = node->func_graph()->NewCNode(inputs_y);
  return cnode;
}
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
