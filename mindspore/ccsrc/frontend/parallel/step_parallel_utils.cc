/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "frontend/parallel/step_parallel_utils.h"

#include <inttypes.h>
#include <sys/time.h>
#include <algorithm>

#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>

#include "base/core_ops.h"
#include "frontend/operator/ops.h"
#include "frontend/optimizer/optimizer.h"
#include "frontend/parallel/context.h"
#include "frontend/parallel/device_manager.h"
#include "frontend/parallel/graph_util/generate_graph.h"
#include "frontend/parallel/graph_util/graph_info.h"
#include "frontend/parallel/graph_util/node_info.h"
#include "frontend/parallel/node_check.h"
#include "ir/param_info.h"
#include "ir/tensor.h"
#include "utils/trace_base.h"
#include "utils/comm_manager.h"
#include "utils/ms_context.h"
#include "utils/symbolic.h"
#include "mindspore/core/utils/parallel_node_check.h"

namespace mindspore {
namespace parallel {
bool IsSomePrimitive(const CNodePtr &cnode, const std::string &name) {
  if (!cnode) {
    return false;
  }
  ValueNodePtr anf_node = cnode->input(0)->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(anf_node);
  PrimitivePtr prim = anf_node->value()->cast<PrimitivePtr>();
  return (prim->name() == name);
}

bool IsParallelCareNode(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  ValueNodePtr prim_node = cnode->input(0)->cast<ValueNodePtr>();
  if (prim_node == nullptr) {
    return false;
  }
  PrimitivePtr prim = prim_node->value()->cast<PrimitivePtr>();
  if (prim == nullptr) {
    return false;
  }
  if (IsInParallelBlackList(prim)) {
    MS_LOG(DEBUG) << "Parallel don't care node: " << prim->name();
    return false;
  }
  // get_next is not in the forward graph, we need mark the get_next as the forward node
  if (prim->name() == GET_NEXT || prim->name() == VIRTUAL_OUTPUT) {
    return true;
  }
  if ((prim->name() == CAST) && !cnode->has_user_data<OperatorInfo>()) {
    return false;
  }

  return cnode->in_forward_flag();
}

Shapes GetValueListShape(const AnfNodePtr &node) {
  Shapes shapes;
  std::vector<ValuePtr> inputs_seq;
  if (IsValueNode<ValueList>(node)) {
    inputs_seq = node->cast<ValueNodePtr>()->value()->cast<ValueListPtr>()->value();
  } else if (IsValueNode<ValueTuple>(node)) {
    inputs_seq = node->cast<ValueNodePtr>()->value()->cast<ValueTuplePtr>()->value();
  } else {
    MS_LOG(EXCEPTION) << "node is eigther ValueList or ValueTuple";
  }
  for (auto &ele : inputs_seq) {
    auto tensor = ele->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(tensor);
    auto one_shape = tensor->shape();
    shapes.push_back(one_shape);
  }
  return shapes;
}

Shapes GetNodeShape(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  Shapes shapes;
  if (IsValueNode<ValueList>(node) || IsValueNode<ValueTuple>(node)) {
    return GetValueListShape(node);
  }
  BaseShapePtr base_shape_ptr = node->Shape();
  if (node->isa<CNode>()) {
    auto cnode = node->cast<CNodePtr>();
    if (IsValueNode<Primitive>(cnode->input(0))) {
      PrimitivePtr prim = GetValueNode<PrimitivePtr>(cnode->input(0));
      MS_EXCEPTION_IF_NULL(prim);
      if (prim->name() == MAKEREF) {
        AnfNodePtr ref_node = cnode->input(1);
        auto func_graph = cnode->func_graph();
        MS_EXCEPTION_IF_NULL(ref_node);
        MS_EXCEPTION_IF_NULL(func_graph);
        return GetRefKeyNodeShape(ref_node, func_graph);
      }
    }
    if (cnode->input(0)->isa<CNode>()) {
      if (cnode->inputs().size() < 2) {
        MS_LOG(EXCEPTION) << "GetNodeShape: " << node->ToString() << " size is smaller than 2";
      }
      base_shape_ptr = cnode->input(1)->Shape();
    }
  }
  if (base_shape_ptr == nullptr) {
    MS_LOG(EXCEPTION) << "GetNodeShape: " << node->ToString() << " shape_ptr is nullptr, full name is "
                      << node->fullname_with_scope();
  }
  auto tuple_shape_ptr = dyn_cast<abstract::SequeueShape>(base_shape_ptr);
  if (tuple_shape_ptr != nullptr) {
    auto tuple_shape = tuple_shape_ptr->shape();
    for (auto &shape : tuple_shape) {
      auto each_shape = dyn_cast<abstract::Shape>(shape);
      MS_EXCEPTION_IF_NULL(each_shape);
      shapes.push_back(each_shape->shape());
    }
  } else {
    auto shape_ptr = dyn_cast<abstract::Shape>(base_shape_ptr);
    MS_EXCEPTION_IF_NULL(shape_ptr);
    shapes.push_back(shape_ptr->shape());
  }
  return shapes;
}
}  // namespace parallel
}  // namespace mindspore
