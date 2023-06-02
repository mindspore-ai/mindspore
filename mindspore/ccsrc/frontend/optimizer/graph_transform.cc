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

#include "frontend/optimizer/graph_transform.h"
#include <vector>
#include <algorithm>
#include "mindspore/core/ops/sequence_ops.h"
#include "ir/graph_utils.h"

namespace mindspore {
/* namespace to support opt */
namespace opt {
bool ContainSparseTensor(const abstract::AbstractBasePtr &abs) {
  MS_EXCEPTION_IF_NULL(abs);
  if (abs->isa<abstract::AbstractSparseTensor>()) {
    return true;
  }
  if (abs->isa<abstract::AbstractTuple>()) {
    auto vec = abs->cast_ptr<abstract::AbstractTuple>()->elements();
    return std::any_of(vec.begin(), vec.end(), ContainSparseTensor);
  }
  return false;
}

bool IsTuple(const AnfNodePtr &param) {
  return param->abstract() != nullptr && param->abstract()->isa<abstract::AbstractTuple>();
}

bool IsSequenceExpandable(const AbstractBasePtr &abs) {
  if (abs == nullptr) {
    return false;
  }
  auto abs_seq = abs->cast<abstract::AbstractTuplePtr>();
  if (abs_seq == nullptr) {
    return false;
  }
  // Not expand SparseTensor.
  if (common::AnfAlgo::CheckAbsSparseTensor(abs)) {
    return false;
  }
  // Not expand dynamic len sequence.
  if (abs_seq->dynamic_len()) {
    return false;
  }
  // Not expand node which is arg of dynamic len parameter.
  return !abs_seq->dyn_len_arg();
}

bool ParamContainSparseTensor(const AnfNodePtr &param) {
  // If SparseTensor, Tuple(SparseTensor,...) or Tuple(...,(..., SparseTensor)), return false and skip this pass.
  return param->abstract() != nullptr && ContainSparseTensor(param->abstract());
}

bool FuncGraphHasTupleInput(const FuncGraphPtr &fg) {
  return std::any_of(fg->parameters().cbegin(), fg->parameters().cend(), IsTuple);
}

bool FuncGraphHasConstantTupleInput(const FuncGraphPtr &fg) {
  return std::any_of(fg->parameters().cbegin(), fg->parameters().cend(),
                     [](const AnfNodePtr &param) { return IsSequenceExpandable(param->abstract()); });
}

std::vector<AnfNodePtr> TransformTupleArgument(const FuncGraphPtr &fg, const AnfNodePtr &node,
                                               const abstract::AbstractTuplePtr &abs) {
  auto &elements = abs->elements();
  std::vector<AnfNodePtr> tuple_node_expanded;
  for (size_t i = 0; i < elements.size(); i++) {
    auto idx = NewValueNode(SizeToLong(i));
    auto abstract_scalar = std::make_shared<abstract::AbstractScalar>(std::make_shared<Int64Imm>(SizeToLong(i)));
    idx->set_abstract(abstract_scalar);
    auto elem_node = fg->NewCNode({NewValueNode(prim::kPrimTupleGetItem), node, idx});
    elem_node->set_abstract(elements[i]);
    if (IsSequenceExpandable(elements[i])) {
      auto nodes = TransformTupleArgument(fg, elem_node, elements[i]->cast<abstract::AbstractTuplePtr>());
      (void)tuple_node_expanded.insert(tuple_node_expanded.cend(), nodes.cbegin(), nodes.cend());
    } else {
      tuple_node_expanded.push_back(elem_node);
    }
  }
  return tuple_node_expanded;
}
}  // namespace opt
}  // namespace mindspore
