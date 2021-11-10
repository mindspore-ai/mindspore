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

#include "utils/anf_utils.h"
#include "base/core_ops.h"
#include "utils/trace_base.h"
#include "utils/utils.h"

namespace mindspore {
bool AnfUtils::IsDimUnknown(const abstract::ShapePtr &shape) {
  MS_EXCEPTION_IF_NULL(shape);
  return std::any_of(shape->shape().begin(), shape->shape().end(), [](int64_t s) { return s < -1; });
}

bool AnfUtils::IsShapeDynamic(const abstract::ShapePtr &shape) {
  MS_EXCEPTION_IF_NULL(shape);
  return std::any_of(shape->shape().begin(), shape->shape().end(), [](int64_t s) { return s < 0; });
}

bool AnfUtils::IsShapeDynamic(const std::vector<size_t> &shape) {
  return std::any_of(shape.begin(), shape.end(), [](int64_t s) { return s < 0; });
}

bool AnfUtils::IsNodeOutputDynamicShape(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto base_shape = node->Shape();
  if (base_shape == nullptr) {
    MS_LOG(INFO) << "Invalid base shape, node: " << node->fullname_with_scope();
    return false;
  }
  if (base_shape->isa<abstract::Shape>() && IsShapeDynamic(base_shape->cast<abstract::ShapePtr>())) {
    return true;
  } else if (base_shape->isa<abstract::TupleShape>()) {
    auto tuple_shape = base_shape->cast<abstract::TupleShapePtr>();
    MS_EXCEPTION_IF_NULL(tuple_shape);
    for (size_t i = 0; i < tuple_shape->size(); i++) {
      auto b_shape = (*tuple_shape)[i];
      if (b_shape->isa<abstract::Shape>() && IsShapeDynamic(b_shape->cast<abstract::ShapePtr>())) {
        return true;
      }
    }
  }
  return false;
}

bool AnfUtils::IsDimUnknown(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto base_shape = node->Shape();
  if (base_shape == nullptr) {
    MS_LOG(INFO) << "Invalid base shape, node: " << node->fullname_with_scope();
    return false;
  }
  if (base_shape->isa<abstract::Shape>()) {
    auto base_shape_ptr = base_shape->cast<abstract::ShapePtr>();
    MS_EXCEPTION_IF_NULL(base_shape_ptr);
    return base_shape_ptr->IsDimUnknown();
  } else if (base_shape->isa<abstract::TupleShape>()) {
    auto tuple_shape_ptr = base_shape->cast<abstract::TupleShapePtr>();
    MS_EXCEPTION_IF_NULL(tuple_shape_ptr);
    return tuple_shape_ptr->IsDimUnknown();
  } else if (base_shape->isa<abstract::SequeueShape>()) {
    auto seq_shape_ptr = base_shape->cast<abstract::SequeueShapePtr>();
    MS_EXCEPTION_IF_NULL(seq_shape_ptr);
    return seq_shape_ptr->IsDimUnknown();
  } else if (base_shape->isa<abstract::ListShape>()) {
    auto list_shape_ptr = base_shape->cast<abstract::ListShapePtr>();
    MS_EXCEPTION_IF_NULL(list_shape_ptr);
    return list_shape_ptr->IsDimUnknown();
  }
  return false;
}

bool AnfUtils::IsRealKernel(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
#ifndef ENABLE_SECURITY
  static const PrimitiveSet virtual_prims = {
    prim::kPrimImageSummary, prim::kPrimScalarSummary, prim::kPrimTensorSummary, prim::kPrimHistogramSummary,
    prim::kPrimMakeTuple,    prim::kPrimStateSetItem,  prim::kPrimTupleGetItem,  prim::kPrimReturn,
    prim::kPrimPartial,      prim::kPrimDepend,        prim::kPrimUpdateState,   prim::kPrimLoad};
#else
  static const PrimitiveSet virtual_prims = {prim::kPrimMakeTuple,   prim::kPrimStateSetItem, prim::kPrimTupleGetItem,
                                             prim::kPrimReturn,      prim::kPrimPartial,      prim::kPrimDepend,
                                             prim::kPrimUpdateState, prim::kPrimLoad};
#endif
  auto cnode = node->cast<CNodePtr>();
  if (cnode == nullptr) {
    // parameter and value node is a real kernel too
    return true;
  }
  if (cnode->size() == 0) {
    MS_LOG(EXCEPTION) << "Illegal null input of cnode(%s)" << node->DebugString()
                      << " trace: " << trace::DumpSourceLines(node);
  }
  return !IsOneOfPrimitive(cnode->input(kAnfPrimitiveIndex), virtual_prims);
}

bool AnfUtils::IsRealCNodeKernel(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return false;
  }
  if (IsPrimitiveCNode(node, prim::kPrimReturn)) {
    return true;
  }
  return AnfUtils::IsRealKernel(node);
}
}  // namespace mindspore
