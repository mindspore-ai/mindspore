/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "include/common/expander/core/infer.h"

#include <algorithm>
#include "mindspore/core/ops/other_ops.h"
#include "mindspore/core/ops/nn_optimizer_ops.h"
#include "mindspore/core/ops/nn_ops.h"
#include "mindspore/core/ops/math_ops.h"
#include "mindspore/core/ops/image_ops.h"
#include "mindspore/core/ops/array_ops.h"
#include "abstract/ops/primitive_infer_map.h"
#include "ir/anf.h"
#include "ir/primitive.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace expander {
void CppInfer::InferAnfnode(const AnfNodePtr &anfnode) const {
  if (anfnode->isa<ValueNode>()) {
    anfnode->set_abstract(anfnode->cast<ValueNodePtr>()->value()->ToAbstract());
    return;
  }
  auto cnode = anfnode->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto prim = GetCNodePrimitive(cnode);
  MS_EXCEPTION_IF_NULL(prim);
  AbstractBasePtrList abs_list;
  abs_list.reserve(cnode->size());
  (void)std::transform(cnode->inputs().cbegin() + 1, cnode->inputs().cend(), std::back_inserter(abs_list),
                       [](const AnfNodePtr &node) {
                         const auto &abs = node->abstract();
                         if (abs == nullptr) {
                           MS_EXCEPTION_IF_CHECK_FAIL(node->isa<ValueNode>(), node->ToString() + " has no abstract");
                           return node->cast<ValueNodePtr>()->value()->ToAbstract();
                         }
                         return abs;
                       });
  auto &infer_impl = CppInfer::infer_impl_cache()[prim];
  if (infer_impl.Get() == nullptr) {
    auto found = abstract::GetPrimitiveInferImpl(prim);
    if (found.has_value() && found.value().IsImplInferShapeAndType()) {
      infer_impl = found.value();
    } else {
      MS_LOG(EXCEPTION) << "The infer function of [" << prim->name() << "] is not defined.";
    }
  }
  cnode->set_abstract(infer_impl.InferShapeAndType(nullptr, prim, abs_list));
}

BaseShapePtr CppInfer::GetShape(const NodePtr &node) {
  auto abs = GetAbstract(node);
  MS_EXCEPTION_IF_NULL(abs);
  return abs->BuildShape();
}

TypePtr CppInfer::GetDtype(const NodePtr &node) {
  auto abs = GetAbstract(node);
  MS_EXCEPTION_IF_NULL(abs);
  return abs->BuildType();
}
}  // namespace expander
}  // namespace mindspore
