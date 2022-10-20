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

#include "common/graph_kernel/bprop/expander/infer.h"

#include <algorithm>
#include "abstract/ops/primitive_infer_map.h"
#include "ir/anf.h"
#include "ir/primitive.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace expander {
void CppInfer::Infer(const NodePtr &node) {
  auto anfnode = node->get();
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
  AbstractBasePtr result = nullptr;
  auto &frontend_infer_func = abstract::GetPrimitiveToEvalImplMap();
  auto iter = frontend_infer_func.find(prim);
  if (iter != frontend_infer_func.end()) {
    MS_EXCEPTION_IF_CHECK_FAIL(iter->second.IsImplInferShapeAndType(), "There is no infer-abstract implement!");
    result = iter->second.InferShapeAndType(nullptr, prim, abs_list);
  } else {
    auto &backend_infer_func = abstract::GetPrimitiveToBackendEvalImplMap();
    auto iter2 = backend_infer_func.find(prim);
    if (iter2 != backend_infer_func.end()) {
      MS_EXCEPTION_IF_CHECK_FAIL(iter2->second.IsImplInferShapeAndType(), "There is no infer-abstract implement!");
      result = iter2->second.InferShapeAndType(nullptr, prim, abs_list);
    } else {
      MS_LOG(EXCEPTION) << "The infer function of [" << prim->name() << "] is not defined.";
    }
  }
  cnode->set_abstract(result);
}
}  // namespace expander
}  // namespace mindspore
