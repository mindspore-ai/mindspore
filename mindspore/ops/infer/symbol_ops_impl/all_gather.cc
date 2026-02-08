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
#include "abstract/symbolic_shape/operation_builder.h"
#include "mindspore/ops/infer/symbol_ops_impl/scalar_mul.h"
#include "ops_utils/op_constants.h"

namespace mindspore {
namespace symshape {
namespace ops {
REG_SYMBOL_OP_BUILDER("AllGather")
  .SetShapeDepend({DependOn::kShape})
  .SetShapeFunc([](OperationBuilder *b) -> SymbolPtr {
    auto x = b->GetInputShape(kIndex0);
    auto rank = b->GetAttr("rank_size");
    MS_EXCEPTION_IF_NULL(rank);
    if (!x->HasData() || !x->is<ListSymbol>()) {
      return nullptr;
    }
    auto result = x->as<ListSymbol>()->symbols();
    if (result.empty()) {
      return nullptr;
    }
    result[0] = b->Emit(std::make_shared<ScalarMul>(result[0], rank));
    return ListSymbol::Make(std::move(result));
  });
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
