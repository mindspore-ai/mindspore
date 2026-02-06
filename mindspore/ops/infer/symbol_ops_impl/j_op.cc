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
#include "mindspore/ops/infer/symbol_ops_impl/j_op.h"
#include <utility>
#include "abstract/symbolic_shape/operation_builder.h"
#include "mindspore/ops/infer/symbol_ops_impl/common.h"

namespace mindspore {
namespace symshape {
namespace ops {
REG_SYMBOL_OP_BUILDER(kJFuncCaller)
  .SetShapeDependN<DependOn::kShape>()
  .SetShapeFunc([](OperationBuilder *b) -> SymbolPtr {
    SymbolPtrList result(b->input_num());
    for (size_t i = 0; i < result.size(); i++) {
      result[i] = b->GetInputShape(i);
    }
    return ListSymbol::Make(std::move(result));
  });
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
