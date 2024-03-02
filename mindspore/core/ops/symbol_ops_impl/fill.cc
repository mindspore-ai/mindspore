/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "mindspore/core/symbolic_shape/operation_builder.h"
#include "mindspore/core/ops/symbol_ops_impl/common.h"

namespace mindspore {
namespace symshape {
namespace ops {
REG_SYMBOL_OP_BUILDER("FillV2")
  .SetShapeDepend({DependOn::kValue, DependOn::kNone})
  .SetShapeFunc([](OperationBuilder *b) {
    auto s = b->GetInputValue(0);
    auto symbolic_shape = s->as<ListSymbol>();
    MS_EXCEPTION_IF_NULL(symbolic_shape);
    InferShapeOp::SetPositive(symbolic_shape);
    return s;
  });
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
