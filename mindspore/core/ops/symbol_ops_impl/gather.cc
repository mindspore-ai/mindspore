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
#include "mindspore/core/ops/symbol_ops_impl/common.h"

namespace mindspore {
namespace symshape {
namespace ops {
class MS_CORE_API Gather : public InferShapeOp {
 public:
  using InferShapeOp::InferShapeOp;
  Gather(const SymbolPtr &param, const SymbolPtr &indices, const SymbolPtr &axis, const SymbolPtr &batch_dims)
      : InferShapeOp({param, indices, axis, batch_dims}) {}
  ~Gather() override = default;
  MS_DECLARE_PARENT(Gather, InferShapeOp)
 protected:
  SymbolPtr Eval() override;
};

SymbolPtr Gather::Eval() {
  auto params = input_as<ListSymbol>(kIndex0);
  auto indices = input_as<ListSymbol>(kIndex1);
  auto axis = input_as<IntSymbol>(kIndex2);
  auto batch_dims = input_as<IntSymbol>(kIndex3)->value();
  if (!params->HasData() || !indices->HasData() || !axis->HasData()) {
    return GenVList();
  }
  DoNotEvalOnRun();
  auto axis_val = LongToSize(NormAxis(axis->value(), params->size()));
  batch_dims = NormAxis(batch_dims, indices->size());
  SymbolPtrList result;
  result.reserve(params->size() + indices->size());
  MS_EXCEPTION_IF_CHECK_FAIL(axis_val < params->size(), "axis out of params size.");
  for (size_t i = 0; i < axis_val; i++) {
    (void)result.emplace_back(params->symbols()[i]);
  }
  for (size_t i = LongToSize(batch_dims); i < indices->size(); i++) {
    (void)result.emplace_back(indices->symbols()[i]);
  }
  for (size_t i = axis_val + 1; i < params->size(); i++) {
    (void)result.emplace_back(params->symbols()[i]);
  }
  return ResultIntList(std::move(result));
}

REG_SYMBOL_OP_BUILDER("Gather")
  .SetShapeDepend({DependOn::kShape, DependOn::kShape, DependOn::kValue, DependOn::kValue})
  .SetShapeFunc([](OperationBuilder *b) -> SymbolPtr {
    auto params = b->GetInputShape(kIndex0);
    auto indices = b->GetInputShape(kIndex1);
    auto axis = b->GetInputValue(kIndex2);
    auto batch_dims = b->GetInputOrAttr(kIndex3, kAttrBatchDims);  // todo, change to DefaultBuilder
    return b->Emit(std::make_shared<Gather>(params, indices, axis, batch_dims));
  });
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
