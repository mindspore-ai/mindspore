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
#include "mindspore/core/ops/symbol_ops_impl/common.h"

namespace mindspore {
namespace symshape {
namespace ops {
class MS_CORE_API Squeeze : public InferShapeOp {
 public:
  using InferShapeOp::InferShapeOp;
  Squeeze(const SymbolPtr &input, const SymbolPtr &axis) : InferShapeOp({input, axis}) {}
  ~Squeeze() override = default;
  MS_DECLARE_PARENT(Squeeze, InferShapeOp)

 protected:
  SymbolPtr Eval() override;
};

SymbolPtr Squeeze::Eval() {
  auto input = input_as<ListSymbol>(kIndex0);
  if (!input->HasData()) {
    return GenVList();
  }
  auto axis = NormAxis(input_as<ListSymbol>(kIndex1), input->size());
  SymbolPtrList result;
  result.reserve(input->size());
  if (axis.empty()) {
    for (size_t i = 0; i < input->size(); i++) {
      if (input->item(i)->HasData() && input->item_as<IntSymbol>(i)->value() == 1LL) {
        continue;
      }
      (void)result.emplace_back(input->item(i));
    }
  } else {
    for (size_t i = 0; i < input->size(); i++) {
      if (axis.count(static_cast<int64_t>(i)) > 0) {
        continue;
      }
      (void)result.emplace_back(input->item(i));
    }
  }
  return ResultIntList(std::move(result));
}

REG_SYMBOL_OP_BUILDER("Squeeze").SetShapeDepend({DependOn::kShape}).SetShapeFunc([](OperationBuilder *b) -> SymbolPtr {
  auto input_shape = b->GetInputShape(kIndex0);
  auto axis = b->GetAttr(kAttrAxis);
  MS_EXCEPTION_IF_NULL(axis);
  return b->Emit(std::make_shared<Squeeze>(input_shape, axis));
});
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
