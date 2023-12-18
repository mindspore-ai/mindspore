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
class MS_CORE_API OneHot : public InferShapeOp {
 public:
  using InferShapeOp::InferShapeOp;
  OneHot(const SymbolPtr &indices, const SymbolPtr &depth, const SymbolPtr &axis)
      : InferShapeOp({indices, depth, axis}) {}
  ~OneHot() override = default;
  MS_DECLARE_PARENT(OneHot, InferShapeOp)
 protected:
  SymbolPtr Eval() override;
};

SymbolPtr OneHot::Eval() {
  auto indices = input_as<ListSymbol>(kIndex0);
  auto depth = input(kIndex1);
  auto axis = input_as<IntSymbol>(kIndex2)->value();
  if (!indices->HasData()) {
    return GenVList();
  }
  SymbolPtrList result = indices->symbols();
  if (axis >= 0) {
    MS_EXCEPTION_IF_CHECK_FAIL(static_cast<size_t>(axis) <= result.size(), "axis out of range of input size");
    (void)result.insert(result.begin() + static_cast<size_t>(axis), depth);
  } else {
    (void)result.emplace_back(depth);
  }
  return ResultIntList(std::move(result));
}

REG_SYMBOL_OP_BUILDER("OneHot")
  .SetShapeDepend({DependOn::kShape, DependOn::kValue, DependOn::kNone, DependOn::kNone, DependOn::kValue})
  .SetShapeFunc([](OperationBuilder *b) -> SymbolPtr {
    auto indices = b->GetInputShape(kIndex0);
    auto depth = b->GetInputValue(kIndex1);
    auto axis = b->GetInputOrAttr(kIndex4, kAttrAxis);  // todo, change to DefaultBuilder
    MS_EXCEPTION_IF_NULL(axis);
    return b->Emit(std::make_shared<OneHot>(indices, depth, axis));
  });
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
