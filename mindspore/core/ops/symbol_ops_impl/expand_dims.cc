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
class MS_CORE_API ExpandDims : public InferShapeOp {
 public:
  using InferShapeOp::InferShapeOp;
  ExpandDims(const SymbolPtr &input, const SymbolPtr &axis) : InferShapeOp({input, axis}) {}
  ~ExpandDims() override = default;
  MS_DECLARE_PARENT(ExpandDims, InferShapeOp)

 protected:
  SymbolPtr Eval() override;
};

SymbolPtr ExpandDims::Eval() {
  auto inp = input_as<ListSymbol>(0);
  auto axis = input(1);
  if (!inp->HasData() || !axis->HasData()) {
    if (inp->HasData() && axis->is<IntSymbol>()) {
      return GenVIntList(inp->size() + 1);
    }
    return GenVList();
  }
  DoNotEvalOnRun();
  auto rank = inp->size();
  SymbolPtrList result(inp->symbols());
  SymbolPtr const1 = GenInt(1);
  auto expand_dims = [&result, rank, &const1](int64_t axis_val) {
    if (axis_val + static_cast<int64_t>(rank) + 1 < 0 || axis_val > static_cast<int64_t>(rank)) {
      MS_LOG(INTERNAL_EXCEPTION) << "The axis value should be in range [" << -rank - 1 << "," << rank << "], but got "
                                 << axis_val;
    }
    (void)result.insert(result.begin() + LongToSize(NormAxis(axis_val, rank + 1)), const1);
  };
  auto axis_list = axis->as<ListSymbol>();
  if (axis_list == nullptr) {
    expand_dims(AsInt(axis));
  } else {
    for (size_t i = 0; i < axis_list->size(); i++) {
      expand_dims(AsInt(axis_list->item(i)));
    }
  }
  return ResultIntList(std::move(result));
}

REG_SYMBOL_OP_BUILDER("ExpandDims")
  .SetShapeDepend({DependOn::kShape, DependOn::kValue})
  .SetShapeFunc(DefaultBuilder<ExpandDims>)
  .SetValueFunc([](OperationBuilder *b) -> SymbolPtr {
    auto v = b->GetInputValue(kIndex0);
    // only support int to intlist for shape calculation
    return v->is<IntSymbol>() ? ListSymbol::Make({v}) : nullptr;
  });
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
