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
#include "utils/check_convert_utils.h"
#include "mindspore/core/ops/symbol_ops_impl/common.h"
#include "mindspore/core/ops/symbol_ops_impl/scalar_add.h"

namespace mindspore {
namespace symshape {
namespace ops {
// only support IntList value for shape calculation
SymbolPtr ConcatValue(OperationBuilder *b) {
  SymbolPtrList result;
  if (b->input_num() == kDim2) {
    // inputs of Concat is a tuple.
    auto inputs = b->GetInputValue(kIndex0)->as_sptr_noexcept<ListSymbol>();
    if (inputs == nullptr) {
      return nullptr;
    }
    result.reserve(inputs->size());
    for (auto &inp : inputs->symbols()) {
      if (auto ilist = inp->as_noexcept<ListSymbol>(); ilist != nullptr && ilist->HasData()) {
        (void)result.insert(result.end(), ilist->symbols().begin(), ilist->symbols().end());
      } else if (inp->is<IntSymbol>()) {
        (void)result.emplace_back(inp);
      } else {
        return nullptr;
      }
    }
  } else {
    // inputs of Concat are spread, and the last input is "axis".
    // todo, remove this branch
    result.reserve(b->input_num());
    for (size_t i = 0; i + 1 < b->input_num(); i++) {
      auto v = b->GetInputValue(i)->as_sptr_noexcept<ListSymbol>();
      if (v != nullptr) {
        (void)result.insert(result.end(), v->symbols().begin(), v->symbols().end());
      } else {
        return nullptr;
      }
    }
  }
  return ListSymbol::Make(std::move(result));
}

class MS_CORE_API Concat : public InferShapeOp {
 public:
  using InferShapeOp::InferShapeOp;
  ~Concat() override = default;
  MS_DECLARE_PARENT(Concat, InferShapeOp)

 protected:
  SymbolPtr Eval() override;
  std::pair<SymbolPtrList, bool> FindOutputShape(const ListSymbol *inputs) const;
};

std::pair<SymbolPtrList, bool> Concat::FindOutputShape(const ListSymbol *inputs) const {
  SymbolPtrList out_shape;
  bool has_dynrank = false;
  for (size_t i = 0; i < inputs->size(); i++) {
    if (!inputs->item(i)->HasData()) {
      has_dynrank = true;
    } else if (out_shape.empty()) {
      out_shape = inputs->item_as<ListSymbol>(i)->symbols();
    }
  }
  return std::make_pair(out_shape, has_dynrank);
}

SymbolPtr Concat::Eval() {
  auto inputs = input_as<ListSymbol>(kIndex0);
  auto axis = input_as<IntSymbol>(kIndex1);
  if (!inputs->HasData() || !axis->HasData()) {
    return out_abstract()->GetShape()->BuildSymbolicShape();
  }
  auto [out_shape, has_dynrank] = FindOutputShape(inputs);
  if (out_shape.empty()) {
    return GenVList();
  }
  auto axis_v = LongToSize(NormAxis(axis->value(), out_shape.size()));
  if (has_dynrank) {
    out_shape[axis_v] = GenVInt();
    return GenList(std::move(out_shape));
  }
  DoNotEvalOnRun();
  SymbolPtrList concat_dims;
  concat_dims.reserve(inputs->size());
  for (size_t i = 0; i < inputs->size(); i++) {
    concat_dims.emplace_back(inputs->item_as<ListSymbol>(i)->item(axis_v));
  }
  out_shape[axis_v] = Accumulate<ScalarAdd>(concat_dims, emitter());
  return GenList(std::move(out_shape));
}

SymbolPtr ConcatShape(OperationBuilder *b) {
  if (!CheckAndConvertUtils::IsTensor(b->GetInput(kIndex0))) {
    return b->Emit(std::make_shared<Concat>(SymbolPtrList{b->GetInputShape(kIndex0), b->GetInputValue(kIndex1)}));
  } else {
    SymbolPtrList inputs;
    inputs.reserve(b->input_num() - 1);
    for (size_t i = 0; i + 1 < b->input_num(); i++) {
      (void)inputs.emplace_back(b->GetInputShape(i));
    }
    SymbolPtr data = ListSymbol::Make(std::move(inputs));
    return b->Emit(std::make_shared<Concat>(SymbolPtrList{data, b->GetInputValue(b->input_num() - 1)}));
  }
}

REG_SYMBOL_OP_BUILDER("Concat")
  .SetShapeDepend([](const PrimitivePtr &, size_t input_num) {
    std::vector<DependOn> depends(input_num, DependOn::kShape);
    depends.back() = DependOn::kValue;
    return depends;
  })
  .SetShapeFunc(ConcatShape)
  .SetValueDependN<DependOn::kValue>()
  .SetValueFunc(ConcatValue);
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
