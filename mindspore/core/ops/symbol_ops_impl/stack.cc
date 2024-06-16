/**
 * Copyright 2023-2024 Huawei Technologies Co., Ltd
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
#include "mindspore/core/ops/symbol_ops_impl/make_tuple.h"
#include "mindspore/core/utils/check_convert_utils.h"

namespace mindspore {
namespace symshape {
namespace ops {
// only support IntList value for shape calculation
SymbolPtr StackValue(OperationBuilder *b) {
  SymbolPtrList result;
  if (b->input_num() == 1) {
    return b->GetInputValue(kIndex0);
  }
  // inputs of Stack is spread.
  return MakeTupleBuilder(b);
}

class MS_CORE_API Stack : public InferShapeOp {
 public:
  Stack(const SymbolPtr &inputs, const SymbolPtr &axis) : InferShapeOp({inputs, axis}) {}
  ~Stack() override = default;
  MS_DECLARE_PARENT(Stack, InferShapeOp)

 protected:
  SymbolPtr Eval() override;
};

SymbolPtr Stack::Eval() {
  auto inputs = input_as<ListSymbol>(kIndex0);
  auto axis = input_as<IntSymbol>(kIndex1)->value();
  SymbolPtrList result;
  bool has_valid_shape = false;
  for (size_t i = 0; i < inputs->size(); i++) {
    auto elem = inputs->item_as<ListSymbol>(i);
    if (!elem->HasData()) {
      continue;
    }
    has_valid_shape = true;
    if (result.empty()) {
      result = elem->symbols();
      if (!is_building()) {
        break;
      }
    } else if (result.size() != elem->size()) {
      MS_LOG(INTERNAL_EXCEPTION) << "The size of inputs of stack should be equal, but got " << result.size() << " vs "
                                 << elem->size();
    } else if (is_building()) {
      for (size_t j = 0; j < result.size(); j++) {
        if (!result[j]->HasData() && elem->item(j)->HasData()) {
          result[j] = elem->item(j);
        }
      }
    }
  }
  if (!has_valid_shape) {
    return GenVList();
  }
  DoNotEvalOnRun();
  auto rank = static_cast<int64_t>(result.size());
  if (axis < -rank - 1 || axis > rank) {
    MS_LOG(INTERNAL_EXCEPTION) << "For 'Stack', the axis should be in range [-rank-1, rank], which rank=" << rank
                               << ". but got axis=" << axis;
  }
  result.insert(result.begin() + NormAxis(axis, result.size() + 1), GenInt(static_cast<int64_t>(inputs->size())));
  return ResultIntList(std::move(result));
}

REG_SYMBOL_OP_BUILDER("Stack")
  .SetShapeDependN<DependOn::kShape>()
  .SetShapeFunc([](OperationBuilder *b) -> SymbolPtr {
    auto axis = b->GetAttr(kAttrAxis);
    if (b->input_num() == 1 && b->GetInput(0)->isa<abstract::AbstractSequence>()) {
      return b->Emit(std::make_shared<Stack>(b->GetInputShape(0), axis));
    }
    SymbolPtrList inputs;
    inputs.reserve(b->input_num());
    for (size_t i = 0; i < b->input_num(); i++) {
      (void)inputs.emplace_back(b->GetInputShape(i));
    }
    return b->Emit(std::make_shared<Stack>(ListSymbol::Make(std::move(inputs)), axis));
  })
  .SetValueDependN<DependOn::kValue>()
  .SetValueFunc(StackValue);
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
