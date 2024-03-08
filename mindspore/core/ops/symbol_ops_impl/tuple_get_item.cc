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
#include "mindspore/core/ops/symbol_ops_impl/make_tuple.h"

namespace mindspore {
namespace symshape {
namespace ops {
SymbolPtr TupleGetItemBuilder(OperationBuilder *b) {
  ListSymbolPtr input = nullptr;
  if (b->is_building_shape()) {
    input = b->GetInputShape(kIndex0)->as_sptr<ListSymbol>();
  } else {
    input = b->GetInputValue(kIndex0)->as_sptr<ListSymbol>();
  }
  MS_EXCEPTION_IF_NULL(input);
  int64_t index = GetValue<int64_t>(b->GetInput(kIndex1)->GetValue());
  return input->item(index);
}

REG_SYMBOL_OP_BUILDER("TupleGetItem").SetShapeFunc(TupleGetItemBuilder).SetValueFunc(TupleGetItemBuilder);
REG_SYMBOL_OP_BUILDER("RealTupleGetItem").SetShapeFunc(TupleGetItemBuilder).SetValueFunc(TupleGetItemBuilder);
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
