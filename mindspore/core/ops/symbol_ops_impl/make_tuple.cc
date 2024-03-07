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
#include <utility>

namespace mindspore {
namespace symshape {
namespace ops {
SymbolPtr MakeTupleBuilder(OperationBuilder *b) {
  SymbolPtrList result(b->input_num());
  if (b->is_building_shape()) {
    for (size_t i = 0; i < result.size(); i++) {
      result[i] = b->GetInputShape(i);
    }
  } else {
    for (size_t i = 0; i < result.size(); i++) {
      result[i] = b->GetInputValue(i);
    }
  }
  return ListSymbol::Make(std::move(result));
}

REG_SYMBOL_OP_BUILDER("MakeTuple").SetShapeFunc(MakeTupleBuilder).SetValueFunc(MakeTupleBuilder);
REG_SYMBOL_OP_BUILDER("RealMakeTuple").SetShapeFunc(MakeTupleBuilder).SetValueFunc(MakeTupleBuilder);
REG_SYMBOL_OP_BUILDER("make_list").SetShapeFunc(MakeTupleBuilder).SetValueFunc(MakeTupleBuilder);
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
