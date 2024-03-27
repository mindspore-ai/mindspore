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
#include "mindspore/core/symbolic_shape/operation_builder.h"

namespace mindspore {
namespace symshape {
namespace ops {
constexpr int64_t kFASLastDim = 8;
SymbolPtr FlashAttentionScoreShapeBuilder(OperationBuilder *b) {
  SymbolPtr batch_size;
  SymbolPtr seq_len;
  SymbolPtr head_num = b->GetAttr("head_num");
  SymbolPtr last_dim = IntSymbol::Make(kFASLastDim);

  auto query_shape = b->GetInputShape(kIndex0)->as_sptr<ListSymbol>();
  MS_EXCEPTION_IF_NULL(query_shape);
  if (!query_shape->HasData()) {
    // does not support dynamic rank.
    return nullptr;
  }
  auto input_layout = b->GetAttr("input_layout");
  MS_EXCEPTION_IF_NULL(input_layout);
  auto layout = input_layout->as<StrSymbol>()->value();
  if (layout == "BSH") {
    batch_size = query_shape->item(kIndex0);
    seq_len = query_shape->item(kIndex1);
  } else {
    batch_size = query_shape->item(kIndex0);
    seq_len = query_shape->item(kIndex2);
  }
  auto shape2 = ListSymbol::Make({batch_size, head_num, seq_len, last_dim});
  return ListSymbol::Make(SymbolPtrList{shape2, shape2, ListSymbol::Make({IntSymbol::Make(1LL)}), query_shape});
}

REG_SYMBOL_OP_BUILDER("FlashAttentionScore")
  .SetShapeDepend({DependOn::kShape})
  .SetShapeFunc(FlashAttentionScoreShapeBuilder);
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
