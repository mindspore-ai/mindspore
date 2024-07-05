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
#include "mindspore/core/ops/ops_func_impl/flash_attention_score.h"
#include "ops/op_enum.h"

namespace mindspore {
namespace symshape {
namespace ops {
using mindspore::ops::FASInputLayoutMode;

constexpr int64_t kFASLastDim = 8;
SymbolPtr FlashAttentionScoreShapeBuilder(OperationBuilder *b) {
  auto query_shape = b->GetInputShape(mindspore::ops::kFlashAttentionScoreInputQueryIndex)->as_sptr<ListSymbol>();
  if (!query_shape->HasData()) {
    // does not support dynamic rank.
    return nullptr;
  }
  auto MakeOutShape = [&query_shape](size_t batch_index, size_t seq_index, const SymbolPtr &head_num) {
    SymbolPtr shape;
    auto batch = query_shape->item(batch_index);
    auto seq = query_shape->item(seq_index);
    if (head_num != nullptr) {
      shape = ListSymbol::Make(SymbolPtrList{batch, head_num, seq, IntSymbol::Make(kFASLastDim)});
    } else {
      shape = ListSymbol::Make(SymbolPtrList{batch, seq, IntSymbol::Make(kFASLastDim)});
    }
    return ListSymbol::Make(SymbolPtrList{shape, shape, ListSymbol::Make({IntSymbol::Make(1LL)}), query_shape});
  };

  // For TND layout, the output softmax shape is 3D. otherwise, the output shape is 4D.
  auto input_layout = b->GetInputValue(mindspore::ops::kFlashAttentionScoreInputLayoutIndex)->as_sptr<IntSymbol>();
  if (!input_layout->HasData()) {
    return nullptr;
  }
  SymbolPtr head_num = nullptr;
  if (input_layout->value() != FASInputLayoutMode::TND) {
    head_num = b->GetInputValue(mindspore::ops::kFlashAttentionScoreInputHeadNumIndex);
  }
  switch (static_cast<FASInputLayoutMode>(input_layout->value())) {
    case FASInputLayoutMode::TND:
      return MakeOutShape(kIndex0, kIndex1, nullptr);
    case FASInputLayoutMode::SBH:
      return MakeOutShape(kIndex1, kIndex0, head_num);
    case FASInputLayoutMode::BNSD:
      return MakeOutShape(kIndex0, kIndex2, head_num);
    case FASInputLayoutMode::BSND:
    case FASInputLayoutMode::BSH:
      return MakeOutShape(kIndex0, kIndex1, head_num);
    default:
      break;
  }
  MS_LOG(EXCEPTION) << "FlashAttentionScore support input layout: BSH, BNSD, SBH, BSND, TND.";
  return nullptr;
}

REG_SYMBOL_OP_BUILDER("FlashAttentionScore")
  .SetShapeDepend([](const PrimitivePtr &, size_t) {
    std::vector<DependOn> depends(mindspore::ops::kFlashAttentionScoreInputsNum, DependOn::kNone);
    depends[mindspore::ops::kFlashAttentionScoreInputQueryIndex] = DependOn::kShape;
    depends[mindspore::ops::kFlashAttentionScoreInputLayoutIndex] = DependOn::kValue;
    depends[mindspore::ops::kFlashAttentionScoreInputHeadNumIndex] = DependOn::kValue;
    return depends;
  })
  .SetShapeFunc(FlashAttentionScoreShapeBuilder);

REG_SYMBOL_OP_BUILDER("FlashAttentionScoreGrad")
  .SetShapeDepend({DependOn::kShape, DependOn::kShape, DependOn::kShape, DependOn::kNone, DependOn::kShape})
  .SetShapeFunc([](OperationBuilder *b) { return ListSymbol::Make(b->GetSymbolsOfDepend()); });
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
