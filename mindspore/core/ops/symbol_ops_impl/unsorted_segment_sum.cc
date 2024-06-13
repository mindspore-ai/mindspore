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
class MS_CORE_API UnsortedSegmentSum : public InferShapeOp {
 public:
  using InferShapeOp::InferShapeOp;
  ~UnsortedSegmentSum() override = default;
  MS_DECLARE_PARENT(UnsortedSegmentSum, InferShapeOp)
 protected:
  SymbolPtr Eval() override;
};

SymbolPtr UnsortedSegmentSum::Eval() {
  auto x = input_as<ListSymbol>(kIndex0);
  auto seg_ids = input_as<ListSymbol>(kIndex1);
  if (!x->HasData() || !seg_ids->HasData()) {
    return GenVList();
  }
  DoNotEvalOnRun();
  SymbolPtrList result{input(kIndex2)};
  result.reserve(x->size() - seg_ids->size() + 1);
  (void)result.insert(result.end(), x->symbols().begin() + seg_ids->size(), x->symbols().end());
  return ResultIntList(std::move(result));
}

REG_SYMBOL_OP_BUILDER("UnsortedSegmentSum")
  .SetShapeDepend({DependOn::kShape, DependOn::kShape, DependOn::kValue})
  .SetShapeFuncWith<UnsortedSegmentSum>();
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
