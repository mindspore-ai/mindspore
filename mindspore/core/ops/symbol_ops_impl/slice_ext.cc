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
#include "mindspore/core/ops/symbol_ops_impl/scalar_sub.h"

namespace mindspore {
namespace symshape {
namespace ops {
class MS_CORE_API SliceExt : public InferShapeOp {
 public:
  using InferShapeOp::InferShapeOp;
  ~SliceExt() override = default;
  MS_DECLARE_PARENT(SliceExt, InferShapeOp)
 protected:
  SymbolPtr Eval() override;
  SymbolPtr Process(const IntSymbolPtr &start, const IntSymbolPtr &end, size_t rank);
};

SymbolPtr SliceExt::Eval() {
  auto x = input_as_sptr<ListSymbol>(kIndex0);
  auto axis = input_as<IntSymbol>(kIndex1);
  auto start = input_as_sptr<IntSymbol>(kIndex2);
  auto end = input_as_sptr<IntSymbol>(kIndex3);
  auto step = input_as_sptr<IntSymbol>(kIndex4);
  if (!step->EqualsTo(kSym1)) {
    // only support step is 1
    MS_LOG(DEBUG) << "InferSymbolicShape of SliceExt only support step=1";
    return nullptr;
  }
  if (!x->HasData()) {
    return GenVList();
  }
  if (!axis->HasData()) {
    return GenVIntList(x->size());
  }
  auto axis_v = LongToSize(NormAxis(axis->value(), x->size()));
  auto result = x->symbols();
  // Assume 'end' is always greater than 'start'
  result.at(axis_v) = Emit(std::make_shared<ScalarSub>(end, start));
  return ResultIntList(std::move(result));
}

REG_SYMBOL_OP_BUILDER("SliceExt")
  .SetShapeDepend({DependOn::kShape, DependOn::kValue, DependOn::kValue, DependOn::kValue, DependOn::kValue})
  .SetShapeFuncWith<SliceExt>();
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
