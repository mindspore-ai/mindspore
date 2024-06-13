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
class MS_CORE_API LinSpace : public InferShapeOp {
 public:
  using InferShapeOp::InferShapeOp;
  ~LinSpace() override = default;
  MS_DECLARE_PARENT(LinSpace, InferShapeOp)

 protected:
  SymbolPtr Eval() override;
};

SymbolPtr LinSpace::Eval() {
  auto shape = input_as<ListSymbol>(kIndex0);
  auto step = input_as_sptr<IntSymbol>(kIndex1);
  if (!shape->HasData()) {
    return GenVList();
  }
  DoNotEvalOnRun();
  auto result = shape->symbols();
  (void)result.emplace_back(step);
  return ResultIntList(std::move(result));
}

REG_SYMBOL_OP_BUILDER("LinSpace")
  .SetShapeDepend({DependOn::kShape, DependOn::kNone, DependOn::kValue})
  .SetShapeFuncWith<LinSpace>();
REG_SYMBOL_OP_BUILDER("LinSpaceExt")
  .SetShapeDepend({DependOn::kShape, DependOn::kNone, DependOn::kValue})
  .SetShapeFuncWith<LinSpace>();
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
