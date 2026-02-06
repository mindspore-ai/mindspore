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
#include "base/base.h"
#include "mindapi/base/macros.h"
#include "mindspore/ops/infer/symbol_ops_impl/common.h"
#include "abstract/symbolic_shape/symbol.h"
#include "utils/log_adapter.h"
#include "ops_utils/op_constants.h"

namespace mindspore {
namespace symshape {
namespace ops {
class OPS_API TupleGetItemShapeOp : public InferShapeOp {
 public:
  using InferShapeOp::InferShapeOp;
  TupleGetItemShapeOp(const SymbolPtr &data, const SymbolPtr &idx) : InferShapeOp({data, idx}) {}
  ~TupleGetItemShapeOp() override = default;
  MS_DECLARE_PARENT(TupleGetItemShapeOp, InferShapeOp)

 protected:
  SymbolPtr Eval() override {
    auto input = input_as<ListSymbol>(kIndex0);
    auto idx = input_as<IntSymbol>(kIndex1);
    if (!input->HasData() || !idx->HasData()) {
      return out_abstract()->GetShape()->BuildSymbolicShape();
    }
    DoNotEvalOnRun();
    return input->item(idx->value());
  }
};

class OPS_API TupleGetItemValueOp : public InferValueOp {
 public:
  using InferValueOp::InferValueOp;
  TupleGetItemValueOp(const SymbolPtr &data, const SymbolPtr &idx) : InferValueOp({data, idx}) {}
  ~TupleGetItemValueOp() override = default;
  MS_DECLARE_PARENT(TupleGetItemValueOp, InferValueOp)

 protected:
  SymbolPtr Eval() override {
    auto input = input_as<ListSymbol>(kIndex0);
    auto idx = input_as<IntSymbol>(kIndex1);
    if (!input->HasData() || !idx->HasData()) {
      return BuildSymbolicValue(out_abstract());
    }
    DoNotEvalOnRun();
    return input->item(idx->value());
  }
};

REG_SYMBOL_OP_BUILDER("TupleGetItem")
  .SetShapeDepend({DependOn::kShape, DependOn::kValue})
  .SetShapeFuncWith<TupleGetItemShapeOp>()
  .SetValueDepend({DependOn::kValue, DependOn::kValue})
  .SetValueFuncWith<TupleGetItemValueOp>();
REG_SYMBOL_OP_BUILDER("RealTupleGetItem")
  .SetShapeDepend({DependOn::kShape, DependOn::kValue})
  .SetShapeFuncWith<TupleGetItemShapeOp>()
  .SetValueDepend({DependOn::kValue, DependOn::kValue})
  .SetValueFuncWith<TupleGetItemValueOp>();
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
