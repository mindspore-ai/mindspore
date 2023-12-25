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
#include "include/api/format.h"
#include "mindspore/core/ops/symbol_ops_impl/common.h"

namespace mindspore {
namespace symshape {
namespace ops {
class MS_CORE_API BiasAddGrad : public InferShapeOp {
 public:
  using InferShapeOp::InferShapeOp;
  BiasAddGrad(const SymbolPtr &x, const SymbolPtr &fmt) : InferShapeOp({x, fmt}) {}
  ~BiasAddGrad() override = default;
  MS_DECLARE_PARENT(BiasAddGrad, InferShapeOp)
 protected:
  SymbolPtr Eval() override;
};

SymbolPtr BiasAddGrad::Eval() {
  auto x = input_as<ListSymbol>(0);
  if (!x->HasData()) {
    return GenVIntList(1);
  }
  DoNotEvalOnRun();
  Format fmt = static_cast<Format>(input_as<IntSymbol>(1)->value());
  MS_EXCEPTION_IF_CHECK_FAIL(x->size() >= 2, "input rank of BiasAddGrad should be >= 2. symbol x: " + x->ToString());
  // return the axis length of "C".
  return GenList({fmt == Format::NCHW ? x->symbols()[1] : x->symbols().back()});
}

REG_SYMBOL_OP_BUILDER("BiasAddGrad")
  .SetShapeDepend({DependOn::kShape, DependOn::kValue})
  .SetShapeFunc([](OperationBuilder *b) -> SymbolPtr {
    auto x = b->GetInputShape(kIndex0);
    auto fmt = b->GetInputOrAttr(kIndex1, "format");  // todo, change to DefaultBuilder
    if (fmt == nullptr) {
      fmt = IntSymbol::Make(0);
    } else if (fmt->is<StrSymbol>()) {
      auto fmt_str = fmt->as<StrSymbol>()->value();
      fmt = IntSymbol::Make(fmt_str == "NCHW" ? 0 : 1);
    }
    return b->Emit(std::make_shared<BiasAddGrad>(x, fmt));
  });
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
