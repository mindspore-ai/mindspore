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
class MS_CORE_API CTCLoss : public InferShapeOp {
 public:
  using InferShapeOp::InferShapeOp;
  ~CTCLoss() override = default;
  MS_DECLARE_PARENT(CTCLoss, InferShapeOp)

 protected:
  SymbolPtr Eval() override {
    auto inputs_shape = input_as<ListSymbol>(kIndex0);
    if (!inputs_shape->HasData()) {
      return GenList({GenVList(), GenVList()});
    }
    DoNotEvalOnRun();
    auto batch = inputs_shape->item(kIndex1);
    auto loss_shape = GenList({batch});
    return GenList({loss_shape, input(kIndex0)});
  }
};

REG_SYMBOL_OP_BUILDER("CTCLoss").SetShapeDepend({DependOn::kShape}).SetShapeFunc(DefaultBuilder<CTCLoss>);
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
