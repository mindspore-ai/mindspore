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
#include "mindspore/core/ops/symbol_ops_impl/scalar_div.h"
#include "mindspore/core/ops/symbol_ops_impl/scalar_min.h"

namespace mindspore {
namespace symshape {
namespace ops {
class MS_CORE_API Range : public InferShapeOp {
 public:
  using InferShapeOp::InferShapeOp;
  ~Range() override = default;
  MS_DECLARE_PARENT(Range, InferShapeOp)
 protected:
  SymbolPtr Eval() override;
};

SymbolPtr Range::Eval() {
  auto start = input(kIndex0);
  auto end = input(kIndex1);
  auto step = input(kIndex2);
  auto maxlen = input(kIndex3);
  DoNotEvalOnRun();
  // range length = (end - start) / step.  (to ceil)
  auto len = Emit(std::make_shared<ScalarSub>(end, start));
  len = Emit(std::make_shared<ScalarCeilDiv>(len, step));
  return Emit(std::make_shared<ScalarMin>(len, maxlen));
}

REG_SYMBOL_OP_BUILDER("Range").SetShapeDependN<DependOn::kValue, 4>().SetShapeFuncWith<Range>();
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
