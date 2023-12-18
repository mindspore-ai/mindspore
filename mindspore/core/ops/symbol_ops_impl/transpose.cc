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
#include "mindspore/core/ops/symbol_ops_impl/common.h"

namespace mindspore {
namespace symshape {
namespace ops {
class MS_CORE_API Transpose : public InferShapeOp {
 public:
  using InferShapeOp::InferShapeOp;
  Transpose(const SymbolPtr &data, const SymbolPtr &perm) : InferShapeOp({data, perm}) {}
  ~Transpose() override = default;
  MS_DECLARE_PARENT(Transpose, InferShapeOp)

 protected:
  SymbolPtr Eval() override;
  void EvalOnRun() override;

  inline SymbolPtrList GenResult(const ListSymbol *inp, const ListSymbol *perm) const {
    MS_EXCEPTION_IF_CHECK_FAIL(inp->size() == perm->size(), "size of input and perm should be equal.");
    SymbolPtrList result(inp->size());
    for (size_t i = 0; i < result.size(); i++) {
      result[i] = inp->symbols()[LongToSize(NormAxis(AsInt(perm->item(i)), result.size()))];
    }
    return result;
  }
};

SymbolPtr Transpose::Eval() {
  auto inp = input_as<ListSymbol>(0);
  auto perm = input_as<ListSymbol>(1);
  if (inp->HasData() && perm->AllHaveData()) {
    DoNotEvalOnRun();
    return GenList(GenResult(inp, perm));
  }
  // dynamic rank
  if (!inp->HasData() && !perm->AllHaveData()) {
    return GenVList();
  }
  size_t rank = inp->HasData() ? inp->size() : perm->size();
  return GenVIntList(rank);
}

void Transpose::EvalOnRun() {
  auto inp = input_as<ListSymbol>(0);
  auto perm = input_as<ListSymbol>(1);
  output_as<ListSymbol>()->UpdateList(GenResult(inp, perm));
}

REG_SYMBOL_OP_BUILDER("Transpose")
  .SetShapeDepend({DependOn::kShape, DependOn::kValue})
  .SetShapeFunc(DefaultBuilder<Transpose>);
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
