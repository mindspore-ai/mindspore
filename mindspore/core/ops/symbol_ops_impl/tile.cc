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
#include "mindspore/core/ops/symbol_ops_impl/scalar_mul.h"

namespace mindspore {
namespace symshape {
namespace ops {
class MS_CORE_API Tile : public InferShapeOp {
 public:
  using InferShapeOp::InferShapeOp;
  Tile(const SymbolPtr &data, const SymbolPtr &multiples) : InferShapeOp({data, multiples}) {}
  ~Tile() override = default;
  MS_DECLARE_PARENT(Tile, InferShapeOp)

 protected:
  SymbolPtr Eval() override;
};

SymbolPtr Tile::Eval() {
  auto input = input_as<ListSymbol>(kIndex0);
  auto multiples = input_as<ListSymbol>(kIndex1);
  if (!multiples->HasData()) {
    return GenVList();
  }
  size_t n = multiples->size();
  if (!input->HasData()) {
    return GenVIntList(n);
  }
  if (input->size() > n) {
    MS_LOG(INTERNAL_EXCEPTION) << "The input's shape size should not be greater than the multiples's size. input:"
                               << input->ToString() << ", multiples:" << multiples->ToString();
  }
  DoNotEvalOnRun();
  SymbolPtrList result(n);
  // if input size less than multiples size, leading "1"s are expanded to the input.
  for (size_t i = n; i > 0; i--) {
    if (i <= input->size()) {
      result[n - i] = Emit(std::make_shared<ScalarMul>(input->item(input->size() - i), multiples->item(n - i)));
    } else {
      result[n - i] = multiples->item(n - i);
    }
  }
  return GenList(std::move(result));
}

REG_SYMBOL_OP_BUILDER("Tile").SetShapeDepend({DependOn::kShape, DependOn::kValue}).SetShapeFunc(DefaultBuilder<Tile>);
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
