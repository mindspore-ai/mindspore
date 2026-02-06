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
#include "mindspore/ops/infer/symbol_ops_impl/common.h"
#include "utils/check_convert_utils.h"
#include "include/op_enum.h"

namespace mindspore {
namespace symshape {
namespace ops {
class OPS_API Meshgrid : public InferShapeOp {
 public:
  using InferShapeOp::InferShapeOp;
  ~Meshgrid() override = default;
  MS_DECLARE_PARENT(Meshgrid, InferShapeOp)
 protected:
  SymbolPtr Eval() override;
};

SymbolPtr Meshgrid::Eval() {
  auto &inputs = input_as<ListSymbol>(kIndex0)->symbols();
  SymbolPtrList out;
  out.reserve(inputs.size());
  (void)CheckAndConvertUtils::CheckInteger("number of input tensors", SizeToLong(inputs.size()), kGreaterThan, 1,
                                           "Meshgrid");
  for (auto &inp : inputs) {
    if (!inp->HasData()) {
      (void)out.emplace_back(GenVInt());
    } else {
      (void)out.emplace_back(inp->as<ListSymbol>()->item(0));
    }
  }
  auto indexing = input_as<IntSymbol>(kIndex1);
  if (indexing->HasData()) {
    if (indexing->value() == mindspore::ops::Indexing::XY) {
      std::swap(out[0], out[1]);
    }
  } else if (!out[0]->EqualsTo(out[1])) {
    out[0] = GenVInt();
    out[1] = GenVInt();
  }
  // when the input num is N, the outputs are N same shape.
  return ListSymbol::Make(SymbolPtrList(out.size(), ListSymbol::Make(std::move(out))));
}

REG_SYMBOL_OP_BUILDER("Meshgrid")
  .SetShapeDepend([](const PrimitivePtr &, size_t input_num) {
    std::vector<DependOn> depends(input_num, DependOn::kShape);
    depends.back() = DependOn::kValue;
    return depends;
  })
  .SetShapeFunc([](OperationBuilder *b) -> SymbolPtr {
    auto inputs = b->GetSymbolsOfDepend();
    if (!CheckAndConvertUtils::IsTensor(b->GetInput(kIndex0))) {
      if (!inputs[0]->HasData()) {
        // dynamic sequence is not supported
        return nullptr;
      }
      return b->Emit(std::make_shared<Meshgrid>(std::move(inputs)));
    } else {
      SymbolPtr data = ListSymbol::Make(SymbolPtrList(inputs.begin(), inputs.end() - 1));
      return b->Emit(std::make_shared<Meshgrid>(SymbolPtrList{data, inputs.back()}));
    }
  });
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
