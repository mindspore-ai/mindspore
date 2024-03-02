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
#include "mindspore/core/ops/symbol_ops_impl/scalar_add.h"

namespace mindspore {
namespace symshape {
namespace ops {
class MS_CORE_API PadV3 : public InferShapeOp {
 public:
  using InferShapeOp::InferShapeOp;
  PadV3(const SymbolPtr &input, const SymbolPtr &padding, const SymbolPtr &contiguous)
      : InferShapeOp({input, padding, contiguous}) {}
  ~PadV3() override = default;
  MS_DECLARE_PARENT(PadV3, InferShapeOp)
 protected:
  SymbolPtr Eval() override;
};

SymbolPtr PadV3::Eval() {
  auto input = input_as<ListSymbol>(kIndex0);
  auto padding = input_as<ListSymbol>(kIndex1);
  auto contiguous = input_as<BoolSymbol>(kIndex2)->value();
  const size_t kNum2 = 2;
  if (!input->HasData() || !padding->HasData()) {
    return GenVList();
  }
  if (padding->size() % kNum2 != 0 || padding->size() > input->size() * kNum2) {
    MS_LOG(INFO) << "For 'PadV3', the padding size should be even number and less-equal to " << (input->size() * kNum2)
                 << ", but got " << padding->size();
    return nullptr;
  }
  DoNotEvalOnRun();

  // when input shape is (A, B, C), contiguous=true
  // paddings: (p0, p1)                 -- pads dim C
  // paddings: (p0, p1, p2, p3)         -- the (p2,p3) pads dim B, the (p0,p1) pads dim C.
  // paddings: (p0, p1, p2, p3, p4, p5) -- the (p4,p5) pads dim A, the (p2,p3) pads dim B, the (p0,p1) pads dim C.
  SymbolPtrList result = input->symbols();
  auto result_iter = result.rbegin();
  for (size_t i = 0; i < input->size(); i++, ++result_iter) {
    size_t begin_i;
    size_t end_i;
    if (contiguous) {
      // the padding is [begin_0, end_0, begin_1, end_1, ..., begin_n, end_n]
      begin_i = i * kNum2;
      end_i = begin_i + 1;
    } else {
      // the padding is [begin_0, begin_1, ..., begin_n, end_0, end_1, ..., end_n]
      begin_i = i;
      end_i = i + padding->size() / kNum2;
    }
    if (end_i >= padding->size()) {
      break;
    }
    auto p = Emit(std::make_shared<ScalarAdd>(padding->symbols()[begin_i], padding->symbols()[end_i]));
    *result_iter = Emit(std::make_shared<ScalarAdd>(*result_iter, p));
  }
  return ResultIntList(std::move(result));
}

REG_SYMBOL_OP_BUILDER("PadV3")
  .SetShapeDepend({DependOn::kShape, DependOn::kValue})
  .SetShapeFunc([](OperationBuilder *b) -> SymbolPtr {
    auto input = b->GetInputShape(kIndex0);
    auto padding = b->GetInputValue(kIndex1);
    auto contiguous = b->GetAttr("paddings_contiguous");
    if (contiguous == nullptr) {
      contiguous = BoolSymbol::Make(true);
    }
    return b->Emit(std::make_shared<PadV3>(input, padding, contiguous));
  });
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
