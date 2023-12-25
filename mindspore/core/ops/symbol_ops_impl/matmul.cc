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
#include "mindspore/core/ops/symbol_ops_impl/elemwise_binop.h"

namespace mindspore {
namespace symshape {
namespace ops {
class MS_CORE_API MatMul : public InferShapeOp {
 public:
  MatMul(bool has_batch, const SymbolPtr &a, const SymbolPtr &b, const SymbolPtr &transpose_a,
         const SymbolPtr &transpose_b)
      : InferShapeOp({a, b, transpose_a, transpose_b}), has_batch_(has_batch) {}
  ~MatMul() override = default;
  MS_DECLARE_PARENT(MatMul, InferShapeOp)
  std::string name() const override { return has_batch_ ? "BatchMatMul" : "MatMul"; }

 protected:
  SymbolPtr Eval() override;
  bool has_batch_;
};

SymbolPtr MatMul::Eval() {
  auto a = input_as<ListSymbol>(kIndex0);
  auto b = input_as<ListSymbol>(kIndex1);
  auto trans_a = input_as<BoolSymbol>(kIndex2)->value();
  auto trans_b = input_as<BoolSymbol>(kIndex3)->value();
  constexpr const size_t kMatMulRank = 2;
  // dynamic rank on building
  if (!a->HasData() || !b->HasData()) {
    if (has_batch_) {
      return GenVList();
    }
    if (a->HasData()) {
      auto m = trans_a ? a->item(1) : a->item(0);
      return GenList(SymbolPtrList{m, GenVInt()});
    }
    if (b->HasData()) {
      auto n = trans_b ? b->item(0) : b->item(1);
      return GenList(SymbolPtrList{GenVInt(), n});
    }
    return GenVIntList(kMatMulRank);
  }
  DoNotEvalOnRun();
  SymbolPtrList result;
  result.reserve(std::max(a->size(), b->size()) + kMatMulRank);
  if (has_batch_) {
    result = ElemwiseBinop::Process(a->symbols(), b->symbols(), emitter(), kMatMulRank);
  }
  // shape of a is "m * k1" (not transposed) or "k1 * n" (transposed)
  auto m = *(++a->symbols().rbegin());
  auto k1 = a->symbols().back();
  if (trans_a) {
    std::swap(m, k1);
  }
  // shape of b is "k2 * n" (not transposed) or "n * k2" (transposed)
  auto k2 = *(++b->symbols().rbegin());
  auto n = b->symbols().back();
  if (trans_b) {
    std::swap(n, k2);
  }
  // k1 should be equal to k2
  if (!k1->HasData() && !k2->HasData()) {
    auto k = k1->as<IntSymbol>();
    MS_EXCEPTION_IF_NULL(k);
    k->SetEqual(k2->as_sptr<IntSymbol>());
  }
  (void)result.emplace_back(std::move(m));
  (void)result.emplace_back(std::move(n));
  return ResultIntList(std::move(result));
}

template <bool HAS_BATCH>
SymbolPtr MatMulShapeBuilder(OperationBuilder *b) {
  auto x = b->GetInputShape(kIndex0);
  auto y = b->GetInputShape(kIndex1);
  auto trans_a = b->GetAttr("transpose_a");
  if (trans_a == nullptr) {
    trans_a = BoolSymbol::Make(false);
  }
  auto trans_b = b->GetAttr("transpose_b");
  if (trans_b == nullptr) {
    trans_b = BoolSymbol::Make(false);
  }
  return b->Emit(std::make_shared<MatMul>(HAS_BATCH, x, y, trans_a, trans_b));
}

REG_SYMBOL_OP_BUILDER("MatMul").SetShapeFunc(MatMulShapeBuilder<false>);
REG_SYMBOL_OP_BUILDER("BatchMatMul").SetShapeFunc(MatMulShapeBuilder<true>);
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
