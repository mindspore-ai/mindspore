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
#include "mindspore/core/ops/symbol_ops_impl/elemwise_op.h"

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
  // for MatMulExt, the inputs a and b support 1D tensor.
  // when a is 2D shape (m,k1), and b is 1D shape (k2,). the output shape is (m,)
  // when a is 1D shape (k1,), and b is 2D shape (k2,n). the output shape is (n,)
  // when both a and b are 1D shape, the output is a scalar.
  DoNotEvalOnRun();
  SymbolPtrList result;
  result.reserve(std::max(a->size(), b->size()) + kMatMulRank);
  if (has_batch_) {
    result = ElemwiseBinop::Process(a->symbols(), b->symbols(), emitter(), kMatMulRank);
  }
  // shape of a is "m * k1" (not transposed) or "k1 * n" (transposed)
  SymbolPtr k1;
  if (a->size() >= kMatMulRank) {
    auto m = *(++a->symbols().rbegin());
    k1 = a->symbols().back();
    if (trans_a) {
      std::swap(m, k1);
    }
    (void)result.emplace_back(std::move(m));
  } else {
    k1 = a->item(kIndex0);
  }
  // shape of b is "k2 * n" (not transposed) or "n * k2" (transposed)
  SymbolPtr k2;
  if (b->size() >= kMatMulRank) {
    k2 = *(++b->symbols().rbegin());
    auto n = b->symbols().back();
    if (trans_b) {
      std::swap(n, k2);
    }
    (void)result.emplace_back(std::move(n));
  } else {
    k2 = b->item(kIndex0);
  }
  // k1 should be equal to k2
  if (is_building()) {
    k1->as<IntSymbol>()->SetEqual(k2->as_sptr<IntSymbol>());
  }
  return ResultIntList(std::move(result));
}

template <bool HAS_BATCH>
SymbolPtr MatMulShapeBuilder(OperationBuilder *b) {
  auto x = b->GetInputShape(kIndex0);
  auto y = b->GetInputShape(kIndex1);
  auto trans_a = b->GetInputOrAttr(kIndex2, "transpose_a");
  auto trans_b = b->GetInputOrAttr(kIndex3, "transpose_b");
  return b->Emit(std::make_shared<MatMul>(HAS_BATCH, x, y, trans_a, trans_b));
}

REG_SYMBOL_OP_BUILDER("MatMul")
  .SetShapeDepend({DependOn::kShape, DependOn::kShape, DependOn::kValue, DependOn::kValue})
  .SetShapeFunc(MatMulShapeBuilder<false>);
REG_SYMBOL_OP_BUILDER("BatchMatMul")
  .SetShapeDepend({DependOn::kShape, DependOn::kShape, DependOn::kValue, DependOn::kValue})
  .SetShapeFunc(MatMulShapeBuilder<true>);

SymbolPtr MatMulExtShapeBuilder(OperationBuilder *b) {
  auto x = b->GetInputShape(kIndex0);
  auto y = b->GetInputShape(kIndex1);
  return b->Emit(std::make_shared<MatMul>(true, x, y, BoolSymbol::Make(false), BoolSymbol::Make(false)));
}
REG_SYMBOL_OP_BUILDER("MatMulExt")
  .SetShapeDepend({DependOn::kShape, DependOn::kShape})
  .SetShapeFunc(MatMulExtShapeBuilder);
REG_SYMBOL_OP_BUILDER("BatchMatMulExt")
  .SetShapeDepend({DependOn::kShape, DependOn::kShape})
  .SetShapeFunc(MatMulExtShapeBuilder);
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
