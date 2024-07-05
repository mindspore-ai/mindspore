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
#include "mindspore/core/ops/symbol_ops_impl/elemwise_op.h"
#include <algorithm>
#include <memory>
#include <utility>
#include "mindspore/core/ops/symbol_ops_impl/scalar_max.h"

namespace mindspore {
namespace symshape {
namespace ops {
SymbolPtr ElemwiseBinop::Eval() {
  auto lhs = input_as<ListSymbol>(0);
  auto rhs = input_as<ListSymbol>(1);
  if (!lhs->HasData() || !rhs->HasData()) {
    return GenVList();
  }
  // the following ScalarMax is added to global ops list.
  DoNotEvalOnRun();
  return GenList(ElemwiseBinop::Process(lhs->symbols(), rhs->symbols(), emitter()));
}

SymbolPtrList ElemwiseBinop::Process(const SymbolPtrList &lhs, const SymbolPtrList &rhs, const OperationEmitter &e,
                                     size_t shift) {
  SymbolPtrList result;
  size_t maxlen = std::max(lhs.size(), rhs.size());
  result.reserve(maxlen);
  for (size_t i = maxlen; i > shift; i--) {
    if (i > lhs.size()) {
      (void)result.emplace_back(rhs[rhs.size() - i]);
      continue;
    }
    if (i > rhs.size()) {
      (void)result.emplace_back(lhs[lhs.size() - i]);
      continue;
    }
    // broadcast rules. assume the input shape is valid.
    // rule 1: s1 & s2 -> s3=max(s1, s2)
    // rule 2: s1 & 1  -> s1
    // rule 3: s1 & n  -> n  (n != 1)
    auto a = lhs[lhs.size() - i]->as_sptr<IntSymbol>();
    auto b = rhs[rhs.size() - i]->as_sptr<IntSymbol>();
    if (a->EqualsTo(b)) {
      (void)result.emplace_back(std::move(a));
      continue;
    }
    if (!a->HasData() && !b->HasData()) {
      if (a->is_greater_than(1)) {
        (void)result.emplace_back(a);
        if (b->is_greater_than(1)) {
          MS_LOG(DEBUG) << "In element-wise operator, both " << a->ToString() << " and " << b->ToString()
                        << " are greater than 1, they must be equal.";
          a->SetEqual(b);
        }
      } else if (b->is_greater_than(1)) {
        (void)result.emplace_back(std::move(b));
      } else {
        (void)result.emplace_back(e.Emit(std::make_shared<ScalarMax>(a, b)));
      }
      continue;
    }
    if (a->HasData() && a->value() == 1) {
      (void)result.emplace_back(std::move(b));
    } else if (b->HasData() && b->value() != 1) {
      (void)result.emplace_back(std::move(b));
    } else {
      (void)result.emplace_back(std::move(a));
    }
  }
  return result;
}

// binary op
REG_SYMBOL_OP_BUILDER("Add").SetShapeDependN<DependOn::kShape, 2>().SetShapeFuncWith<ElemwiseBinop>();
REG_SYMBOL_OP_BUILDER("AddExt").SetShapeDependN<DependOn::kShape, 2>().SetShapeFuncWith<ElemwiseBinop>();
REG_SYMBOL_OP_BUILDER("Div").SetShapeDependN<DependOn::kShape, 2>().SetShapeFuncWith<ElemwiseBinop>();
REG_SYMBOL_OP_BUILDER("Equal").SetShapeDependN<DependOn::kShape, 2>().SetShapeFuncWith<ElemwiseBinop>();
REG_SYMBOL_OP_BUILDER("Greater").SetShapeDependN<DependOn::kShape, 2>().SetShapeFuncWith<ElemwiseBinop>();
REG_SYMBOL_OP_BUILDER("GreaterEqual").SetShapeDependN<DependOn::kShape, 2>().SetShapeFuncWith<ElemwiseBinop>();
REG_SYMBOL_OP_BUILDER("Less").SetShapeDependN<DependOn::kShape, 2>().SetShapeFuncWith<ElemwiseBinop>();
REG_SYMBOL_OP_BUILDER("LessEqual").SetShapeDependN<DependOn::kShape, 2>().SetShapeFuncWith<ElemwiseBinop>();
REG_SYMBOL_OP_BUILDER("LogicalAnd").SetShapeDependN<DependOn::kShape, 2>().SetShapeFuncWith<ElemwiseBinop>();
REG_SYMBOL_OP_BUILDER("LogicalOr").SetShapeDependN<DependOn::kShape, 2>().SetShapeFuncWith<ElemwiseBinop>();
REG_SYMBOL_OP_BUILDER("Maximum").SetShapeDependN<DependOn::kShape, 2>().SetShapeFuncWith<ElemwiseBinop>();
REG_SYMBOL_OP_BUILDER("Minimum").SetShapeDependN<DependOn::kShape, 2>().SetShapeFuncWith<ElemwiseBinop>();
REG_SYMBOL_OP_BUILDER("Mul").SetShapeDependN<DependOn::kShape, 2>().SetShapeFuncWith<ElemwiseBinop>();
REG_SYMBOL_OP_BUILDER("NotEqual").SetShapeDependN<DependOn::kShape, 2>().SetShapeFuncWith<ElemwiseBinop>();
REG_SYMBOL_OP_BUILDER("Pow").SetShapeDependN<DependOn::kShape, 2>().SetShapeFuncWith<ElemwiseBinop>();
REG_SYMBOL_OP_BUILDER("RealDiv").SetShapeDependN<DependOn::kShape, 2>().SetShapeFuncWith<ElemwiseBinop>();
REG_SYMBOL_OP_BUILDER("Sub").SetShapeDependN<DependOn::kShape, 2>().SetShapeFuncWith<ElemwiseBinop>();
REG_SYMBOL_OP_BUILDER("SubExt").SetShapeDependN<DependOn::kShape, 2>().SetShapeFuncWith<ElemwiseBinop>();

SymbolPtr AddnBuildShape(OperationBuilder *b, const SymbolPtrList &symbols) {
  if (symbols.empty()) {
    return nullptr;
  }
  auto result = symbols[0];
  for (size_t i = 1; i < symbols.size(); i++) {
    result = b->Emit(std::make_shared<ElemwiseBinop>(result, symbols[i]));
  }
  return result;
}
SymbolPtr GeneralBuildShape(OperationBuilder *b) { return AddnBuildShape(b, b->GetSymbolsOfDepend()); }

// ops of more inputs
REG_SYMBOL_OP_BUILDER("AddN").SetShapeDependN<DependOn::kShape>().SetShapeFunc([](OperationBuilder *b) {
  if (b->input_num() > kDim1) {
    return AddnBuildShape(b, b->GetSymbolsOfDepend());
  }
  auto inputs = b->GetInputShape(kIndex0)->as_sptr<ListSymbol>();
  return AddnBuildShape(b, inputs->symbols());
});
REG_SYMBOL_OP_BUILDER("Addcdiv").SetShapeDependN<DependOn::kShape, 4>().SetShapeFunc(GeneralBuildShape);
REG_SYMBOL_OP_BUILDER("Addcmul").SetShapeDependN<DependOn::kShape, 4>().SetShapeFunc(GeneralBuildShape);
REG_SYMBOL_OP_BUILDER("Select").SetShapeDependN<DependOn::kShape, 3>().SetShapeFunc(GeneralBuildShape);
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
