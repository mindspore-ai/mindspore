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
#include "mindspore/core/ops/symbol_ops_impl/scalar_div.h"

namespace mindspore {
namespace symshape {
namespace ops {
class MS_CORE_API Reshape : public InferShapeOp {
 public:
  using InferShapeOp::InferShapeOp;
  Reshape(const SymbolPtr &input, const SymbolPtr &shape) : InferShapeOp({input, shape}) {}
  ~Reshape() override = default;
  MS_DECLARE_PARENT(Reshape, InferShapeOp)

 protected:
  SymbolPtr Eval() override;
  void EvalOnRun() override;
  bool ProductShape(const ListSymbol *shape);
  std::pair<SymbolPtr, int64_t> ProductData(const ListSymbol *data);
  void UpdateMathInfo() override;

  int64_t shape_size_{1LL};
  int unknown_dim_idx_{-1};
  bool shape_all_have_data_on_building_{false};
  OpPtrList inner_ops_;
};

bool Reshape::ProductShape(const ListSymbol *shape) {
  shape_size_ = -1;
  for (size_t i = 0; i < shape->size(); i++) {
    auto s = AsInt(shape->symbols()[i]);
    if (s < 0) {
      unknown_dim_idx_ = static_cast<int>(i);
    }
    shape_size_ *= s;
  }
  // means no "-1" in "shape".
  return shape_size_ < 0;
}

std::pair<SymbolPtr, int64_t> Reshape::ProductData(const ListSymbol *data) {
  int64_t input_const_dims = 1LL;
  SymbolPtr input_unknown_dims = nullptr;
  for (auto &s : data->symbols()) {
    if (s->HasData()) {
      input_const_dims *= AsInt(s);
    } else {
      if (input_unknown_dims == nullptr) {
        input_unknown_dims = s;
      } else {
        input_unknown_dims = Emit(std::make_shared<ScalarMul>(input_unknown_dims, s));
      }
    }
  }
  return std::make_pair(input_unknown_dims, input_const_dims);
}

SymbolPtr Reshape::Eval() {
  // only eval on Building
  auto data = input_as<ListSymbol>(0);
  auto shape = input_as<ListSymbol>(1);
  if (shape->AllHaveData()) {
    if (ProductShape(shape)) {
      // no -1 in "shape", the output is static shape.
      DoNotEvalOnRun();
      return input(1);
    }
  } else {  // "shape" is unknown, maybe from previous shape-calculation operators.
    if (shape->is_dyn_len()) {
      return GenVList();
    } else {
      // if the symbol in "shape" is positive number, it's unnecessary to create a new symbol.
      SymbolPtrList result(shape->size());
      bool has_new_symbol = false;
      (void)std::transform(shape->symbols().cbegin(), shape->symbols().cend(), result.begin(),
                           [this, &has_new_symbol](const SymbolPtr &s) {
                             auto ints = s->as<IntSymbol>();
                             MS_EXCEPTION_IF_NULL(ints);
                             if (ints->is_positive()) {
                               return s;
                             }
                             has_new_symbol = true;
                             return this->GenVInt();
                           });
      if (!has_new_symbol) {
        DoNotEvalOnRun();
      }
      return GenList(std::move(result));
    }
  }
  // "shape" is constant tensor and "-1" exists in it.
  SymbolPtrList result = shape->symbols();
  if (data->is_dyn_len()) {
    result[unknown_dim_idx_] = GenVInt();
    return GenList(std::move(result));
  }
  // do not add the inner operation to global op list.
  OperationEmitter e(&inner_ops_);
  SetEmitter(&e);
  SymbolPtr input_unknown_dims;
  int64_t input_const_dims;
  std::tie(input_unknown_dims, input_const_dims) = ProductData(data);
  // no symbol in data
  if (input_unknown_dims == nullptr) {
    result[unknown_dim_idx_] = GenInt(input_const_dims / shape_size_);
    return GenList(std::move(result));
  }
  // Reshape (const1, s1) to (const2, s2), the `s2 = const1 * s1 / const2`
  // if `const1 % const2 == 0`, then simplify to `s2 = (const1/const2) * s1`
  if (input_const_dims % shape_size_ == 0) {
    // s2 = s1 * (const1 / const2)
    auto c = GenInt(input_const_dims / shape_size_);
    result[unknown_dim_idx_] = Emit(std::make_shared<ScalarMul>(input_unknown_dims, c));
  } else {
    // s2 = (s1 * const1) / const2
    auto tmp = Emit(std::make_shared<ScalarMul>(input_unknown_dims, GenInt(input_const_dims)));
    result[unknown_dim_idx_] = Emit(std::make_shared<ScalarDiv>(tmp, GenInt(shape_size_)));
  }
  return GenList(std::move(result));
}

/**
 * Reshape "(const1, s1)"" to "(const2, s2)", if both "s1" and "s2" are from input operation,
 * it's sure that "s2 == s1 * (const1 / const2)"
 *
 * Only support the case that one unknown symbol in "shape" but it's positive, and others are const value.
 * example:
 *   data.shape = (A, 1024)
 *   shape = (32, B, 1024)
 *  when B > 0, the output shape is directly set to "(32, B, 1024)".
 *  but we can also set "B == A / 32" in MathInfo.
 */
void Reshape::UpdateMathInfo() {
  InferShapeOp::UpdateMathInfo();
  if (need_eval()) {
    return;
  }
  auto data = input_as<ListSymbol>(0);
  auto shape = input_as<ListSymbol>(1);
  if (data->is_dyn_len()) {
    return;
  }
  IntSymbolPtr s2 = nullptr;
  int64_t const2 = 1LL;
  for (auto &shape_v : shape->symbols()) {
    if (shape_v->HasData()) {
      const2 *= AsInt(shape_v);
    } else {
      if (s2 != nullptr) {
        return;
      }
      s2 = shape_v->as_sptr<IntSymbol>();
    }
  }
  if (s2 == nullptr) {
    return;
  }
  // do not add the inner operation to global op list.
  OperationEmitter e(&inner_ops_);
  SetEmitter(&e);
  SymbolPtr s1;
  int64_t const1;
  std::tie(s1, const1) = ProductData(data);
  if (s1 == nullptr) {
    return;
  }
  auto tmp = Emit(std::make_shared<ScalarDiv>(Emit(std::make_shared<ScalarMul>(s1, GenInt(const1))), GenInt(const2)));
  MS_LOG(DEBUG) << "In Reshape, the " << tmp->ToString() << " and " << s2->ToString() << " can be set equal.";
  s2->SetEqual(tmp->as_sptr<IntSymbol>());
}

void Reshape::EvalOnRun() {
  auto shape = input_as<ListSymbol>(1);
  if (!shape_all_have_data_on_building_ && ProductShape(shape)) {
    // no "-1" in "shape"
    output_->Update(input(1));
    return;
  }
  auto data = input_as<ListSymbol>(0);
  int64_t data_size = 1;
  for (auto &s : data->symbols()) {
    data_size *= AsInt(s);
  }
  // the size of "-1"
  auto size_of_unknown_dim = GenInt(data_size / shape_size_);
  if (shape_all_have_data_on_building_) {
    output_as<ListSymbol>()->item(unknown_dim_idx_)->Update(size_of_unknown_dim);
  } else {
    auto result = shape->symbols();
    result[unknown_dim_idx_] = size_of_unknown_dim;
    output_as<ListSymbol>()->UpdateList(std::move(result));
  }
}

REG_SYMBOL_OP_BUILDER("Reshape")
  .SetShapeDepend({DependOn::kShape, DependOn::kValue})
  .SetShapeFunc(DefaultBuilder<Reshape>);
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
