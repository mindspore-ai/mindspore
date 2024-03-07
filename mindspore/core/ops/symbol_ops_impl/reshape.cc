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
  std::string DumpText() const override;

 protected:
  SymbolPtr Eval() override;
  void EvalOnRun() override;
  void ProductShape(const ListSymbol *shape);
  std::pair<SymbolPtr, int64_t> ProductData(const ListSymbol *data);

  int unknown_dim_idx_{-1};
  SymbolPtr shape_unknown_dims_{nullptr};
  int64_t shape_const_dims_{1};
  OpPtrList inner_ops_;
};

std::string Reshape::DumpText() const {
  std::ostringstream oss;
  oss << InferShapeOp::DumpText();
  for (auto &inner_op : inner_ops_) {
    oss << "  " << inner_op->DumpText();
  }
  return oss.str();
}

/// \brief calculate the product value of shape
void Reshape::ProductShape(const ListSymbol *shape) {
  shape_unknown_dims_ = nullptr;
  shape_const_dims_ = 1LL;
  unknown_dim_idx_ = -1;
  size_t notpositive_symbol_cnt = 0;
  int notpositive_symbol_idx = 0;
  for (size_t i = 0; i < shape->size(); i++) {
    auto item = shape->symbols()[i]->as<IntSymbol>();
    MS_EXCEPTION_IF_NULL(item);
    if (item->HasData()) {
      auto s = item->value();
      if (s < 0) {
        unknown_dim_idx_ = static_cast<int>(i);
      } else {
        shape_const_dims_ *= s;
      }
    } else {
      if (!item->is_positive()) {
        notpositive_symbol_cnt++;
        notpositive_symbol_idx = static_cast<int>(i);
      }
    }
  }

  if (notpositive_symbol_cnt > 0) {
    if (unknown_dim_idx_ >= 0) {
      // there's a "-1" in shape, so other symbols are all positive.
      for (size_t i = 0; i < shape->size(); i++) {
        auto item = shape->item_as_sptr<IntSymbol>(i);
        if (static_cast<int>(i) != unknown_dim_idx_ && !item->is_positive()) {
          item->SetRangeMin(1);
        }
      }
    } else if (notpositive_symbol_cnt == 1) {
      // there's no "-1" in shape, but is only one symbol that is not always positive, it can be treated as "-1".
      unknown_dim_idx_ = notpositive_symbol_idx;
    }
  }
  // product the unknown dims, except the "-1" item.
  for (size_t i = 0; i < shape->size(); i++) {
    auto item = shape->item_as_sptr<IntSymbol>(i);
    if (static_cast<int>(i) == unknown_dim_idx_ || item->HasData()) {
      continue;
    }
    if (shape_unknown_dims_ == nullptr) {
      shape_unknown_dims_ = item;
    } else {
      shape_unknown_dims_ = Emit(std::make_shared<ScalarMul>(shape_unknown_dims_, item));
    }
  }
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
  if (!shape->HasData()) {
    return GenVList();
  }
  if (std::all_of(shape->symbols().begin(), shape->symbols().end(),
                  [](const SymbolPtr &s) { return s->as<IntSymbol>()->is_positive(); })) {
    // all items of "shape" are positive,
    DoNotEvalOnRun();
    return input(1);
  }
  // do not add the inner operation to global op list.
  OperationEmitter e(&inner_ops_);
  SetEmitter(&e);
  ProductShape(shape);
  SymbolPtrList result = shape->symbols();
  if (unknown_dim_idx_ < 0) {
    for (size_t i = 0; i < result.size(); i++) {
      if (!result[i]->as<IntSymbol>()->is_positive()) {
        result[i] = GenVInt();
      }
    }
    return GenList(std::move(result));
  }

  // "-1" exists in shape.
  if (data->is_dyn_len()) {
    result[unknown_dim_idx_] = GenVInt();
    return GenList(std::move(result));
  }
  SymbolPtr input_unknown_dims;
  int64_t input_const_dims;
  std::tie(input_unknown_dims, input_const_dims) = ProductData(data);
  if (input_unknown_dims == nullptr && shape_unknown_dims_ == nullptr) {
    result[unknown_dim_idx_] = GenInt(input_const_dims / shape_const_dims_);
    return GenList(std::move(result));
  }

  // Reshape (const1, s1) to (const2, s2, U), the `U = (const1 * s1) / (const2 * s2)`
  // if `const1 % const2 == 0`, then simplify to `U = (const1/const2) * s1 / s2`
  if (input_const_dims % shape_const_dims_ == 0) {
    auto c = GenInt(input_const_dims / shape_const_dims_);
    auto c_s1 = input_unknown_dims != nullptr ? Emit(std::make_shared<ScalarMul>(input_unknown_dims, c)) : c;
    auto u = shape_unknown_dims_ != nullptr ? Emit(std::make_shared<ScalarDiv>(c_s1, shape_unknown_dims_)) : c_s1;
    result[unknown_dim_idx_] = u;
  } else {
    auto tmp1 = input_unknown_dims != nullptr
                  ? Emit(std::make_shared<ScalarMul>(input_unknown_dims, GenInt(input_const_dims)))
                  : GenInt(input_const_dims);
    auto tmp2 = shape_unknown_dims_ != nullptr
                  ? Emit(std::make_shared<ScalarMul>(shape_unknown_dims_, GenInt(shape_const_dims_)))
                  : GenInt(shape_const_dims_);
    result[unknown_dim_idx_] = Emit(std::make_shared<ScalarDiv>(tmp1, tmp2));
  }
  return GenList(std::move(result));
}

void Reshape::EvalOnRun() {
  auto shape = input_as<ListSymbol>(1);
  ProductShape(shape);
  if (unknown_dim_idx_ < 0) {
    // no "-1" in "shape"
    output_->Update(input(1));
    return;
  }
  int64_t data_size;
  std::tie(std::ignore, data_size) = ProductData(input_as<ListSymbol>(kIndex0));
  // the size of "-1"
  auto result = shape->symbols();
  result[unknown_dim_idx_] = GenInt(data_size / shape_const_dims_);
  output_as<ListSymbol>()->UpdateList(std::move(result));
}

REG_SYMBOL_OP_BUILDER("Reshape")
  .SetShapeDepend({DependOn::kShape, DependOn::kValue})
  .SetShapeFunc(DefaultBuilder<Reshape>);
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
