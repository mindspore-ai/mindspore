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
#include "backend/common/graph_kernel/symbol_engine/operations/infershape_op.h"
#include <algorithm>
#include <utility>
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "utils/shape_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/hash_set.h"
#include "backend/common/graph_kernel/symbol_engine/operations/common_op.h"
#include "backend/common/graph_kernel/symbol_engine/utils.h"

namespace mindspore::graphkernel::symbol {
namespace ops::infershape {
SymbolPtr RealShape::Eval() {
  auto base_shape_ptr = input_as<InputSymbol>(0)->abstract()->BuildShape();
  // todo, support tuple shape
  auto shape_ptr = base_shape_ptr->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(shape_ptr);
  return FromShape(shape_ptr->shape());
}

SymbolPtr BinElemwise::Eval() {
  auto lhs = input_as<IListSymbol>(0);
  auto rhs = input_as<IListSymbol>(1);
  if (!lhs->HasData() || !rhs->HasData()) {
    return GenVList();
  }
  // the following ScalarMax is added to global ops list.
  DoNotEvalOnRun();
  return GenList(BinElemwise::Process(lhs->symbols(), rhs->symbols(), emitter()));
}

SymbolPtrList BinElemwise::Process(const SymbolPtrList &lhs, const SymbolPtrList &rhs, const Emitter &e, size_t shift) {
  MS_EXCEPTION_IF_CHECK_FAIL(shift <= std::min(lhs.size(), rhs.size()),
                             "shift should not greater than the minimum size of lhs and rhs");
  SymbolPtrList result(std::max(lhs.size(), rhs.size()) - shift);
  int i = static_cast<int>(result.size()) - 1;
  auto a = lhs.rbegin() + shift;
  auto b = rhs.rbegin() + shift;
  for (; i >= 0; i--) {
    if (b == rhs.rend()) {
      result[i] = *a;
      ++a;
    } else if (a == lhs.rend()) {
      result[i] = *b;
      ++b;
    } else {
      // broadcast rules. assume the input shape is valid.
      // rule 1: s1 & s2 -> s3=max(s1, s2)
      // rule 2: s1 & 1  -> s1
      // rule 3: s1 & n  -> n  (n != 1)
      if (!(*a)->HasData() && !(*b)->HasData()) {
        result[i] = e.Emit(std::make_shared<ScalarMax>(*a, *b));
      } else if ((*a)->HasData() && AsInt(*a) == 1) {
        result[i] = *b;
      } else if ((*b)->HasData() && AsInt(*b) != 1) {
        result[i] = *b;
      } else {
        result[i] = *a;
      }
      ++a;
      ++b;
    }
  }
  return result;
}

bool Reduce::GetAxisSet(const SymbolPtr &axis, int64_t rank, bool skip_mode, HashSet<int64_t> *axis_set) const {
  auto axis_list = axis->as<IListSymbol>();
  if (axis_list != nullptr) {
    if (axis_list->symbols().empty()) {
      if (skip_mode) {
        return false;
      } else {
        for (int64_t i = 0; i < rank; i++) {
          (void)axis_set->insert(i);
        }
      }
    } else {
      for (auto &x : axis_list->symbols()) {
        (void)axis_set->insert(AsInt(x));
      }
    }
  } else {
    (void)axis_set->insert(AsInt(axis));
  }
  return true;
}

SymbolPtr Reduce::Eval() {
  auto data = input_as<ListSymbol>(kIndex0);
  auto axis = input(kIndex1);
  bool keep_dims = input_as<BoolSymbol>(kIndex2)->value();
  bool skip_mode = input_as<BoolSymbol>(kIndex3)->value();
  if (!data->HasData() || !axis->HasData()) {
    // for empty axis
    auto axis_list = axis->as<ListSymbol>();
    if (axis_list != nullptr && axis_list->HasData() && axis_list->size() == 0) {
      if (skip_mode) {
        DoNotEvalOnRun();
        return input(kIndex0);
      }
      // if not keepdims, and the axis is a reduce-all case, then output shape is empty.
      if (!keep_dims) {
        return GenList({});
      }
    }
    return GenVList();
  }
  DoNotEvalOnRun();
  HashSet<int64_t> axis_set;
  auto &inp = data->symbols();
  auto rank = SizeToLong(inp.size());
  if (!GetAxisSet(axis, rank, skip_mode, &axis_set)) {
    return input(kIndex0);
  }
  SymbolPtrList out_shape;
  out_shape.reserve(inp.size());
  for (int64_t i = 0; i < rank; i++) {
    if (axis_set.count(i) != 0 || axis_set.count(i - rank) != 0) {
      if (keep_dims) {
        (void)out_shape.emplace_back(GenInt(1));
      }
    } else {
      (void)out_shape.emplace_back(inp[i]);
    }
  }
  return GenList(out_shape);
}

bool Reshape::ProductShape(const IListSymbol *shape) {
  shape_size_ = -1;
  for (size_t i = 0; i < shape->size(); i++) {
    auto s = shape->item(i);
    if (s < 0) {
      unknown_dim_idx_ = static_cast<int>(i);
    }
    shape_size_ *= s;
  }
  // means no "-1" in "shape".
  return shape_size_ < 0;
}

SymbolPtr Reshape::Eval() {
  // only eval on Building
  auto data = input(0)->as<IListSymbol>();
  auto shape = input(1)->as<IListSymbol>();
  MS_EXCEPTION_IF_NULL(shape);
  if (shape->AllHaveData()) {
    if (ProductShape(shape)) {
      // no -1 in "shape", the output is static shape.
      DoNotEvalOnRun();
      return input(1);
    }
  } else {
    // "shape" is unknown
    if (shape->is_dyn_len()) {
      return GenVList();
    } else {
      return GenVIntList(shape->size());
    }
  }
  // "shape" is constant tensor and "-1" exists in it.
  SymbolPtrList result = shape->symbols();
  if (data->is_dyn_len()) {
    result[unknown_dim_idx_] = GenVInt();
    return GenList(std::move(result));
  }
  int64_t input_const_dims = 1LL;
  SymbolPtr input_unknown_dims = nullptr;

  // do not add the inner operation to global op list.
  OperationEmitter e(&inner_ops_);
  SetEmitter(&e);
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
  // no symbol in data
  if (input_unknown_dims == nullptr) {
    result[unknown_dim_idx_] = GenInt(input_const_dims / shape_size_);
    return GenList(std::move(result));
  }
  // Reshape (const1, s1) to (const2, s2), the `s2 = const1 * s1 / const2`
  // if (const1 % const2 == 0), then simplify to `s2 = (const1/const2) * s1`
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

void Reshape::EvalOnRun() {
  auto shape = input_as<IListSymbol>(1);
  if (!shape_all_have_data_on_building_ && ProductShape(shape)) {
    // no "-1" in "shape"
    output_->Update(input(1));
    return;
  }
  auto data = input_as<IListSymbol>(0);
  int64_t data_size = 1;
  for (auto &s : data->symbols()) {
    data_size *= AsInt(s);
  }
  // the size of "-1"
  auto size_of_unknown_dim = GenInt(data_size / shape_size_);
  if (shape_all_have_data_on_building_) {
    output_as<IListSymbol>()->symbols()[unknown_dim_idx_]->Update(size_of_unknown_dim);
  } else {
    auto result = shape->symbols();
    result[unknown_dim_idx_] = size_of_unknown_dim;
    output_as<IListSymbol>()->UpdateList(std::move(result));
  }
}

SymbolPtr Transpose::Eval() {
  auto inp = input_as<ListSymbol>(0);
  auto perm = input_as<IListSymbol>(1);
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
  auto perm = input_as<IListSymbol>(1);
  output_as<ListSymbol>()->UpdateList(GenResult(inp, perm));
}

SymbolPtr MatMul::Eval() {
  auto a = input_as<IListSymbol>(kIndex0);
  auto b = input_as<IListSymbol>(kIndex1);
  auto trans_a = input_as<BoolSymbol>(kIndex2)->value();
  auto trans_b = input_as<BoolSymbol>(kIndex3)->value();
  constexpr const size_t kMatMulRank = 2;
  // dynamic rank on building
  if (!a->HasData() || !b->HasData()) {
    if (has_batch_) {
      return GenVList();
    }
    if (a->HasData()) {
      auto m = trans_a ? a->symbols()[1] : a->symbols()[0];
      return GenList(SymbolPtrList{m, GenVInt()});
    }
    if (b->HasData()) {
      auto n = trans_b ? b->symbols()[0] : b->symbols()[1];
      return GenList(SymbolPtrList{GenVInt(), n});
    }
    return GenVIntList(kMatMulRank);
  }
  DoNotEvalOnRun();
  SymbolPtrList result;
  result.reserve(std::max(a->size(), b->size()) + kMatMulRank);
  if (has_batch_) {
    result = BinElemwise::Process(a->symbols(), b->symbols(), emitter(), kMatMulRank);
  }
  // shape of `a` is k*m(transposed) or m*k(not transposed)
  auto m = trans_a ? a->symbols().back() : *(++a->symbols().rbegin());
  // shape of `b` is n*k(transposed) or k*n(not transposed)
  auto n = trans_b ? *(++b->symbols().rbegin()) : b->symbols().back();
  (void)result.emplace_back(std::move(m));
  (void)result.emplace_back(std::move(n));
  if (is_building()) {
    return GenList(std::move(result));
  }
  output_as<ListSymbol>()->UpdateList(std::move(result));
  return nullptr;
}
}  // namespace ops::infershape
}  // namespace mindspore::graphkernel::symbol
