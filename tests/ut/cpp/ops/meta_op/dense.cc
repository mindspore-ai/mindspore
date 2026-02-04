/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "tests/ut/cpp/ops/meta_op/dense.h"
#include "ops/op_def_utils.h"
#include "utils/check_convert_utils.h"
#include "primitive/array_ops.h"
#include "mindspore/ops/ops_utils/op_constants.h"
#include "primitive/sequence_ops.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"
#include "primitive/auto_generate/gen_ops_primitive_t.h"
#include "primitive/auto_generate/gen_ops_primitive_c.h"
#include "primitive/auto_generate/gen_ops_primitive_s.h"
#include "primitive/auto_generate/gen_ops_primitive_r.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"

namespace mindspore::prim {
namespace {
void CheckDenseType(const PrimitivePtr &primitive, const AbstractBasePtrList &input_args) {
  constexpr auto kInputNum = 3;
  auto input_len = SizeToLong(input_args.size());
  MS_CHECK_VALUE(
    SizeToLong(input_args.size()) == kInputNum,
    CheckAndConvertUtils::FormatCheckIntegerMsg("input_args number", input_len, kEqual, kInputNum, primitive));
  const auto &x_shp = input_args[kIndex0]->GetShape()->GetShapeVector();
  const auto &w_shp = input_args[kIndex1]->GetShape()->GetShapeVector();
  if (IsDynamicRank(x_shp) || IsDynamicRank(w_shp)) {
    return;
  }

  if (w_shp.size() == 1) {
    const auto kDimW = " if the dim of w is 1.";
    if (x_shp.size() < 1) {
      MS_EXCEPTION(ValueError) << "The dim of x should be at least 1" << kDimW;
    }
    if (x_shp[x_shp.size() - 1] != w_shp[0]) {
      MS_EXCEPTION(ValueError) << "The value of x.shape[-1] should be equal to w.shape[0]" << kDimW;
    }
    if (!input_args[kIndex2]->GetType()->isa<TypeNone>()) {
      const auto &b_shp = input_args[kIndex2]->GetShape()->GetShapeVector();
      if (b_shp.size() != 0) {
        MS_EXCEPTION(ValueError) << "The dim of b should be equal to 0" << kDimW;
      }
    }
    return;
  }

  const auto kDimW = " if the dim of w is 2.";
  if (w_shp.size() != 2) {
    MS_EXCEPTION(ValueError) << "The dim of w should be equal to 1 or 2.";
  }
  if (x_shp.size() < 1) {
    MS_EXCEPTION(ValueError) << "The dim of x should be at least 1" << kDimW;
  }
  if (!input_args[kIndex2]->GetType()->isa<TypeNone>()) {
    const auto &b_shp = input_args[kIndex2]->GetShape()->GetShapeVector();
    if (b_shp.size() != 0 && b_shp.size() != 1) {
      MS_EXCEPTION(ValueError) << "The dim of b should be equal to 0 or 1" << kDimW;
    }
  }

  auto x_col = x_shp[x_shp.size() - kDim1];
  auto w_row = w_shp[kDim1];
  if (x_col != abstract::Shape::kShapeDimAny && w_row != abstract::Shape::kShapeDimAny && x_col != w_row &&
      x_col >= 0 && w_row >= 0) {
    MS_EXCEPTION(ValueError) << "Dense shape error, got x_col: " << x_col << ", w_row: " << w_row
                             << ". In Dense x_col and w_row should be equal." << kDimW;
  }
}

void CheckDenseShape(const PrimitivePtr &primitive, const AbstractBasePtrList &input_args) {
  const auto &op_name = primitive->name();
  const std::set valid_types = {kUInt8,   kInt8,    kInt16,   kInt32,     kInt64,     kBFloat16,
                                kFloat16, kFloat32, kFloat64, kComplex64, kComplex128};
  std::map<std::string, TypePtr> types;
  (void)types.emplace("x", input_args[kIndex0]->GetType());
  (void)types.emplace("w", input_args[kIndex1]->GetType());
  if (!input_args[kIndex2]->GetType()->isa<TypeNone>()) {
    (void)types.emplace("b", input_args[kIndex2]->GetType());
  }
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, op_name);
}
}  // namespace

void CheckDenseInputs(const PrimitivePtr &primitive, const AbstractBasePtrList &input_args) {
  CheckDenseType(primitive, input_args);
  CheckDenseShape(primitive, input_args);
}

/**
 * Python code for comparison:
 * def dense(input, weight, bias):
 *   def get_transpose_perm(weight):
 *     size = ops.Rank()(weight)
 *     perm = list(range(size))
 *     if size < 2:
 *       perm[0] = 0
 *       return tuple(perm)
 *     perm[size - 1] = size - 2
 *     perm[size - 2] = size - 1
 *     return tuple(perm)
 *
 *   perm = get_transpose_perm(weight)
 *   weight_transposed = ops.Transpose()(weight, perm)
 *   contiguous_out = ops.Contiguous()(weight_transposed)
 *   output = ops.MatMulExt()(input, contiguous_out)
 *   if bias is not None:
 *     output = ops.Add()(output, bias)
 *   return output
 **/
BeginFunction(Dense, input, weight, bias) {
  auto get_transpose_perm = [&](const NodePtr &weight) {
    auto size = Call(Prim(Rank), weight);
    auto true_branch = [&]() {
      auto perm = Call(Prim(MakeRange), size);
      perm = Call(Prim(TupleSetItem), perm, Value(0), Value(0));
      Return(perm);
    };
    auto false_branch = [&]() {
      auto perm = Call(Prim(MakeRange), size);
      auto minus_one = Call(Prim(ScalarSub), size, Value(1));
      auto minus_two = Call(Prim(ScalarSub), size, Value(2));
      perm = Call(Prim(TupleSetItem), perm, minus_one, minus_two);
      perm = Call(Prim(TupleSetItem), perm, minus_two, minus_one);
      Return(perm);
    };
    auto condition = Call(Prim(ScalarLt), size, Value(2));
    return If(condition, true_branch, false_branch);
  };

  auto perm = get_transpose_perm(weight);
  auto weight_transposed = Call(Prim(Transpose), weight, perm);
  auto contiguous_out = Call(Prim(Contiguous), weight_transposed);
  auto output = Call(Prim(MatMulExt), input, contiguous_out);
  auto true_branch = [&]() { Return(Call(Prim(Add), output, bias)); };
  auto false_branch = [&]() { Return(output); };
  Return(If(IsNotNone(bias), true_branch, false_branch));
}
EndFunction(Dense)
}  // namespace mindspore::prim
