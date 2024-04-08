/**
 * Copyright 2024 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "ops/ops_func_impl/solve_triangular.h"
#include <set>
#include <vector>
#include <memory>
#include "abstract/dshape.h"
#include "ops/op_name.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr SolveTriangularFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                 const std::vector<AbstractBasePtr> &input_args) const {
  auto a_shape = input_args[kInputIndex0]->GetShape()->GetShapeVector();
  auto b_shape = input_args[kInputIndex1]->GetShape()->GetShapeVector();
  if (IsDynamic(a_shape) || IsDynamic(b_shape)) {
    auto b_shape_ptr = input_args[kInputIndex1]->GetShape()->Clone();
    return b_shape_ptr->Clone();
  }

  constexpr size_t square_size = 2;
  const size_t expected_b_dim = (b_shape.size() == a_shape.size() - 1) ? 1 : square_size;
  auto a_rank = a_shape.size();
  auto b_rank = b_shape.size();
  if (a_rank < square_size) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                             << "', dim of matrix a must greater or equal to 2, but got a at " << a_rank
                             << "-dimensional ";
  }
  if (a_rank != b_rank && a_rank != b_rank + 1) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                             << "', the dimension of `b` should be 'a.dim' or 'a.dim' - 1, which is " << a_rank
                             << " or " << (a_rank - 1) << ", but got " << b_rank << "-dimensions.";
  }
  if (a_shape[a_rank - kIndex1] != a_shape[a_rank - square_size]) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                             << "', the last two dimensions of `a` should be the same, but got shape of " << a_shape
                             << ". Please make sure that the shape of `a` be like [..., N, N].";
  }
  if (a_shape[a_rank - square_size] != b_shape[b_rank - expected_b_dim]) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                             << "', the last two dimensions of `a` and `b` should be matched, but got shape of "
                             << a_shape << " and " << b_shape
                             << ". Please make sure that the shape of `a` and `b` be like [..., N, N] X [..., N, M] or "
                                "[..., N, N ] X[..., N].";
  }
  if (!std::equal(a_shape.begin(), a_shape.begin() + (a_rank - square_size), b_shape.begin(),
                  b_shape.begin() + (b_rank - expected_b_dim))) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                             << "', the batch dimensions of `a` and `b` should all be the same, but got shape of "
                             << a_shape << " and " << b_shape
                             << ". Please make sure that the shape of `a` and `b` be like [a, b, c, ..., N, N] X [a, "
                                "b, c, ..., N, M] or [a, b, c, ..., N, N] X [a, b, c, ..., N].";
  }
  auto b_shape_ptr = input_args[kInputIndex1]->GetShape()->Clone();
  return b_shape_ptr->Clone();
}

TypePtr SolveTriangularFuncImpl::InferType(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) const {
  auto op_name = primitive->name();
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex0]);
  auto input_a_type = input_args[kInputIndex0]->GetType();
  auto a_type_ptr = input_a_type->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(a_type_ptr);
  auto input_a_type_id = a_type_ptr->element()->type_id();
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex1]);
  auto input_b_type = input_args[kInputIndex1]->GetType();
  auto b_type_ptr = input_b_type->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(b_type_ptr);
  auto input_b_type_id = b_type_ptr->element()->type_id();
  if (input_a_type_id != input_b_type_id) {
    MS_EXCEPTION(TypeError) << "For '" << primitive->name()
                            << "' the type of a and b must be same, but got type of a is different from that of b! ";
  }
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kFloat64, kComplex64, kComplex128,
                                         kInt8,    kInt16,   kInt32,   kInt64};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("a", input_a_type, valid_types, op_name);
  static const std::vector<TypeId> type_to_float32 = {
    kNumberTypeInt8,
    kNumberTypeInt16,
    kNumberTypeInt32,
  };
  bool is_type_to_float32 =
    std::any_of(type_to_float32.begin(), type_to_float32.end(),
                [&input_a_type_id](const TypeId &type_id) { return input_a_type_id == type_id; });
  if (is_type_to_float32) return std::make_shared<TensorType>(kFloat32);

  static const std::vector<TypeId> type_to_float64 = {kNumberTypeInt64};
  bool is_type_to_float64 =
    std::any_of(type_to_float64.begin(), type_to_float64.end(),
                [&input_a_type_id](const TypeId &type_id) { return input_a_type_id == type_id; });
  if (is_type_to_float64) return std::make_shared<TensorType>(kFloat64);

  return input_a_type->Clone();
}
}  // namespace ops
}  // namespace mindspore
