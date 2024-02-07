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

#include <set>
#include <memory>
#include <unordered_map>
#include "ops/op_utils.h"
#include "ops/ops_func_impl/fft.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr FFTFuncImpl::InferShape(const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) const {
  auto input_shape_ptr = input_args[kIndex0]->GetShape();
  auto input_shape = input_shape_ptr->GetShapeVector();

  // When input is a dynamic rank, it needs to be processed in the kernel
  if (IsDynamicRank(input_shape)) {
    ShapeVector dyn_output{abstract::TensorShape::kShapeRankAny};
    return std::make_shared<abstract::TensorShape>(dyn_output);
  }

  auto y_shape = input_shape;
  if (!input_args[kInputIndex1]->GetType()->isa<TypeNone>()) {
    auto n_opt = GetScalarValue<int64_t>(input_args[kInputIndex1]->GetValue());
    if (n_opt.has_value()) {
      auto n = n_opt.value();
      auto dim_opt = GetScalarValue<int64_t>(input_args[kInputIndex2]->GetValue());
      auto x_rank = input_shape.size();

      int64_t tmp_pos;
      if (dim_opt.has_value()) {
        int64_t dim = dim_opt.value();
        tmp_pos = dim < 0 ? x_rank + dim : dim;
      } else {
        tmp_pos = x_rank - 1;
      }
      y_shape[tmp_pos] = n;
    }
  }

  return std::make_shared<abstract::TensorShape>(y_shape);
}

TypePtr FFTFuncImpl::InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
  auto input_type = input_args[kIndex0]->GetType();
  auto input_type_id = input_type->cast<TensorTypePtr>()->element()->type_id();

  static const std::vector<TypeId> double_type = {kNumberTypeFloat64, kNumberTypeComplex128};
  bool is_double_type = std::any_of(double_type.begin(), double_type.end(),
                                    [&input_type_id](const TypeId &type_id) { return input_type_id == type_id; });
  if (is_double_type) {
    return std::make_shared<TensorType>(kComplex128);
  } else {
    return std::make_shared<TensorType>(kComplex64);
  }
}

/*
  Error list:
  1) `input.ndim` is not in the range of "[1, 8]".
  2) The value in `dim` is not in the range of "[-`input.ndim`, `input.ndim`)"
  3) The value in `n` is less than or equal to 0.
*/
int32_t FFTFuncImpl::CheckValidation(const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) const {
  auto check_status = OP_CHECK_SUCCESS;
  const auto &input_x_shape = input_args[kIndex0]->GetShape();
  auto x_shape_vec = input_x_shape->GetShapeVector();

  if (MS_UNLIKELY(IsDynamicRank(x_shape_vec))) {
    check_status = OP_CHECK_RETRY;
  }
  const int64_t x_min_rank = 1;
  const int64_t x_max_rank = 8;
  int64_t x_rank = SizeToLong(x_shape_vec.size());

  MS_CHECK_VALUE(x_shape_vec.size() >= x_min_rank && x_shape_vec.size() <= x_max_rank,
                 CheckAndConvertUtils::FormatCheckInRangeMsg("rank of input", x_rank, kIncludeBoth,
                                                             {x_min_rank, x_max_rank}, primitive));

  if (x_rank == 1 && x_shape_vec[0] == 0) {
    MS_EXCEPTION(ValueError) << "Unsupported input shape dimension. The shape should not be empty.";
  }

  if (!input_args[kInputIndex1]->GetType()->isa<TypeNone>()) {
    auto n_opt = GetScalarValue<int64_t>(input_args[kInputIndex1]->GetValue());
    if (n_opt.has_value()) {
      int64_t n = n_opt.value();
      (void)CheckAndConvertUtils::CheckInteger("n", n, kGreaterThan, 0);
    }
  }

  auto dim_opt = GetScalarValue<int64_t>(input_args[kInputIndex2]->GetValue());
  if (dim_opt.has_value()) {
    int64_t dim = dim_opt.value();
    MS_CHECK_VALUE(dim >= -x_rank && dim < x_rank, CheckAndConvertUtils::FormatCheckInRangeMsg(
                                                     "dim", dim, kIncludeRight, {-x_rank, x_rank}, primitive));
  }

  return check_status;
}
}  // namespace ops
}  // namespace mindspore
