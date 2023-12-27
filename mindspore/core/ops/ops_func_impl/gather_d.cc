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

#include "ops/ops_func_impl/gather_d.h"
#include "utils/check_convert_utils.h"
#include "ops/op_name.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr GatherDFuncImpl::InferShape(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) const {
  return input_args[kInputIndex2]->GetShape()->Clone();
}

TypePtr GatherDFuncImpl::InferType(const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) const {
  return input_args[kInputIndex0]->GetType()->Clone();
}

int32_t GatherDFuncImpl::CheckValidation(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) const {
  auto x_shape_vec = input_args[kInputIndex0]->GetShape()->GetShapeVector();
  auto index_shape_vec = input_args[kInputIndex2]->GetShape()->GetShapeVector();

  if (IsDynamicRank(x_shape_vec) || IsDynamicRank(index_shape_vec)) {
    return OP_CHECK_RETRY;
  }

  int64_t x_rank = SizeToLong(x_shape_vec.size());
  int64_t index_rank = SizeToLong(index_shape_vec.size());
  MS_CHECK_VALUE(x_rank == index_rank,
                 CheckAndConvertUtils::FormatCheckIntegerMsg("x_rank", x_rank, kEqual, index_rank, primitive));

  auto dims_scalar = GetScalarValue<int64_t>(input_args[kInputIndex1]->GetValue());
  int64_t dim = 0;
  if (dims_scalar.has_value()) {
    dim = dims_scalar.value();
    if (dim < 0) {
      dim = dim + x_rank;
    }
  } else {
    return OP_CHECK_RETRY;
  }

  MS_CHECK_VALUE(dim >= 0 && dim < x_rank, CheckAndConvertUtils::FormatCheckInRangeMsg("dim value", dim, kIncludeBoth,
                                                                                       {-x_rank, x_rank}, primitive));

  for (int64_t i = 0; i < x_rank; ++i) {
    if (i == dim) {
      continue;
    }

    if (x_shape_vec[i] == abstract::TensorShape::kShapeDimAny ||
        index_shape_vec[i] == abstract::TensorShape::kShapeDimAny) {
      return OP_CHECK_RETRY;
    }

    MS_CHECK_VALUE(
      x_shape_vec[i] == index_shape_vec[i],
      CheckAndConvertUtils::FormatCheckIntegerMsg("x shape", x_shape_vec[i], kEqual, index_shape_vec[i], primitive));
  }

  return OP_CHECK_SUCCESS;
}
}  // namespace ops
}  // namespace mindspore
