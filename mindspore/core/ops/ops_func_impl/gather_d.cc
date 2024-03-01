/**
 * Copyright 2023-2024 Huawei Technologies Co., Ltd
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
#include "utils/shape_utils.h"

namespace mindspore {
namespace ops {
inline BaseShapePtr InferDynamicForcely(const ShapeVector &index_shape) {
  auto out_shape = index_shape;
  if (index_shape.size() > 0) {
    out_shape[0] = abstract::Shape::kShapeDimAny;
  }

  return std::make_shared<abstract::Shape>(out_shape);
}

BaseShapePtr GatherDFuncImpl::InferShape(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) const {
  auto dim_scalar = GetScalarValue<int64_t>(input_args[kInputIndex1]->GetValue());
  auto index_shape = input_args[kInputIndex2]->GetShape()->GetShapeVector();
  auto x_shape = input_args[kInputIndex0]->GetShape()->GetShapeVector();
  if (!dim_scalar.has_value()) {
    if (!IsDynamic(index_shape) && !IsDynamic(x_shape)) {
      // GahterD will map to GatherElements when backend is GE, however GatherElements only support a constant dim
      // so we set the out shape to dynamic to run with aclnn forcely
      return InferDynamicForcely(index_shape);
    }
  }

  return input_args[kInputIndex2]->GetShape()->Clone();
}

TypePtr GatherDFuncImpl::InferType(const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) const {
  auto index_type = input_args[kInputIndex2]->GetType();
  const std::set<TypePtr> valid_types = {kInt32, kInt64};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("index", index_type, valid_types, primitive->name());
  return input_args[kInputIndex0]->GetType()->Clone();
}

int32_t GatherDFuncImpl::CheckValidation(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) const {
  auto x_shape_vec = input_args[kInputIndex0]->GetShape()->GetShapeVector();
  auto index_shape_vec = input_args[kInputIndex2]->GetShape()->GetShapeVector();

  if (IsDynamicRank(x_shape_vec) || IsDynamicRank(index_shape_vec)) {
    return OP_CHECK_RETRY;
  }

  x_shape_vec = x_shape_vec.empty() ? ShapeVector({1}) : x_shape_vec;
  index_shape_vec = index_shape_vec.empty() ? ShapeVector({1}) : index_shape_vec;

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

    MS_CHECK_VALUE(x_shape_vec[i] >= index_shape_vec[i],
                   CheckAndConvertUtils::FormatCheckIntegerMsg("x shape", x_shape_vec[i], kGreaterEqual,
                                                               index_shape_vec[i], primitive));
  }

  return OP_CHECK_SUCCESS;
}
}  // namespace ops
}  // namespace mindspore
