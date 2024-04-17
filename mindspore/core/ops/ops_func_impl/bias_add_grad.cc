/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "ops/ops_func_impl/bias_add_grad.h"
#include <memory>
#include <utility>
#include <string>
#include <vector>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace ops {
BaseShapePtr BiasAddGradFuncImpl::InferShape(const PrimitivePtr &primitive,
                                             const std::vector<AbstractBasePtr> &input_args) const {
  auto input_shape_base = input_args[kInputIndex0]->GetShape();
  const auto &input_shape = input_shape_base->GetShapeVector();
  if (IsDynamicRank(input_shape)) {
    return std::make_shared<abstract::TensorShape>(ShapeVector{abstract::Shape::kShapeDimAny});
  }
  const int64_t x_min_rank = 2;
  const int64_t x_max_rank = 5;
  MS_CHECK_VALUE(input_shape.size() >= x_min_rank && input_shape.size() <= x_max_rank,
                 CheckAndConvertUtils::FormatCheckInRangeMsg("rank of input_x", SizeToLong(input_shape.size()),
                                                             kIncludeBoth, {x_min_rank, x_max_rank}, primitive));
  std::vector<int64_t> output_shape;
  auto data_format_ptr = input_args[kInputIndex1]->GetValue();
  MS_EXCEPTION_IF_NULL(data_format_ptr);
  auto data_format_opt = GetScalarValue<int64_t>(data_format_ptr);
  if (!data_format_opt.has_value()) {
    if (input_shape.back() == input_shape[1]) {
      (void)output_shape.emplace_back(input_shape.back());
    } else {
      (void)output_shape.emplace_back(abstract::Shape::kShapeDimAny);
    }
    return std::make_shared<abstract::TensorShape>(std::move(output_shape));
  }

  auto data_format = data_format_opt.value();
  if (data_format == Format::NHWC) {
    (void)output_shape.emplace_back(input_shape.back());
  } else {
    (void)output_shape.emplace_back(input_shape[1]);
  }
  return std::make_shared<abstract::TensorShape>(std::move(output_shape));
}
TypePtr BiasAddGradFuncImpl::InferType(const PrimitivePtr &primitive,
                                       const std::vector<AbstractBasePtr> &input_args) const {
  auto input_type = input_args[0]->GetType();
  return input_type->Clone();
}

int32_t BiasAddGradFuncImpl::CheckValidation(const PrimitivePtr &primitive,
                                             const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto data_format_ptr = input_args[kInputIndex1]->GetValue();
  MS_EXCEPTION_IF_NULL(data_format_ptr);
  auto data_format_opt = GetScalarValue<int64_t>(data_format_ptr);
  if (!data_format_opt.has_value()) {
    return OP_CHECK_RETRY;
  }
  auto data_format = data_format_opt.value();
  if (data_format != Format::NCHW && data_format != Format::NHWC && data_format != Format::NCDHW) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name
                             << "', the data_format must be NCHW, NHWC, or NCDHW, but got: " << data_format << ".";
  }
  return OP_CHECK_SUCCESS;
}
}  // namespace ops
}  // namespace mindspore
