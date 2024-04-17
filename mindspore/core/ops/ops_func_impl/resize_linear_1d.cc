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
#include "ops/ops_func_impl/resize_linear_1d.h"
#include <memory>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr ResizeLinear1DFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(input_args[0]);
  auto x_shape = input_args[0]->GetShape();
  MS_EXCEPTION_IF_NULL(x_shape);
  auto x_shape_vec = x_shape->GetShapeVector();

  const int64_t image_rank = 3;
  std::vector<int64_t> output_shape(image_rank, abstract::Shape::kShapeDimAny);
  if (!IsDynamicRank(x_shape_vec)) {
    MS_CHECK_VALUE(x_shape_vec.size() == image_rank,
                   CheckAndConvertUtils::FormatCheckIntegerMsg("image rank", SizeToLong(x_shape_vec.size()), kEqual,
                                                               SizeToLong(image_rank), primitive));
    output_shape[kDim0] = x_shape_vec[kDim0];
    output_shape[kDim1] = x_shape_vec[kDim1];
  }

  auto size_array_opt = GetArrayValue<int64_t>(input_args[1]);
  if (!size_array_opt.has_value()) {
    return std::make_shared<abstract::Shape>(output_shape);
  }
  const int64_t size_ele_num = 1;
  auto size_array = size_array_opt.value();
  MS_CHECK_VALUE(size_array.size() == size_ele_num,
                 CheckAndConvertUtils::FormatCheckIntegerMsg("the number of size", SizeToLong(size_array.size()),
                                                             kEqual, SizeToLong(size_ele_num), primitive));
  if (!size_array.IsValueUnknown(kIndex0)) {
    MS_CHECK_VALUE(size_array[kIndex0] > 0,
                   CheckAndConvertUtils::FormatCheckIntegerMsg("size value", SizeToLong(size_array[kIndex0]),
                                                               kGreaterThan, 0, primitive));
    output_shape[kDim2] = size_array[kIndex0];
  }
  return std::make_shared<abstract::Shape>(output_shape);
}

TypePtr ResizeLinear1DFuncImpl::InferType(const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(input_args[0]->GetType());
  return input_args[0]->GetType()->Clone();
}
}  // namespace ops
}  // namespace mindspore
