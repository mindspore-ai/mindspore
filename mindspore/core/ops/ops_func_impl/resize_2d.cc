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

#include "ops/ops_func_impl/resize_2d.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr Resize2DInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(input_args[0]);
  auto image_shape = input_args[0]->GetShape();
  MS_EXCEPTION_IF_NULL(image_shape);
  auto image_shape_vec = image_shape->GetShapeVector();

  const int64_t image_rank = 4;
  std::vector<int64_t> output_shape(image_rank, abstract::Shape::kShapeDimAny);
  if (!IsDynamicRank(image_shape_vec)) {
    MS_CHECK_VALUE(image_shape_vec.size() == image_rank,
                   CheckAndConvertUtils::FormatCheckIntegerMsg("image rank", SizeToLong(image_shape_vec.size()), kEqual,
                                                               SizeToLong(image_rank), primitive));
    output_shape[kDim0] = image_shape_vec[kDim0];
    output_shape[kDim1] = image_shape_vec[kDim1];
  }

  MS_EXCEPTION_IF_NULL(input_args[1]);
  auto size_value = input_args[1]->GetValue();
  MS_EXCEPTION_IF_NULL(size_value);
  auto size_array_opt = GetArrayValue<int64_t>(size_value);
  if (!size_array_opt.has_value()) {
    return std::make_shared<abstract::Shape>(output_shape);
  }
  const int64_t size_ele_num = 2;
  auto size_array = size_array_opt.value();
  MS_CHECK_VALUE(size_array.size() == size_ele_num,
                 CheckAndConvertUtils::FormatCheckIntegerMsg("the number of size", SizeToLong(size_array.size()),
                                                             kEqual, SizeToLong(size_ele_num), primitive));
  for (size_t i = 0; i < size_array.size(); ++i) {
    if (!size_array.IsValueUnknown(i)) {
      MS_CHECK_VALUE(size_array[i] > 0, CheckAndConvertUtils::FormatCheckIntegerMsg(
                                          "size value", SizeToLong(size_array[i]), kGreaterThan, 0, primitive));
      output_shape[i + kDim2] = size_array[i];
    }
  }
  return std::make_shared<abstract::Shape>(output_shape);
}
}  // namespace ops
}  // namespace mindspore
