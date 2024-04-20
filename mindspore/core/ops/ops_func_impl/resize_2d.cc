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

#include "ops/ops_func_impl/resize_2d.h"

#include <memory>
#include <utility>

#include "ops/ops_func_impl/op_func_impl.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
namespace {
BaseShapePtr Resize2DInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  MS_EXCEPTION_IF_NULL(input_args.at(0));
  auto image_shape = input_args[0]->GetShape();
  MS_EXCEPTION_IF_NULL(image_shape);
  const auto &image_shape_vec = image_shape->GetShapeVector();

  const int64_t image_rank = 4;
  std::vector<int64_t> output_shape(image_rank, abstract::Shape::kShapeDimAny);
  if (MS_LIKELY(!IsDynamicRank(image_shape_vec))) {
    MS_CHECK_VALUE(image_shape_vec.size() == image_rank,
                   CheckAndConvertUtils::FormatCheckIntegerMsg("image rank", SizeToLong(image_shape_vec.size()), kEqual,
                                                               SizeToLong(image_rank), primitive));
    output_shape[kDim0] = image_shape_vec[kDim0];
    output_shape[kDim1] = image_shape_vec[kDim1];
  }

  auto size_array_opt = GetArrayValue<int64_t>(input_args[1]);
  if (MS_UNLIKELY(!size_array_opt.has_value())) {
    return std::make_shared<abstract::TensorShape>(std::move(output_shape));
  }

  const int64_t size_ele_num = 2;
  auto size_array = size_array_opt.value();
  MS_CHECK_VALUE(size_array.size() == size_ele_num,
                 CheckAndConvertUtils::FormatCheckIntegerMsg("the number of size", SizeToLong(size_array.size()),
                                                             kEqual, SizeToLong(size_ele_num), primitive));
  for (size_t i = 0; i < size_array.size(); ++i) {
    if (!size_array.IsValueUnknown(i)) {
      MS_CHECK_VALUE(size_array[i] > 0, CheckAndConvertUtils::FormatCheckIntegerMsg("size value", size_array[i],
                                                                                    kGreaterThan, 0, primitive));
      output_shape[i + kDim2] = size_array[i];
    }
  }

  return std::make_shared<abstract::TensorShape>(std::move(output_shape));
}

int32_t Resize2DCheckValidation(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  MS_EXCEPTION_IF_NULL(input_args[2]);
  MS_EXCEPTION_IF_NULL(input_args[3]);
  auto align_corners_opt = GetScalarValue<bool>(input_args[2]->GetValue());
  auto half_pixel_centers_opt = GetScalarValue<bool>(input_args[3]->GetValue());
  if (MS_UNLIKELY(!align_corners_opt.has_value() || !half_pixel_centers_opt.has_value())) {
    return OP_CHECK_RETRY;
  }

  auto align_corners = align_corners_opt.value();
  auto half_pixel_centers = half_pixel_centers_opt.value();
  if (MS_UNLIKELY(align_corners && half_pixel_centers)) {
    MS_EXCEPTION(ValueError)
      << "For " << primitive->name()
      << ", the half_pixel_centers must be false when align_corners is true, but half_pixel_centers got True.";
  }

  return OP_CHECK_SUCCESS;
}
}  // namespace

BaseShapePtr Resize2DBaseFuncImpl::InferShape(const PrimitivePtr &primitive,
                                              const std::vector<AbstractBasePtr> &input_args) const {
  return Resize2DInferShape(primitive, input_args);
}

TypePtr Resize2DBaseFuncImpl::InferType(const PrimitivePtr &primitive,
                                        const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(input_args[0]->GetType());
  return input_args[0]->GetType()->Clone();
}

int32_t Resize2DBaseFuncImpl::CheckValidation(const PrimitivePtr &primitive,
                                              const std::vector<AbstractBasePtr> &input_args) const {
  return Resize2DCheckValidation(primitive, input_args);
}
}  // namespace ops
}  // namespace mindspore
