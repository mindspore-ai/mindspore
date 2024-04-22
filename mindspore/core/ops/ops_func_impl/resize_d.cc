/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "ops/ops_func_impl/resize_d.h"

#include <memory>
#include <utility>
#include <string>

#include "mindapi/base/types.h"
#include "ops/ops_func_impl/op_func_impl.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr ResizeDFuncImpl::InferShape(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) const {
  const size_t expect_rank = 4;
  auto x_shape = input_args.at(0)->GetShape()->GetShapeVector();
  if (MS_LIKELY(!IsDynamic(x_shape))) {
    MS_CHECK_VALUE(x_shape.size() == expect_rank,
                   CheckAndConvertUtils::FormatCheckIntegerMsg("image rank", SizeToLong(x_shape.size()), kEqual,
                                                               SizeToLong(expect_rank), primitive));
    if (x_shape[kIndex2] != 1) {
      MS_LOG(EXCEPTION) << "For " << primitive->name() << " with `linear` mode, x_shape[2] should be 1, but got "
                        << x_shape[kIndex2] << ".";
    }
  } else {
    MS_LOG(EXCEPTION) << "ResizeD do not support dynamic shape input.";
  }

  auto sizes_opt = GetArrayValue<int64_t>(input_args[kIndex1]);
  if (MS_UNLIKELY(!sizes_opt.has_value())) {
    MS_LOG(EXCEPTION) << "For " << primitive->name() << ", sizes should be const.";
  }
  auto sizes_array = sizes_opt.value();
  MS_CHECK_VALUE(sizes_array.size() == 1,
                 CheckAndConvertUtils::FormatCheckIntegerMsg("the number of sizes", SizeToLong(sizes_array.size()),
                                                             kEqual, SizeToLong(1), primitive));
  if (MS_UNLIKELY(sizes_array.IsValueUnknown(0))) {
    MS_LOG(EXCEPTION) << "For " << primitive->name() << ", sizes should be const.";
  }
  MS_CHECK_VALUE(sizes_array[0] > 0,
                 CheckAndConvertUtils::FormatCheckIntegerMsg("sizes[0]", sizes_array[0], kGreaterThan, 0, primitive));
  x_shape[kIndex3] = sizes_array[0];

  return std::make_shared<abstract::TensorShape>(std::move(x_shape));
}

TypePtr ResizeDFuncImpl::InferType(const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(input_args[0]->GetType());
  return input_args[0]->GetType()->Clone();
}

int32_t ResizeDFuncImpl::CheckValidation(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) const {
  auto mode_ptr = primitive->GetAttr("mode");
  MS_EXCEPTION_IF_NULL(mode_ptr);
  auto mode = GetValue<std::string>(mode_ptr);
  if (mode != "linear") {
    MS_LOG(EXCEPTION) << "For " << primitive->name() << ", mode should be linear.";
  }

  auto scales_opt = GetArrayValue<pyfloat>(input_args[kIndex2]);
  if (MS_UNLIKELY(!scales_opt.has_value())) {
    MS_LOG(EXCEPTION) << "For " << primitive->name() << ", scales should be const.";
  }
  auto scales_array = scales_opt.value();
  MS_CHECK_VALUE(scales_array.size() == 1,
                 CheckAndConvertUtils::FormatCheckIntegerMsg("the number of scales", SizeToLong(scales_array.size()),
                                                             kEqual, SizeToLong(1), primitive));

  auto coordinate_transformation_mode_opt = GetScalarValue<int64_t>(input_args[kIndex3]->GetValue());
  if (MS_UNLIKELY(!coordinate_transformation_mode_opt.has_value())) {
    MS_LOG(EXCEPTION) << "For " << primitive->name() << ", coordinate_transformation_mode should be linear.";
  }
  auto coordinate_transformation_mode =
    static_cast<CoordinateTransformMode>(coordinate_transformation_mode_opt.value());
  if (coordinate_transformation_mode != CoordinateTransformMode::ALIGN_CORNERS &&
      coordinate_transformation_mode != CoordinateTransformMode::HALF_PIXEL) {
    MS_LOG(EXCEPTION) << "For " << primitive->name()
                      << ", coordinate_transformation_mode should be `align_corners` or `half_pixel`, but got "
                      << coordinate_transformation_mode;
  }

  return OP_CHECK_SUCCESS;
}
}  // namespace ops
}  // namespace mindspore
