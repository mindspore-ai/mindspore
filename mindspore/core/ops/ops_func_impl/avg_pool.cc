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

#include "ops/ops_func_impl/avg_pool.h"
#include <algorithm>
#include <memory>
#include <utility>
#include "abstract/dshape.h"
#include "ops/op_name.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace ops {
// Compute out h and w for VALID pad mode.
// Note: should check whether kernel_size and strides contain unknown values.
inline int64_t ComputeValid(int64_t in_value, const ArrayValue<int64_t> &kernel_size_array,
                            const ArrayValue<int64_t> &strides_array, size_t index) {
  ShapeValueDType out_value;
  if (in_value == abstract::Shape::kShapeDimAny) {
    out_value = abstract::Shape::kShapeDimAny;
  } else if (kernel_size_array.IsValueUnknown(index) || strides_array.IsValueUnknown(index)) {
    out_value = abstract::Shape::kShapeDimAny;
  } else {
    out_value = static_cast<int64_t>(
      std::ceil((in_value - (kernel_size_array[index] - 1)) / static_cast<float>(strides_array[index])));
  }

  return out_value;
}

// Compute out h and w for SAME pad mode.
// Note: should check whether strides contain unknown values.
inline int64_t ComputeSame(int64_t in_value, const ArrayValue<int64_t> &strides_array, size_t index) {
  ShapeValueDType out_value;
  if (in_value == abstract::Shape::kShapeDimAny) {
    out_value = abstract::Shape::kShapeDimAny;
  } else if (strides_array.IsValueUnknown(index)) {
    out_value = abstract::Shape::kShapeDimAny;
  } else {
    out_value = static_cast<int64_t>(std::ceil(in_value / static_cast<float>(strides_array[index])));
  }
  return out_value;
}

BaseShapePtr AvgPoolFuncImpl::InferShape(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) const {
  // Step1. Get input tensor shape.
  BaseShapePtr base_shape = input_args[kInputIndex0]->GetShape();
  // Could use GetShapeVector to get shape for Tensor type only.
  const auto &in_shape = base_shape->GetShapeVector();

  if (MS_UNLIKELY(IsDynamicRank(in_shape))) {
    ShapeVector dyn_output{abstract::Shape::kShapeRankAny};
    return std::make_shared<abstract::Shape>(std::move(dyn_output));
  }

  constexpr auto x_rank = 4;
  MS_CHECK_VALUE(in_shape.size() == x_rank,
                 CheckAndConvertUtils::FormatCheckIntegerMsg("x_rank", SizeToLong(in_shape.size()), kEqual,
                                                             SizeToLong(x_rank), primitive));

  // Step2. Get input format value.
  auto format_value = input_args[kInputIndex4]->GetValue();
  auto format_opt = GetScalarValue<int64_t>(format_value);
  // If the value of format is ValueAny, return a dynamic shape and only the Batch dimension can be inferred.
  if (!format_opt.has_value()) {
    ShapeVector dyn_output{in_shape[0], abstract::Shape::kShapeDimAny, abstract::Shape::kShapeDimAny,
                           abstract::Shape::kShapeDimAny};
    return std::make_shared<abstract::Shape>(std::move(dyn_output));
  }

  ShapeValueDType batch = in_shape[kInputIndex0];
  ShapeValueDType channel;
  ShapeValueDType in_h;
  ShapeValueDType in_w;
  mindspore::Format format = static_cast<mindspore::Format>(format_opt.value());
  if (format == NCHW) {
    channel = in_shape[kInputIndex1];
    in_h = in_shape[kInputIndex2];
    in_w = in_shape[kInputIndex3];
  } else {
    channel = in_shape[kInputIndex3];
    in_h = in_shape[kInputIndex1];
    in_w = in_shape[kInputIndex2];
  }

  // Step3. Get and check kernel_size, strides and pad_mode value.
  auto kernel_size_value = input_args[kInputIndex1]->GetValue();
  auto strides_value = input_args[kInputIndex2]->GetValue();
  auto pad_mode_value = input_args[kInputIndex3]->GetValue();

  auto kernel_size_array_opt = GetArrayValue<int64_t>(kernel_size_value);
  auto strides_array_opt = GetArrayValue<int64_t>(strides_value);
  auto pad_mode_opt = GetScalarValue<int64_t>(pad_mode_value);

  // If any of the above values is ValueAny, return a dynamic shape, only the Batch and Channel dimension can be
  // inferred.
  if (!kernel_size_array_opt.has_value() || !strides_array_opt.has_value() || !pad_mode_opt.has_value()) {
    if (format == NHWC) {
      ShapeVector dyn_output{batch, abstract::Shape::kShapeDimAny, abstract::Shape::kShapeDimAny, channel};
      return std::make_shared<abstract::Shape>(std::move(dyn_output));
    }
    ShapeVector dyn_output{batch, channel, abstract::Shape::kShapeDimAny, abstract::Shape::kShapeDimAny};
    return std::make_shared<abstract::TensorShape>(std::move(dyn_output));
  }

  mindspore::PadMode pad_mode = static_cast<mindspore::PadMode>(pad_mode_opt.value());
  const auto &kernel_size_array = kernel_size_array_opt.value();
  const auto &strides_array = strides_array_opt.value();

  // Step4. Compute output shape.
  ShapeValueDType out_h = abstract::Shape::kShapeDimAny;
  ShapeValueDType out_w = abstract::Shape::kShapeDimAny;

  if (pad_mode == VALID) {
    out_h = ComputeValid(in_h, kernel_size_array, strides_array, kInputIndex0);
    out_w = ComputeValid(in_w, kernel_size_array, strides_array, kInputIndex1);
  } else if (pad_mode == SAME) {
    out_h = ComputeSame(in_h, strides_array, kInputIndex0);
    out_w = ComputeSame(in_w, strides_array, kInputIndex1);
  }

  ShapeVector out_shape;
  if (format == NCHW) {
    out_shape = {batch, channel, out_h, out_w};
  } else {
    out_shape = {batch, out_h, out_w, channel};
  }

  return std::make_shared<abstract::TensorShape>(std::move(out_shape));
}

TypePtr AvgPoolFuncImpl::InferType(const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  // Get input object type.
  auto input_type = input_args[kInputIndex0]->GetType();
  MS_EXCEPTION_IF_NULL(input_type);

  // TensorType
  return input_type;
}

// Check kernel_size and strides length and value.
inline void CheckKernelSizeAndStrides(const PrimitivePtr &primitive, const ArrayValue<int64_t> &kernel_size_array,
                                      const ArrayValue<int64_t> &strides_array) {
  const size_t attr_size = 2;
  MS_CHECK_VALUE(kernel_size_array.size() == attr_size,
                 CheckAndConvertUtils::FormatCheckIntegerMsg("kernel", SizeToLong(kernel_size_array.size()), kEqual,
                                                             SizeToLong(attr_size), primitive));
  MS_CHECK_VALUE(strides_array.size() == attr_size,
                 CheckAndConvertUtils::FormatCheckIntegerMsg("strides", SizeToLong(strides_array.size()), kEqual,
                                                             SizeToLong(attr_size), primitive));

  auto op_name = primitive->name();
  for (size_t i = 0; i < kernel_size_array.size(); i++) {
    if (MS_UNLIKELY(kernel_size_array[i] <= 0)) {
      MS_LOG(EXCEPTION) << "For '" << op_name << "', kernel size must be positive, but it's "
                        << kernel_size_array.ToString() << ".";
    }
  }
  for (size_t i = 0; i < strides_array.size(); i++) {
    if (MS_UNLIKELY(strides_array[i] <= 0)) {
      MS_LOG(EXCEPTION) << "For '" << op_name << "', kernel size must be positive, but it's "
                        << strides_array.ToString() << ".";
    }
  }
}

int32_t AvgPoolFuncImpl::CheckValidation(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) const {
  int32_t check_status = OP_CHECK_SUCCESS;

  // Step1: Check format valid.
  auto format_value = input_args[kInputIndex4]->GetValue();
  auto format_opt = GetScalarValue<int64_t>(format_value);
  // If the value of format is ValueAny, return a dynamic shape and only the Batch dimension can be inferred.
  if (MS_UNLIKELY(!format_opt.has_value())) {
    check_status = OP_CHECK_RETRY;
  } else {
    mindspore::Format format = static_cast<mindspore::Format>(format_opt.value());
    if (MS_UNLIKELY(format != NCHW && format != NHWC)) {
      MS_LOG(EXCEPTION) << "The data format value " << FormatEnumToString(format) << " is invalid, "
                        << primitive->name() << " only support NCHW and NHWC";
    }
  }

  // Step2: Check kernel_size and strides valid.
  auto kernel_size_value = input_args[kInputIndex1]->GetValue();
  auto strides_value = input_args[kInputIndex2]->GetValue();
  auto kernel_size_array_opt = GetArrayValue<int64_t>(kernel_size_value);
  auto strides_array_opt = GetArrayValue<int64_t>(strides_value);
  if (MS_UNLIKELY(!kernel_size_array_opt.has_value() || !strides_array_opt.has_value())) {
    check_status = OP_CHECK_RETRY;
  } else {
    const auto &kernel_size_array = kernel_size_array_opt.value();
    const auto &strides_array = strides_array_opt.value();
    if (MS_UNLIKELY(kernel_size_array.HasUnknownValue() || strides_array.HasUnknownValue())) {
      check_status = OP_CHECK_RETRY;
    } else {
      CheckKernelSizeAndStrides(primitive, kernel_size_array, strides_array);
    }
  }
  return check_status;
}
}  // namespace ops
}  // namespace mindspore
