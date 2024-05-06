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

#include "ops/ops_func_impl/avg_pool2d.h"

#include <tuple>
#include <string>
#include <functional>
#include <algorithm>
#include <utility>

#include "ops/op_name.h"
#include "ops/op_utils.h"
#include "mindapi/base/types.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"

namespace mindspore {
namespace ops {
namespace {
void AvgPool2DCheckTupleIntParam(const PrimitivePtr &primitive, const ArrayValue<int64_t> &sequence_array,
                                 const int64_t min_ele_num, const int64_t max_ele_num, const int64_t compare_value,
                                 const std::string &arg_name) {
  const auto ele_num = SizeToLong(sequence_array.size());
  MS_CHECK_VALUE(ele_num >= min_ele_num && ele_num <= max_ele_num,
                 CheckAndConvertUtils::FormatCheckInRangeMsg("number of " + arg_name, ele_num, kIncludeBoth,
                                                             {min_ele_num, max_ele_num}, primitive));
  for (size_t i = 0; i < sequence_array.size(); ++i) {
    if (MS_UNLIKELY(sequence_array.IsValueUnknown(i))) {
      continue;
    }
    MS_CHECK_VALUE(sequence_array[i] > compare_value,
                   CheckAndConvertUtils::FormatCheckIntegerMsg(arg_name + " value", sequence_array[i], kGreaterThan,
                                                               compare_value, primitive));
  }
}

static inline int64_t AvgPool2DCeilDiv(int64_t x, int64_t y) {
  auto z = DoubleToLong(x * 1.0 / y);
  return z;
}

static inline int64_t AvgPool2DOutputShapePadLR(int64_t input_size, int64_t kernel_size, int64_t pad, int64_t stride,
                                                bool ceil_mode) {
  auto output_size = AvgPool2DCeilDiv(input_size + 2 * pad - kernel_size + (ceil_mode ? stride - 1 : 0), stride) + 1;
  if (ceil_mode) {
    // ensure that the last pooling starts inside the image needed to avoid problems in ceil mode
    if ((output_size - 1) * stride >= input_size + pad) {
      --output_size;
    }
  }
  return output_size;
}
}  // namespace
BaseShapePtr AvgPool2DFuncImpl::InferShape(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) const {
  const size_t kNum2 = 2;
  const size_t kRank3 = 3;
  const size_t kRank4 = 4;

  const auto &input_shape = input_args[kInputIndex0]->GetShape()->GetShapeVector();
  if (MS_UNLIKELY(IsDynamicRank(input_shape))) {
    return std::make_shared<abstract::TensorShape>(std::vector<int64_t>{abstract::TensorShape::kShapeRankAny});
  }
  auto input_rank = input_shape.size();

  MS_CHECK_VALUE(input_rank >= kRank3 && input_rank <= kRank4,
                 CheckAndConvertUtils::FormatCheckInRangeMsg("input rank", SizeToLong(input_rank), kIncludeBoth,
                                                             {kRank3, kRank4}, primitive));

  std::vector<int64_t> output_shape(input_rank, abstract::TensorShape::kShapeDimAny);
  std::transform(input_shape.begin(), input_shape.begin() + input_rank - kNum2, output_shape.begin(),
                 [](const int64_t v) { return v; });

  auto kernel_size_opt = GetArrayValue<int64_t>(input_args[kInputIndex1]);
  auto stride_opt = GetArrayValue<int64_t>(input_args[kInputIndex2]);
  auto padding_opt = GetArrayValue<int64_t>(input_args[kInputIndex3]);
  auto ceil_mode_opt = GetScalarValue<bool>(input_args[kInputIndex4]->GetValue());
  if (MS_LIKELY((ceil_mode_opt.has_value() && kernel_size_opt.has_value() && stride_opt.has_value() &&
                 padding_opt.has_value()))) {
    auto ceil_mode = ceil_mode_opt.value();
    auto kernel_size = kernel_size_opt.value();
    auto stride = stride_opt.value();
    auto padding = padding_opt.value();
    for (size_t i = 0; i < kNum2; ++i) {
      auto dim = input_rank - kNum2 + i;
      auto cur_dim_value = input_shape[dim];
      auto idx_kernel_size = i % kernel_size.size();
      auto idx_stride = i % stride.size();
      auto idx_padding = i % padding.size();
      if (MS_UNLIKELY(cur_dim_value == abstract::TensorShape::kShapeDimAny ||
                      kernel_size.IsValueUnknown(idx_kernel_size) || stride.IsValueUnknown(idx_stride) ||
                      padding.IsValueUnknown(idx_padding))) {
        continue;
      }

      auto output_size = AvgPool2DOutputShapePadLR(cur_dim_value, kernel_size[idx_kernel_size], padding[idx_padding],
                                                   stride[idx_stride], ceil_mode);
      MS_CHECK_VALUE(output_size > 0, CheckAndConvertUtils::FormatCheckIntegerMsg("output_size", output_size,
                                                                                  kGreaterThan, 0, primitive));

      output_shape[dim] = output_size;
    }
  }

  return std::make_shared<abstract::TensorShape>(std::move(output_shape));
}

TypePtr AvgPool2DFuncImpl::InferType(const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) const {
  const auto &input_type = input_args[kInputIndex0]->GetType();
  return input_type->Clone();
}

int32_t AvgPool2DFuncImpl::CheckValidation(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) const {
  auto kernel_size_opt = GetArrayValue<int64_t>(input_args[kInputIndex1]);
  if (MS_UNLIKELY(!kernel_size_opt.has_value())) {
    return OP_CHECK_RETRY;
  }

  const int64_t min_ele_num = 1;
  const int64_t max_ele_num = 2;
  auto kernel_size = kernel_size_opt.value();
  AvgPool2DCheckTupleIntParam(primitive, kernel_size, min_ele_num, max_ele_num, 0, "kernel_size");

  auto stride_opt = GetArrayValue<int64_t>(input_args[kInputIndex2]);
  if (MS_UNLIKELY(!stride_opt.has_value())) {
    return OP_CHECK_RETRY;
  }

  auto stride = stride_opt.value();
  AvgPool2DCheckTupleIntParam(primitive, stride, min_ele_num, max_ele_num, 0, "stride");

  auto padding_opt = GetArrayValue<int64_t>(input_args[kInputIndex3]);
  if (MS_UNLIKELY(!padding_opt.has_value())) {
    return OP_CHECK_RETRY;
  }

  auto padding = padding_opt.value();
  AvgPool2DCheckTupleIntParam(primitive, padding, min_ele_num, max_ele_num, -1, "padding");

  for (size_t i = 0; i < LongToSize(max_ele_num); ++i) {
    auto idx_kernel_size = i % kernel_size.size();
    auto idx_padding = i % padding.size();
    if (MS_UNLIKELY(kernel_size.IsValueUnknown(idx_kernel_size) || padding.IsValueUnknown(idx_padding))) {
      continue;
    }
    const int kNum2 = 2;
    if (MS_UNLIKELY(kernel_size[idx_kernel_size] / kNum2 < padding[idx_padding])) {
      MS_EXCEPTION(ValueError) << "For " << primitive->name()
                               << ", pad should be at most half of kernel size, but got pad = " << padding[idx_padding]
                               << " and kernel_size = " << kernel_size[idx_kernel_size];
    }
  }

  return OP_CHECK_SUCCESS;
}
}  // namespace ops
}  // namespace mindspore
