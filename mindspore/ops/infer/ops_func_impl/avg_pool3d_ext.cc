/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "infer/ops_func_impl/avg_pool3d_ext.h"
#include <memory>
#include <string>
#include <algorithm>
#include <utility>
#include "primitive/op_name.h"
#include "utils/check_convert_utils.h"
#include "mindspore/ops/ops_utils/op_utils.h"

namespace mindspore {
namespace ops {

static inline int64_t AvgPool3DExtCeilDiv(int64_t x, int64_t y) {
  auto z = DoubleToLong(x * 1.0 / y);
  return z;
}

static inline int64_t AvgPool3DExtOutputShapePadLR(int64_t input_size, int64_t kernel_size, int64_t pad, int64_t stride,
                                                   bool ceil_mode) {
  auto output_size = AvgPool3DExtCeilDiv(input_size + 2 * pad - kernel_size + (ceil_mode ? stride - 1 : 0), stride) + 1;
  if (ceil_mode) {
    // ensure that the last pooling starts inside the image needed to avoid problems in ceil mode
    if ((output_size - 1) * stride >= input_size + pad) {
      --output_size;
    }
  }
  return output_size;
}

static inline void AvgPool3DExtCheckTupleIntParam(const PrimitivePtr &primitive,
                                                  const ArrayValue<int64_t> &sequence_array, const int64_t min_ele_num,
                                                  const int64_t max_ele_num, const int64_t compare_value,
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

void AvgPool3DExtCheckPaddingAndKernelSize(const PrimitivePtr &primitive, size_t max_ele_num,
                                           const ArrayValue<int64_t> &kernel_size, const ArrayValue<int64_t> &padding) {
  const int64_t kNum2 = 2;
  for (size_t i = 0; i < max_ele_num; ++i) {
    auto idx_kernel_size = i % kernel_size.size();
    auto idx_padding = i % padding.size();
    if (MS_UNLIKELY(kernel_size.IsValueUnknown(idx_kernel_size) || padding.IsValueUnknown(idx_padding))) {
      continue;
    }
    if (MS_UNLIKELY(kernel_size[idx_kernel_size] / kNum2 < padding[idx_padding])) {
      MS_EXCEPTION(ValueError) << "For " << primitive->name()
                               << ", pad should be at most half of kernel size, but got pad = " << padding[idx_padding]
                               << " and kernel_size = " << kernel_size[idx_kernel_size];
    }
  }
}

void AvgPool3DExtInferOutputShape(const PrimitivePtr &primitive, const std::vector<int64_t> &input_shape,
                                  const ArrayValue<int64_t> &kernel_size, const ArrayValue<int64_t> &stride,
                                  const ArrayValue<int64_t> &padding, std::vector<int64_t> *const output_shape,
                                  bool ceil_mode) {
  const size_t input_rank = input_shape.size();
  for (size_t i = 0; i < kInputIndex3; ++i) {
    auto dim = input_rank - kInputIndex3 + i;
    auto cur_dim_value = input_shape[dim];
    auto idx_kernel_size = i % kernel_size.size();
    auto idx_stride = i % stride.size();
    auto idx_padding = i % padding.size();
    if (MS_UNLIKELY(cur_dim_value == abstract::TensorShape::kShapeDimAny ||
                    kernel_size.IsValueUnknown(idx_kernel_size) || stride.IsValueUnknown(idx_stride) ||
                    padding.IsValueUnknown(idx_padding))) {
      continue;
    }

    auto output_size = AvgPool3DExtOutputShapePadLR(cur_dim_value, kernel_size[idx_kernel_size], padding[idx_padding],
                                                    stride[idx_stride], ceil_mode);
    MS_CHECK_VALUE(output_size > 0,
                   CheckAndConvertUtils::FormatCheckIntegerMsg("output_size", output_size, kGreaterThan, 0, primitive));

    (*output_shape)[dim] = output_size;
  }
}

void AvgPool3DExtCheckInputShape(const PrimitivePtr &primitive, const std::vector<int64_t> &input_shape,
                                 size_t no_batch_rank, size_t batch_rank) {
  auto input_rank = input_shape.size();
  MS_CHECK_VALUE(input_rank >= no_batch_rank && input_rank <= batch_rank,
                 CheckAndConvertUtils::FormatCheckInRangeMsg("input rank", input_rank, kIncludeBoth,
                                                             {no_batch_rank, batch_rank}, primitive));

  auto ShapeElementCheckFunc = [](int64_t dim_value) {
    if (dim_value != abstract::TensorShape::kShapeDimAny && dim_value <= 0) {
      return false;
    }
    return true;
  };
  auto first_dim_after_batch = input_shape.size() == batch_rank ? kInputIndex1 : kInputIndex0;
  auto check_result =
    std::all_of(input_shape.begin() + first_dim_after_batch, input_shape.end(), ShapeElementCheckFunc);
  if (MS_UNLIKELY(!check_result)) {
    MS_EXCEPTION(ValueError)
      << "For " << primitive->name() << ", expected " << no_batch_rank << "D or " << batch_rank
      << "D (batch mode) tensor with possibly 0 batch size and other non-zero dimensions for input, but got "
      << input_shape;
  }
}

void AvgPool3DExtCheckDivisorOverride(const PrimitivePtr &primitive, int64_t divisor) {
  if (MS_UNLIKELY(divisor == 0)) {
    MS_EXCEPTION(ValueError) << "For " << primitive->name() << ", divisor should not be zero.";
  }
}

ShapeArray AvgPool3DExtFuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  auto input_shape = input_infos[0]->GetShape();
  if (input_infos[0]->IsDynamicRank()) {
    return {ShapeVector{abstract::Shape::kShapeRankAny}};
  }
  AvgPool3DExtCheckInputShape(primitive, input_shape, no_batch_rank_, batch_rank_);
  auto input_rank = input_shape.size();
  std::vector<int64_t> output_shape(input_rank, abstract::TensorShape::kShapeDimAny);
  std::transform(input_shape.begin(), input_shape.begin() + input_rank - kInputIndex3, output_shape.begin(),
                 [](const int64_t v) { return v; });

  auto kernel_size_opt = input_infos[kInputIndex1]->GetArrayValue<int64_t>();
  auto stride_opt =
    !input_infos[kInputIndex2]->IsNone() ? input_infos[kInputIndex2]->GetArrayValue<int64_t>() : kernel_size_opt;
  auto padding_opt = input_infos[kInputIndex3]->GetArrayValue<int64_t>();
  auto ceil_mode_opt = input_infos[kInputIndex4]->GetScalarValue<bool>();
  const auto &count_include_pad_opt = input_infos[kInputIndex5]->GetScalarValue<bool>();
  if (MS_UNLIKELY(!kernel_size_opt.has_value() || !stride_opt.has_value() || !padding_opt.has_value() ||
                  !ceil_mode_opt.has_value() || !count_include_pad_opt.has_value())) {
    return {output_shape};
  }

  const auto &kernel_size = kernel_size_opt.value();
  AvgPool3DExtCheckTupleIntParam(primitive, kernel_size, tuple_min_ele_num_, tuple_max_ele_num_, 0, "kernel_size");
  const auto &stride = stride_opt.value();
  AvgPool3DExtCheckTupleIntParam(primitive, stride, tuple_min_ele_num_, tuple_max_ele_num_, 0, "stride");
  const auto &padding = padding_opt.value();
  AvgPool3DExtCheckTupleIntParam(primitive, padding, tuple_min_ele_num_, tuple_max_ele_num_, -1, "padding");
  AvgPool3DExtCheckPaddingAndKernelSize(primitive, LongToSize(tuple_max_ele_num_), kernel_size, padding);
  auto ceil_mode = ceil_mode_opt.value();

  if (!input_infos[kInputIndex6]->IsNone()) {
    auto divisor = input_infos[kInputIndex6]->GetScalarValue<int64_t>();
    AvgPool3DExtCheckDivisorOverride(primitive, divisor.value());
  }

  // infer output_shape
  AvgPool3DExtInferOutputShape(primitive, input_shape, kernel_size, stride, padding, &output_shape, ceil_mode);

  return {std::move(output_shape)};
}

std::vector<TypeId> AvgPool3DExtFuncImpl::InferType(const PrimitivePtr &primitive,
                                                    const InferInfoPtrList &input_infos) const {
  return {input_infos[0]->GetType()};
}
}  // namespace ops
}  // namespace mindspore
