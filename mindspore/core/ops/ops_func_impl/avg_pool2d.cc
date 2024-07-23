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

#include "ir/dtype/number.h"
#include "ops/auto_generate/gen_ops_name.h"
#include "ops/op_name.h"
#include "ops/op_utils.h"
#include "mindapi/base/types.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "ops/ops_func_impl/simple_infer.h"

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

void Avgpool2DCheckUnsupportedScenirao(bool ceil_mode, bool count_include_pad) {
  // There are bugs in aclnnAvgpool2d when count_include_pad and ceil_mode are both true,
  // so the scenirao is not supported now.
  if (ceil_mode && count_include_pad) {
    MS_EXCEPTION(ValueError)
      << "For AvgPool2D, the scenirao, where ceil_mode and count_include_pad are both true, had not been supported.";
  }
}

void Avgpool2DCheckPaddingAndKernelSize(const PrimitivePtr &primitive, size_t max_ele_num,
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

void Avgpool2DInferOutputShape(const PrimitivePtr &primitive, const std::vector<int64_t> &input_shape,
                               const ArrayValue<int64_t> &kernel_size, const ArrayValue<int64_t> &stride,
                               const ArrayValue<int64_t> &padding, std::vector<int64_t> *const output_shape,
                               bool ceil_mode) {
  const size_t input_rank = input_shape.size();
  for (size_t i = 0; i < kIndex2; ++i) {
    auto dim = input_rank - kIndex2 + i;
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
    MS_CHECK_VALUE(output_size > 0,
                   CheckAndConvertUtils::FormatCheckIntegerMsg("output_size", output_size, kGreaterThan, 0, primitive));

    (*output_shape)[dim] = output_size;
  }
}

void AvgPool2DCheckInputShape(const PrimitivePtr &primitive, const std::vector<int64_t> &input_shape,
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
  auto first_dim_after_batch = input_shape.size() == batch_rank ? kIndex1 : kIndex0;
  auto check_result =
    std::all_of(input_shape.begin() + first_dim_after_batch, input_shape.end(), ShapeElementCheckFunc);
  if (MS_UNLIKELY(!check_result)) {
    MS_EXCEPTION(ValueError)
      << "For " << primitive->name() << ", expected " << no_batch_rank << "D or " << batch_rank
      << "D (batch mode) tensor with possibly 0 batch size and other non-zero dimensions for input, but got "
      << input_shape;
  }
}

void AvgPool2dCheckDivisorOverride(const PrimitivePtr &primitive, int64_t divisor) {
  if (MS_UNLIKELY(divisor == 0)) {
    MS_EXCEPTION(ValueError) << "For " << primitive->name() << ", divisor should not be zero.";
  }
}
}  // namespace
BaseShapePtr AvgPool2DFuncImpl::InferShape(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) const {
  const auto &input_shape = input_args[kIndex0]->GetShape()->GetShapeVector();
  if (MS_UNLIKELY(IsDynamicRank(input_shape))) {
    return std::make_shared<abstract::TensorShape>(std::vector<int64_t>{abstract::TensorShape::kShapeRankAny});
  }

  AvgPool2DCheckInputShape(primitive, input_shape, no_batch_rank_, batch_rank_);
  auto input_rank = input_shape.size();
  std::vector<int64_t> output_shape(input_rank, abstract::TensorShape::kShapeDimAny);
  std::transform(input_shape.begin(), input_shape.begin() + input_rank - kIndex2, output_shape.begin(),
                 [](const int64_t v) { return v; });

  auto kernel_size_opt = GetArrayValue<int64_t>(input_args[kIndex1]);
  auto stride_opt = GetArrayValue<int64_t>(input_args[kIndex2]);
  auto padding_opt = GetArrayValue<int64_t>(input_args[kIndex3]);
  auto ceil_mode_opt = GetScalarValue<bool>(input_args[kIndex4]->GetValue());
  if (MS_LIKELY((ceil_mode_opt.has_value() && kernel_size_opt.has_value() && stride_opt.has_value() &&
                 padding_opt.has_value()))) {
    auto ceil_mode = ceil_mode_opt.value();
    const auto &kernel_size = kernel_size_opt.value();
    const auto &stride = stride_opt.value();
    const auto &padding = padding_opt.value();
    Avgpool2DInferOutputShape(primitive, input_shape, kernel_size, stride, padding, &output_shape, ceil_mode);
  }

  return std::make_shared<abstract::TensorShape>(std::move(output_shape));
}

TypePtr AvgPool2DFuncImpl::InferType(const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(input_args.at(kIndex0));
  auto input_type = input_args[kIndex0]->GetType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("input", input_type, valid_types_, primitive->name());
  return input_type;
}

int32_t AvgPool2DFuncImpl::CheckValidation(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) const {
  auto kernel_size_opt = GetArrayValue<int64_t>(input_args[kIndex1]);
  if (MS_UNLIKELY(!kernel_size_opt.has_value())) {
    return OP_CHECK_RETRY;
  }

  auto kernel_size = kernel_size_opt.value();
  AvgPool2DCheckTupleIntParam(primitive, kernel_size, tuple_min_ele_num_, tuple_max_ele_num_, 0, "kernel_size");

  auto stride_opt = GetArrayValue<int64_t>(input_args[kIndex2]);
  if (MS_UNLIKELY(!stride_opt.has_value())) {
    return OP_CHECK_RETRY;
  }

  auto stride = stride_opt.value();
  AvgPool2DCheckTupleIntParam(primitive, stride, tuple_min_ele_num_, tuple_max_ele_num_, 0, "stride");

  auto padding_opt = GetArrayValue<int64_t>(input_args[kIndex3]);
  if (MS_UNLIKELY(!padding_opt.has_value())) {
    return OP_CHECK_RETRY;
  }

  auto padding = padding_opt.value();
  AvgPool2DCheckTupleIntParam(primitive, padding, tuple_min_ele_num_, tuple_max_ele_num_, -1, "padding");
  Avgpool2DCheckPaddingAndKernelSize(primitive, LongToSize(tuple_max_ele_num_), kernel_size, padding);

  auto ceil_mode_opt = GetScalarValue<bool>(input_args[kIndex4]->GetValue());
  auto count_include_pad_opt = GetScalarValue<bool>(input_args[kIndex5]->GetValue());
  if (MS_UNLIKELY(!ceil_mode_opt.has_value() || !count_include_pad_opt.has_value())) {
    return OP_CHECK_RETRY;
  }
  Avgpool2DCheckUnsupportedScenirao(ceil_mode_opt.value(), count_include_pad_opt.value());

  if (input_args[kIndex6]->GetType()->type_id() != kMetaTypeNone) {
    auto divisor_opt = GetScalarValue<int64_t>(input_args[kIndex6]->GetValue());
    if (MS_UNLIKELY(!divisor_opt.has_value())) {
      return OP_CHECK_RETRY;
    }
    AvgPool2dCheckDivisorOverride(primitive, divisor_opt.value());
  }

  return OP_CHECK_SUCCESS;
}

ShapeArray AvgPool2DFuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &input = input_values[kIndex0]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(input);
  const auto &input_shape = input->shape();
  AvgPool2DCheckInputShape(primitive, input_shape, no_batch_rank_, batch_rank_);

  const auto &kernel_size_opt = GetArrayValue<int64_t>(input_values[kIndex1]);
  const auto &stride_opt = GetArrayValue<int64_t>(input_values[kIndex2]);
  const auto &padding_opt = GetArrayValue<int64_t>(input_values[kIndex3]);
  const auto &ceil_mode_opt = GetScalarValue<bool>(input_values[kIndex4]);
  const auto &count_include_pad_opt = GetScalarValue<bool>(input_values[kIndex5]);
  if (MS_UNLIKELY(!kernel_size_opt.has_value() || !stride_opt.has_value() || !padding_opt.has_value() ||
                  !ceil_mode_opt.has_value() || !count_include_pad_opt.has_value())) {
    MS_LOG(EXCEPTION) << "Something unexpected happened.";
  }

  const auto &kernel_size = kernel_size_opt.value();
  AvgPool2DCheckTupleIntParam(primitive, kernel_size, tuple_min_ele_num_, tuple_max_ele_num_, 0, "kernel_size");
  const auto &stride = stride_opt.value();
  AvgPool2DCheckTupleIntParam(primitive, stride, tuple_min_ele_num_, tuple_max_ele_num_, 0, "stride");
  const auto &padding = padding_opt.value();
  AvgPool2DCheckTupleIntParam(primitive, padding, tuple_min_ele_num_, tuple_max_ele_num_, -1, "padding");
  Avgpool2DCheckPaddingAndKernelSize(primitive, LongToSize(tuple_max_ele_num_), kernel_size, padding);
  auto ceil_mode = ceil_mode_opt.value();
  Avgpool2DCheckUnsupportedScenirao(ceil_mode, count_include_pad_opt.value());

  if (input_values[kIndex6] != mindspore::kNone) {
    auto divisor = GetValue<int64_t>(input_values[kIndex6]);
    AvgPool2dCheckDivisorOverride(primitive, divisor);
  }

  // infer output_shape
  auto input_rank = input_shape.size();
  std::vector<int64_t> output_shape(input_rank, abstract::TensorShape::kShapeDimAny);
  std::transform(input_shape.begin(), input_shape.begin() + input_rank - kIndex2, output_shape.begin(),
                 [](const int64_t v) { return v; });
  Avgpool2DInferOutputShape(primitive, input_shape, kernel_size, stride, padding, &output_shape, ceil_mode);

  return {std::move(output_shape)};
}

TypePtrList AvgPool2DFuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &input = input_values[kIndex0]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(input);
  auto input_type = input->Dtype();
  (void)CheckAndConvertUtils::CheckTypeValid("input", input_type, valid_types_, primitive->name());
  return {input_type};
}

REGISTER_SIMPLE_INFER(kNameAvgPool2D, AvgPool2DFuncImpl)
}  // namespace ops
}  // namespace mindspore
