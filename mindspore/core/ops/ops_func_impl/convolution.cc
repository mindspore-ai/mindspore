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

#include "ops/ops_func_impl/convolution.h"
#include <string>
#include <set>
#include "utils/check_convert_utils.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
namespace {
constexpr size_t expand_vec_size = 2;
constexpr size_t kConvolutionInputArgsSize = 9;
constexpr size_t kConvolutionInputDims = 4;
constexpr size_t kInputIdx = 0;
constexpr size_t kWightIdx = 1;
constexpr size_t kStrideIdx = 3;
constexpr size_t kPaddingIdx = 4;
constexpr size_t kDilationIdx = 5;
constexpr size_t kTransposedIdx = 6;
constexpr size_t kOutputPaddingIdx = 7;
constexpr size_t kGroupsIdx = 8;

int64_t GetOutputHW(const ShapeVector &input_shape, const ShapeVector &weight_shape, size_t shape_pos, size_t i,
                    const ArrayValue<int64_t> &stride, const ArrayValue<int64_t> &padding,
                    const ArrayValue<int64_t> &dilation, bool transposed, const ArrayValue<int64_t> &output_padding) {
  if (input_shape[shape_pos] == abstract::Shape::kShapeDimAny ||
      weight_shape[shape_pos] == abstract::Shape::kShapeDimAny || padding.IsValueUnknown(i) ||
      dilation.IsValueUnknown(i) || stride.IsValueUnknown(i)) {
    return abstract::Shape::kShapeDimAny;
  }

  if (!transposed) {
    return (input_shape[shape_pos] + 2 * padding[i] - dilation[i] * (weight_shape[shape_pos] - 1) - 1) / stride[i] + 1;
  } else {
    if (output_padding.IsValueUnknown(i)) {
      return abstract::Shape::kShapeDimAny;
    }

    return (input_shape[shape_pos] - 1) * stride[i] - 2 * padding[i] + dilation[i] * (weight_shape[shape_pos] - 1) +
           output_padding[i] + 1;
  }
}
}  // namespace

BaseShapePtr ConvolutionFuncImpl::InferShape(const PrimitivePtr &primitive,
                                             const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  if (input_args.size() != kConvolutionInputArgsSize) {
    MS_LOG(EXCEPTION) << "input args size should be  " << kConvolutionInputArgsSize << ", but got "
                      << input_args.size();
  }

  auto input_shape_ptr = input_args[kInputIdx]->GetShape();
  auto weight_shape_ptr = input_args[kWightIdx]->GetShape();
  MS_EXCEPTION_IF_NULL(input_shape_ptr);
  MS_EXCEPTION_IF_NULL(weight_shape_ptr);
  const auto &input_shape = input_shape_ptr->GetShapeVector();
  const auto &weight_shape = weight_shape_ptr->GetShapeVector();

  if (IsDynamicRank(input_shape) || IsDynamicRank(weight_shape)) {
    std::vector<int64_t> output_shape = {abstract::Shape::kShapeRankAny};
    return std::make_shared<abstract::Shape>(output_shape);
  }

  // Support conv2d first
  if (input_shape.size() != kConvolutionInputDims || weight_shape.size() != kConvolutionInputDims) {
    MS_LOG(EXCEPTION) << "Input and weight shape size must be " << kConvolutionInputDims
                      << ", but got input_shape:" << input_shape << ", weight_shape:" << weight_shape;
  }

  int64_t N = input_shape[0];
  int64_t Co = abstract::Shape::kShapeDimAny;
  int64_t Ho = abstract::Shape::kShapeDimAny;
  int64_t Wo = abstract::Shape::kShapeDimAny;

  auto transposed_opt = GetScalarValue<bool>(input_args[kTransposedIdx]->BuildValue());
  if (!transposed_opt.has_value()) {
    // 'Co/Ho/Wo' is unknown, if transposed is any value
    auto output_shape = {N, Co, Ho, Wo};
    MS_LOG(DEBUG) << "transposed_opt has no value, output_shape:" << output_shape;
    return std::make_shared<abstract::Shape>(output_shape);
  }

  auto transposed = transposed_opt.value();
  if (transposed) {
    auto groups_opt = GetScalarValue<int64_t>(input_args[kGroupsIdx]->BuildValue());
    if (groups_opt.has_value() && weight_shape[1] != abstract::Shape::kShapeDimAny) {
      Co = weight_shape[1] * groups_opt.value();
    }
  } else {
    Co = weight_shape[0];
  }

  auto stride_opt = GetArrayValue<int64_t>(input_args[kStrideIdx]);
  auto padding_opt = GetArrayValue<int64_t>(input_args[kPaddingIdx]);
  auto dilation_opt = GetArrayValue<int64_t>(input_args[kDilationIdx]);
  auto output_padding_opt = GetArrayValue<int64_t>(input_args[kOutputPaddingIdx]);
  if (!stride_opt.has_value() || !padding_opt.has_value() || !dilation_opt.has_value() ||
      (transposed && !output_padding_opt.has_value())) {
    auto output_shape = {N, Co, Ho, Wo};
    MS_LOG(DEBUG) << "stride has_value:" << stride_opt.has_value() << ", paddind has_value:" << padding_opt.has_value()
                  << ", dilation has_value:" << dilation_opt.has_value()
                  << ", output_padding has_value:" << output_padding_opt.has_value()
                  << ", output_shape:" << output_shape;
    return std::make_shared<abstract::Shape>(output_shape);
  }

  const auto &stride = stride_opt.value();
  const auto &padding = padding_opt.value();
  const auto &dilation = dilation_opt.value();
  const auto &output_padding = output_padding_opt.value();

  constexpr size_t h_begin_pos = 2;  // 'NCHW', the pos of 'H' is 2
  constexpr size_t w_begin_pos = 3;  // 'NCHW', the pos of 'W' is 3
  Ho = GetOutputHW(input_shape, weight_shape, h_begin_pos, 0, stride, padding, dilation, transposed, output_padding);
  Wo = GetOutputHW(input_shape, weight_shape, w_begin_pos, 1, stride, padding, dilation, transposed, output_padding);
  auto output_shape = {N, Co, Ho, Wo};
  return std::make_shared<abstract::Shape>(output_shape);
}

TypePtr ConvolutionFuncImpl::InferType(const PrimitivePtr &primitive,
                                       const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  const std::set<TypePtr> valid_types = {kInt8, kInt32, kInt64, kFloat16, kFloat32, kBFloat16};
  auto out_type =
    CheckAndConvertUtils::CheckTypeValid("input", input_args[kInputIdx]->GetType(), valid_types, primitive->name());

  return out_type;
}
}  // namespace ops
}  // namespace mindspore
