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

#include <memory>

#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "ops/ops_func_impl/im2col_ext.h"
#include "ops/ops_func_impl/col2im_ext.h"
#include "ops/ops_func_impl/col2im_grad.h"
#include "ops/ops_func_impl/simple_infer.h"

namespace mindspore {
namespace ops {
namespace {
std::pair<int64_t, int64_t> Im2ColComputeOutputHeightAndWeight(const std::pair<int64_t, int64_t> &input_hw,
                                                               const std::vector<int64_t> &kernel_size,
                                                               const std::vector<int64_t> &dilation,
                                                               const std::vector<int64_t> &padding,
                                                               const std::vector<int64_t> &stride) {
  auto &[input_height, input_width] = input_hw;
  auto kernel_height = kernel_size[0];
  auto kernel_width = kernel_size[1];
  auto dilation_height = dilation[0];
  auto dilation_width = dilation[1];
  auto pad_height = padding[0];
  auto pad_width = padding[1];
  auto stride_height = stride[0];
  auto stride_width = stride[1];

  int64_t output_height =
    (input_height + 2 * pad_height - (dilation_height * (kernel_height - 1) + 1)) / stride_height + 1;
  int64_t output_width = (input_width + 2 * pad_width - (dilation_width * (kernel_width - 1) + 1)) / stride_width + 1;

  return std::make_pair(output_height, output_width);
}

std::vector<int64_t> Im2ColAndCol2ImCommonCheckArray(const PrimitivePtr &primitive, const ArrayValue<int64_t> &array,
                                                     const std::string &arg_name, size_t ele_num, int64_t min_value) {
  std::vector<int64_t> values(ele_num, abstract::Shape::kShapeDimAny);
  MS_CHECK_VALUE(array.size() == ele_num,
                 CheckAndConvertUtils::FormatCheckIntegerMsg("number of " + arg_name, SizeToLong(array.size()), kEqual,
                                                             SizeToLong(ele_num), primitive));
  for (size_t i = 0; i < array.size(); ++i) {
    if (MS_UNLIKELY(array.IsValueUnknown(i))) {
      continue;
    }
    MS_CHECK_VALUE(array[i] > min_value,
                   CheckAndConvertUtils::FormatCheckIntegerMsg(arg_name, array[i], kGreaterThan, min_value, primitive));
    values[i] = array[i];
  }
  return values;
}

std::pair<int32_t, std::vector<ArrayValue<int64_t>>> Im2ColAndCol2ImCommonCheckValidation(
  const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args,
  const std::vector<std::string> &arg_names, size_t ele_num, const std::vector<int64_t> &min_values, size_t start_idx) {
  assert((input_args.size() - start_idx) == arg_names.size());

  std::vector<ArrayValue<int64_t>> arrays;
  for (size_t i = 0; i < arg_names.size(); ++i) {
    auto array_opt = GetArrayValue<int64_t>(input_args[start_idx + i]);
    if (MS_UNLIKELY(!array_opt.has_value())) {
      return std::make_pair(OP_CHECK_RETRY, std::move(arrays));
    }
    auto array = std::move(array_opt.value());
    (void)Im2ColAndCol2ImCommonCheckArray(primitive, array, arg_names[i], ele_num, min_values[i]);
    arrays.emplace_back(std::move(array));
  }

  return std::make_pair(OP_CHECK_SUCCESS, std::move(arrays));
}

void Im2ColAndCol2ImCommonCheckShape(const PrimitivePtr &primitive, const std::vector<int64_t> &input_shape,
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

int32_t Im2ColAndCol2ImCommonCheckShape(const PrimitivePtr &primitive, const AbstractBasePtr &input_arg,
                                        size_t no_batch_rank, size_t batch_rank) {
  MS_EXCEPTION_IF_NULL(input_arg);
  const auto &input_shape = input_arg->GetShape()->GetShapeVector();
  if (MS_UNLIKELY(IsDynamicRank(input_shape))) {
    return OP_CHECK_RETRY;
  }
  Im2ColAndCol2ImCommonCheckShape(primitive, input_shape, no_batch_rank, batch_rank);
  return OP_CHECK_SUCCESS;
}

bool ArrayOptHasUnknownValue(const std::optional<ArrayValue<int64_t>> &array_opt) {
  if (MS_UNLIKELY(!array_opt.has_value())) {
    return true;
  }
  const auto &array = array_opt.value();
  return array.HasUnknownValue();
}

void Im2ColOutputLengthError(const PrimitivePtr &primitive, const std::vector<int64_t> &kernel_size,
                             const std::vector<int64_t> &dilation, const std::vector<int64_t> &padding,
                             const std::vector<int64_t> &stride, int64_t input_height, int64_t output_height,
                             int64_t input_width, int64_t output_width) {
  MS_EXCEPTION(ValueError) << "For " << primitive->name() << ", given input with spatial size (" << input_height << ", "
                           << input_width << "), kernel_size=(" << kernel_size[0] << ", " << kernel_size[1]
                           << "), dilation=(" << dilation[0] << ", " << dilation[1] << "), padding=(" << padding[0]
                           << ", " << padding[1] << "), calculated shape of the array of sliding blocks as ("
                           << output_height << ", " << output_width << "), which is too small (non-positive).";
}

void Col2ImCheckNInputPlane(const PrimitivePtr &primitive, int64_t n_input_plane,
                            const std::vector<int64_t> &kernel_size) {
  if (MS_UNLIKELY(n_input_plane % (kernel_size[0] * kernel_size[1]) != 0)) {
    MS_EXCEPTION(ValueError) << "For " << primitive->name()
                             << ", expected size of input's dimension 1 to be divisible by the product of "
                                "kernel_size, but got input.size(1)="
                             << n_input_plane << " and kernel_size=(" << kernel_size[0] << ", " << kernel_size[1]
                             << ").";
  }
}

void Col2ImCheckInputLength(const PrimitivePtr &primitive, const std::vector<int64_t> &output_size,
                            const std::vector<int64_t> &kernel_size, const std::vector<int64_t> &dilation,
                            const std::vector<int64_t> &padding, const std::vector<int64_t> &stride,
                            int64_t input_length) {
  auto [n_blocks_height, n_blocks_width] = Im2ColComputeOutputHeightAndWeight(
    std::make_pair(output_size[0], output_size[1]), kernel_size, dilation, padding, stride);

  if (MS_UNLIKELY(n_blocks_height < 1 || n_blocks_width < 1)) {
    MS_EXCEPTION(ValueError) << "For " << primitive->name() << ", given output_size=(" << output_size[0] << ", "
                             << output_size[1] << "), kernel_size=(" << kernel_size[0] << ", " << kernel_size[1]
                             << "), dilation=(" << dilation[0] << ", " << dilation[1] << "), padding=(" << padding[0]
                             << ", " << padding[1] << "), stride=(" << stride[0] << ", " << stride[1]
                             << "), calculated shape of the array of sliding blocks as (" << n_blocks_height << ", "
                             << n_blocks_width << "), which is too small (non-positive)";
  }

  if (MS_UNLIKELY(input_length != (n_blocks_height * n_blocks_width))) {
    MS_EXCEPTION(ValueError)
      << "For " << primitive->name() << ", given output_size=(" << output_size[0] << ", " << output_size[1]
      << "), kernel_size=(" << kernel_size[0] << ", " << kernel_size[1] << "), dilation=(" << dilation[0] << ", "
      << dilation[1] << "), padding=(" << padding[0] << ", " << padding[1] << "), stride=(" << stride[0] << ", "
      << stride[1] << "), expected size of the input's dimension 2 to match the calculated number of sliding blocks "
      << n_blocks_height << " * " << n_blocks_width << " = " << (n_blocks_height * n_blocks_width)
      << ", but got input.size(2)=" << input_length << ".";
  }
}
}  // namespace

ShapeArray Im2ColExtFuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &input_tensor = input_values[kIndex0]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(input_tensor);
  const auto &input_shape = input_tensor->shape();
  Im2ColAndCol2ImCommonCheckShape(primitive, input_shape, no_batch_rank_, batch_rank_);

  const size_t ele_num = 2;
  auto kernel_size_array = GetArrayValue<int64_t>(input_values[kIndex1]).value();
  auto kernel_size = Im2ColAndCol2ImCommonCheckArray(primitive, kernel_size_array, "kernel_size", ele_num, 0);
  auto dilation_array = GetArrayValue<int64_t>(input_values[kIndex2]).value();
  auto dilation = Im2ColAndCol2ImCommonCheckArray(primitive, dilation_array, "dilation", ele_num, 0);
  auto padding_array = GetArrayValue<int64_t>(input_values[kIndex3]).value();
  auto padding = Im2ColAndCol2ImCommonCheckArray(primitive, padding_array, "padding", ele_num, -1);
  auto stride_array = GetArrayValue<int64_t>(input_values[kIndex4]).value();
  auto stride = Im2ColAndCol2ImCommonCheckArray(primitive, stride_array, "stride", ele_num, 0);

  std::vector<int64_t> out_shape;
  auto input_rank = input_shape.size();
  if (input_rank == batch_rank_) {
    out_shape.push_back(input_shape[kIndex0]);
  }

  auto n_output_plane = input_shape[input_rank - kIndex3] * kernel_size[0] * kernel_size[1];
  out_shape.push_back(n_output_plane);

  auto input_height = input_shape[input_rank - kIndex2];
  auto input_width = input_shape[input_rank - kIndex1];
  auto [output_height, output_width] = Im2ColComputeOutputHeightAndWeight(std::make_pair(input_height, input_width),
                                                                          kernel_size, dilation, padding, stride);
  if (MS_UNLIKELY(output_height < 1 || output_width < 1)) {
    Im2ColOutputLengthError(primitive, kernel_size, dilation, padding, stride, input_height, output_height, input_width,
                            output_width);
  }
  auto output_length = output_height * output_width;
  out_shape.push_back(output_length);

  return {std::move(out_shape)};
}

TypePtrList Im2ColExtFuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &input_tensor = input_values[kIndex0]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(input_tensor);
  return {input_tensor->Dtype()};
}

BaseShapePtr Im2ColExtFuncImpl::InferShape(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) const {
  const auto &input_shape = input_args[0]->GetShape()->GetShapeVector();
  if (MS_UNLIKELY(IsDynamicRank(input_shape))) {
    return std::make_shared<abstract::TensorShape>(std::vector<int64_t>{abstract::TensorShape::kShapeRankAny});
  }

  auto is_dynamic_dim = [](int64_t dim_value) { return dim_value == abstract::TensorShape::kShapeDimAny; };
  auto input_rank = input_shape.size();
  std::vector<int64_t> out_shape;
  if (input_rank == batch_rank_) {
    out_shape.push_back(input_shape[kIndex0]);
  }

  auto channle_dim = input_shape[input_rank - kIndex3];
  auto kernel_size_opt = GetArrayValue<int64_t>(input_args[kIndex1]);
  auto is_kernel_unknown = ArrayOptHasUnknownValue(kernel_size_opt);
  if (MS_UNLIKELY(is_dynamic_dim(channle_dim) || is_kernel_unknown)) {
    out_shape.push_back(abstract::TensorShape::kShapeDimAny);
  } else {
    const auto &kernel_size = kernel_size_opt.value();
    auto n_output_plane = channle_dim * kernel_size[0] * kernel_size[1];
    out_shape.push_back(n_output_plane);
  }

  auto input_height = input_shape[input_rank - kIndex2];
  auto input_width = input_shape[input_rank - kIndex1];

  auto dilation_opt = GetArrayValue<int64_t>(input_args[kIndex2]);
  auto is_dilation_unknown = ArrayOptHasUnknownValue(dilation_opt);
  auto padding_opt = GetArrayValue<int64_t>(input_args[kIndex3]);
  auto is_padding_unknown = ArrayOptHasUnknownValue(padding_opt);
  auto stride_opt = GetArrayValue<int64_t>(input_args[kIndex4]);
  auto is_stride_unknown = ArrayOptHasUnknownValue(stride_opt);

  if (MS_UNLIKELY(is_dynamic_dim(input_height) || is_dynamic_dim(input_width) || is_kernel_unknown ||
                  is_padding_unknown || is_dilation_unknown || is_stride_unknown)) {
    out_shape.push_back(abstract::TensorShape::kShapeDimAny);
  } else {
    const auto &kernel_size = kernel_size_opt.value().ToVector();
    const auto &dilation = dilation_opt.value().ToVector();
    const auto &padding = padding_opt.value().ToVector();
    const auto &stride = stride_opt.value().ToVector();
    auto [output_height, output_width] = Im2ColComputeOutputHeightAndWeight(std::make_pair(input_height, input_width),
                                                                            kernel_size, dilation, padding, stride);
    if (MS_UNLIKELY(output_height < 1 || output_width < 1)) {
      Im2ColOutputLengthError(primitive, kernel_size, dilation, padding, stride, input_height, output_height,
                              input_width, output_width);
    }
    auto output_length = output_height * output_width;
    out_shape.push_back(output_length);
  }

  return std::make_shared<abstract::TensorShape>(std::move(out_shape));
}

TypePtr Im2ColExtFuncImpl::InferType(const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(input_args.at(0));
  auto input_type = input_args[0]->GetType();
  return input_type;
}

int32_t Im2ColExtFuncImpl::CheckValidation(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) const {
  auto check_shape = Im2ColAndCol2ImCommonCheckShape(primitive, input_args[kIndex0], no_batch_rank_, batch_rank_);

  const size_t ele_num = 2;
  static std::vector<std::string> arg_names{"kernel_size", "dilation", "padding", "stride"};
  static std::vector<int64_t> min_values{0, 0, -1, 0};
  auto check_pair =
    Im2ColAndCol2ImCommonCheckValidation(primitive, input_args, arg_names, ele_num, min_values, kIndex1);
  auto &check_args = check_pair.first;

  if (MS_UNLIKELY(check_shape == OP_CHECK_RETRY || check_args == OP_CHECK_RETRY)) {
    return OP_CHECK_RETRY;
  }
  return OP_CHECK_SUCCESS;
}

REGISTER_SIMPLE_INFER(kNameIm2ColExt, Im2ColExtFuncImpl)
REGISTER_SIMPLE_INFER(kNameCol2ImGrad, Col2ImGradFuncImpl)

BaseShapePtr Col2ImExtFuncImpl::InferShape(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) const {
  const auto &input_shape = input_args[0]->GetShape()->GetShapeVector();
  if (MS_UNLIKELY(IsDynamicRank(input_shape))) {
    return std::make_shared<abstract::TensorShape>(std::vector<int64_t>{abstract::TensorShape::kShapeRankAny});
  }

  std::vector<int64_t> out_shape;
  if (input_shape.size() == batch_rank_) {
    out_shape.push_back(input_shape[kIndex0]);
  }

  auto n_input_plane = input_shape[input_shape.size() - kIndex2];
  auto kernel_size_opt = GetArrayValue<int64_t>(input_args[kIndex2]);
  if (MS_UNLIKELY(n_input_plane == abstract::Shape::kShapeDimAny || ArrayOptHasUnknownValue(kernel_size_opt))) {
    out_shape.push_back(abstract::Shape::kShapeDimAny);
  } else {
    const auto &kernel_size = kernel_size_opt.value();
    out_shape.emplace_back(n_input_plane / (kernel_size[0] * kernel_size[1]));
  }

  auto output_size_opt = GetArrayValue<int64_t>(input_args[kIndex1]);
  if (MS_UNLIKELY(!output_size_opt.has_value())) {
    out_shape.insert(out_shape.end(), kIndex2, abstract::Shape::kShapeDimAny);
  } else {
    auto output_size = Im2ColAndCol2ImCommonCheckArray(primitive, output_size_opt.value(), "output_size", kIndex2, 0);
    out_shape.insert(out_shape.end(), output_size.begin(), output_size.end());
  }

  return std::make_shared<abstract::TensorShape>(std::move(out_shape));
}

TypePtr Col2ImExtFuncImpl::InferType(const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(input_args.at(0));
  auto input_type = input_args[0]->GetType();
  return input_type;
}

int32_t Col2ImExtFuncImpl::CheckValidation(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) const {
  auto check_shape = Im2ColAndCol2ImCommonCheckShape(primitive, input_args[kIndex0], no_batch_rank_, batch_rank_);
  if (MS_UNLIKELY(check_shape == OP_CHECK_RETRY)) {
    return OP_CHECK_RETRY;
  }

  const auto &input_shape = input_args[0]->GetShape()->GetShapeVector();
  auto n_input_plane = input_shape[input_shape.size() - kIndex2];
  if (MS_UNLIKELY(n_input_plane == abstract::TensorShape::kShapeDimAny)) {
    return OP_CHECK_RETRY;
  }

  auto kernel_size_opt = GetArrayValue<int64_t>(input_args[kIndex2]);
  if (MS_UNLIKELY(!kernel_size_opt.has_value())) {
    return OP_CHECK_RETRY;
  }

  const size_t ele_num = 2;
  const auto &kernel_size_array = kernel_size_opt.value();
  auto kernel_size = Im2ColAndCol2ImCommonCheckArray(primitive, kernel_size_array, "kernel_size", ele_num, 0);
  if (MS_UNLIKELY(kernel_size_array.HasUnknownValue())) {
    return OP_CHECK_RETRY;
  }

  Col2ImCheckNInputPlane(primitive, n_input_plane, kernel_size);

  auto input_length = input_shape[input_shape.size() - kIndex1];
  auto output_size_opt = GetArrayValue<int64_t>(input_args[kIndex1]);
  if (MS_UNLIKELY(input_length == abstract::TensorShape::kShapeDimAny || ArrayOptHasUnknownValue(output_size_opt))) {
    return OP_CHECK_RETRY;
  }

  const auto &output_size_array = output_size_opt.value();
  auto output_size = Im2ColAndCol2ImCommonCheckArray(primitive, output_size_array, "output_size", ele_num, 0);

  static std::vector<std::string> arg_names{"dilation", "padding", "stride"};
  static std::vector<int64_t> min_values{0, -1, 0};
  auto check_pair =
    Im2ColAndCol2ImCommonCheckValidation(primitive, input_args, arg_names, ele_num, min_values, kIndex3);
  auto &check_other_args = check_pair.first;
  if (MS_UNLIKELY(check_other_args != OP_CHECK_SUCCESS)) {
    return OP_CHECK_RETRY;
  }

  const auto &arrays = check_pair.second;
  const auto &dilation = arrays[kIndex0].ToVector();
  const auto &padding = arrays[kIndex1].ToVector();
  const auto &stride = arrays[kIndex2].ToVector();
  Col2ImCheckInputLength(primitive, output_size, kernel_size, dilation, padding, stride, input_length);

  return OP_CHECK_SUCCESS;
}

ShapeArray Col2ImExtFuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &input_tensor = input_values[kIndex0]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(input_tensor);
  const auto &input_shape = input_tensor->shape();
  Im2ColAndCol2ImCommonCheckShape(primitive, input_shape, no_batch_rank_, batch_rank_);

  auto input_rank = input_shape.size();
  std::vector<int64_t> out_shape;
  if (input_rank == batch_rank_) {
    out_shape.push_back(input_shape[kIndex0]);
  }

  const size_t ele_num = 2;
  auto output_size_array = GetArrayValue<int64_t>(input_values[kIndex1]).value();
  auto output_size = Im2ColAndCol2ImCommonCheckArray(primitive, output_size_array, "output_size", ele_num, 0);
  auto kernel_size_array = GetArrayValue<int64_t>(input_values[kIndex2]).value();
  auto kernel_size = Im2ColAndCol2ImCommonCheckArray(primitive, kernel_size_array, "kernel_size", ele_num, 0);
  auto dilation_array = GetArrayValue<int64_t>(input_values[kIndex3]).value();
  auto dilation = Im2ColAndCol2ImCommonCheckArray(primitive, dilation_array, "dilation", ele_num, 0);
  auto padding_array = GetArrayValue<int64_t>(input_values[kIndex4]).value();
  auto padding = Im2ColAndCol2ImCommonCheckArray(primitive, padding_array, "padding", ele_num, -1);
  auto stride_array = GetArrayValue<int64_t>(input_values[kIndex5]).value();
  auto stride = Im2ColAndCol2ImCommonCheckArray(primitive, stride_array, "stride", ele_num, 0);

  auto n_input_plane = input_shape[input_rank - kIndex2];
  Col2ImCheckNInputPlane(primitive, n_input_plane, kernel_size);
  out_shape.emplace_back(n_input_plane / (kernel_size[0] * kernel_size[1]));

  auto input_length = input_shape[input_rank - kIndex1];
  Col2ImCheckInputLength(primitive, output_size, kernel_size, dilation, padding, stride, input_length);
  out_shape.insert(out_shape.end(), output_size.begin(), output_size.end());

  return {std::move(out_shape)};
}

TypePtrList Col2ImExtFuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &input_tensor = input_values[kIndex0]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(input_tensor);
  return {input_tensor->Dtype()};
}

REGISTER_SIMPLE_INFER(kNameCol2ImExt, Col2ImExtFuncImpl)
}  // namespace ops
}  // namespace mindspore
