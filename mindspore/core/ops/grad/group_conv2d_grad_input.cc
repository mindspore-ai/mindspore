/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "ops/grad/group_conv2d_grad_input.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
void GroupConv2DGradInput::Init(const int64_t &in_channel, const int64_t &out_channel,
                                const std::vector<int64_t> &kernel_size, const PadMode &pad_mode,
                                const std::vector<int64_t> &pad_list, const std::vector<int64_t> &stride,
                                const std::vector<int64_t> &dilation, const int64_t &group,
                                const std::vector<int64_t> &input_shape, const Format &format,
                                const ActivationType &activation_type, const bool has_bias) {
  set_in_channel(in_channel);
  set_out_channel(out_channel);
  set_kernel_size(kernel_size);
  set_pad_mode(pad_mode);
  set_pad_list(pad_list);
  set_stride(stride);
  set_dilation(dilation);
  set_group(group);
  set_input_shape(input_shape);
  set_format(format);
  set_activation_type(activation_type);
  set_has_bias(has_bias);
}

void GroupConv2DGradInput::set_in_channel(const int64_t &in_channel) {
  (void)this->AddAttr(kInChannel, MakeValue(in_channel));
}

int64_t GroupConv2DGradInput::get_in_channel() const {
  auto value_ptr = GetAttr(kInChannel);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<int64_t>(value_ptr);
}

void GroupConv2DGradInput::set_out_channel(const int64_t &out_channel) {
  (void)this->AddAttr(kOutChannel, MakeValue(out_channel));
}

int64_t GroupConv2DGradInput::get_out_channel() const {
  auto value_ptr = GetAttr(kOutChannel);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<int64_t>(value_ptr);
}

void GroupConv2DGradInput::set_kernel_size(const std::vector<int64_t> &kernel_size) {
  (void)this->AddAttr(kKernelSize, MakeValue(kernel_size));
}

std::vector<int64_t> GroupConv2DGradInput::get_kernel_size() const {
  auto value_ptr = GetAttr(kKernelSize);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

void GroupConv2DGradInput::set_pad_mode(const PadMode &pad_mode) {
  int64_t swi = pad_mode;
  (void)this->AddAttr(kPadMode, MakeValue(swi));
}

PadMode GroupConv2DGradInput::get_pad_mode() const {
  auto value_ptr = GetAttr(kPadMode);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return PadMode(GetValue<int64_t>(value_ptr));
}

void GroupConv2DGradInput::set_pad_list(const std::vector<int64_t> &pad_list) {
  (void)this->AddAttr(kPadList, MakeValue(pad_list));
}

std::vector<int64_t> GroupConv2DGradInput::get_pad_list() const {
  auto value_ptr = GetAttr(kPadList);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

void GroupConv2DGradInput::set_stride(const std::vector<int64_t> &stride) {
  (void)this->AddAttr(kStride, MakeValue(stride));
}

std::vector<int64_t> GroupConv2DGradInput::get_stride() const {
  auto value_ptr = GetAttr(kStride);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

void GroupConv2DGradInput::set_dilation(const std::vector<int64_t> &dilation) {
  (void)this->AddAttr(kDilation, MakeValue(dilation));
}

std::vector<int64_t> GroupConv2DGradInput::get_dilation() const {
  auto value_ptr = GetAttr(kDilation);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

void GroupConv2DGradInput::set_group(const int64_t &group) { (void)this->AddAttr(kGroup, MakeValue(group)); }

int64_t GroupConv2DGradInput::get_group() const {
  auto value_ptr = GetAttr(kGroup);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<int64_t>(value_ptr);
}

void GroupConv2DGradInput::set_input_shape(const std::vector<int64_t> &input_shape) {
  (void)this->AddAttr(kInputShape, MakeValue(input_shape));
}

std::vector<int64_t> GroupConv2DGradInput::get_input_shape() const {
  auto value_ptr = GetAttr(kInputShape);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

void GroupConv2DGradInput::set_format(const Format &format) {
  int64_t swi = format;
  (void)this->AddAttr(kFormat, MakeValue(swi));
}

Format GroupConv2DGradInput::get_format() const {
  auto value_ptr = GetAttr(kFormat);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return Format(GetValue<int64_t>(value_ptr));
}

void GroupConv2DGradInput::set_activation_type(const ActivationType &activation_type) {
  int64_t swi = activation_type;
  (void)this->AddAttr(kActivationType, MakeValue(swi));
}

ActivationType GroupConv2DGradInput::get_activation_type() const {
  auto value_ptr = GetAttr(kActivationType);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return ActivationType(GetValue<int64_t>(value_ptr));
}

void GroupConv2DGradInput::set_has_bias(const bool has_bias) { (void)this->AddAttr(kHasBias, MakeValue(has_bias)); }

bool GroupConv2DGradInput::get_has_bias() const {
  auto value_ptr = GetAttr(kHasBias);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<bool>(value_ptr);
}
AbstractBasePtr GroupConv2DGradInputInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t input_num = 2;
  (void)CheckAndConvertUtils::CheckInteger("group_conv_2D_infer", SizeToLong(input_args.size()), kGreaterEqual,
                                           input_num, prim_name);
  MS_EXCEPTION_IF_NULL(input_args[0]);

  // Infer shape
  auto shape_ptr = primitive->GetAttr(kInputShape);
  MS_EXCEPTION_IF_NULL(shape_ptr);
  auto shape = GetValue<std::vector<int64_t>>(shape_ptr);

  // Infer type
  auto type_ptr = input_args[0]->BuildType();
  MS_EXCEPTION_IF_NULL(type_ptr);
  auto type_tensor_ptr = type_ptr->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(type_tensor_ptr);
  auto type = type_tensor_ptr->element();
  return std::make_shared<abstract::AbstractTensor>(type, shape);
}
REGISTER_PRIMITIVE_C(kNameGroupConv2DGradInput, GroupConv2DGradInput);
}  // namespace ops
}  // namespace mindspore
