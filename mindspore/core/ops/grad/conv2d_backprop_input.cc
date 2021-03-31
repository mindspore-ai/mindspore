/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include <map>
#include <vector>
#include <string>
#include <memory>

#include "ops/grad/conv2d_backprop_input.h"

namespace mindspore {
namespace ops {
void SetPadList(const PrimitivePtr &primitive, const std::vector<int64_t> &dout_shape_norm,
                const std::vector<int64_t> &x_size_v) {
  auto kernel_size = GetValue<std::vector<int64_t>>(primitive->GetAttr(kKernelSize));
  auto stride = GetValue<std::vector<int64_t>>(primitive->GetAttr(kStride));
  auto dilation = GetValue<std::vector<int64_t>>(primitive->GetAttr(kStride));
  auto pad_list = GetValue<std::vector<int64_t>>(primitive->GetAttr(kPadList));
  auto pad_mode = PadMode(GetValue<int64_t>(primitive->GetAttr(kPadMode)));
  if (std::all_of(pad_list.begin(), pad_list.end(), [](int64_t elem) -> bool { return elem != 0; })) {
    primitive->AddAttr(kPadList, MakeValue(pad_list));
  } else if (pad_mode == SAME) {
    auto stride_h = stride[0];
    auto stride_w = stride[1];
    auto kernel_h = kernel_size[0];
    auto kernel_w = kernel_size[1];
    auto dilation_h = dilation[2];
    auto dilation_w = dilation[3];
    auto pad_needed_h = (dout_shape_norm[2] - 1) * stride_h + dilation_h * (kernel_h - 1) + 1 - x_size_v[2];
    pad_needed_h = 0 > pad_needed_h ? 0 : pad_needed_h;
    auto pad_top = pad_needed_h / 2;
    auto pad_bottom = pad_needed_h - pad_top;
    auto pad_needed_w = (dout_shape_norm[3] - 1) * stride_w + dilation_w * (kernel_w - 1) + 1 - x_size_v[2];
    pad_needed_w = pad_needed_w > 0L ? pad_needed_w : 0L;
    auto pad_left = pad_needed_w / 2;
    auto pad_right = pad_needed_w - pad_left;
    pad_list = {pad_top, pad_bottom, pad_left, pad_right};
  } else if (pad_mode == PAD) {
    pad_list = GetValue<std::vector<int64_t>>(primitive->GetAttr(kPad));
  }
  primitive->AddAttr(kPadList, MakeValue(pad_list));
}
AbstractBasePtr Conv2DBackpropInputInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  CheckAndConvertUtils::CheckInteger("input number", input_args.size(), kEqual, 3, prim_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto doutput = input_args[0];
  auto x_size = input_args[2];
  auto x_size_value = x_size->GetValueTrack();
  MS_EXCEPTION_IF_NULL(x_size);
  auto x_size_v = GetValue<std::vector<int64_t>>(x_size_value);
  // infer dtype
  auto dtype = doutput->BuildType();
  if (!dtype->isa<TensorType>()) {
    MS_LOG(EXCEPTION) << "Conv2DBackpropInputInfer doutput must be tensor but got" << dtype->ToString();
  }
  auto input_tensor_type = dtype->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(input_tensor_type);
  auto element = input_tensor_type->element();
  // infer shape
  auto dout_shape = doutput->BuildShape();
  MS_EXCEPTION_IF_NULL(doutput);
  auto dout_shapeptr = dout_shape->cast<abstract::ShapePtr>();
  auto dout_shape_norm = dout_shapeptr->shape();
  SetPadList(primitive, dout_shape_norm, x_size_v);
  return std::make_shared<abstract::AbstractTensor>(element, std::make_shared<abstract::Shape>(x_size_v));
}

void Conv2DBackpropInput::Init(int64_t out_channel, const std::vector<int64_t> &kernel_size, int64_t mode,
                               const PadMode &pad_mode, const std::vector<int64_t> &pad,
                               const std::vector<int64_t> &stride, const std::vector<int64_t> &dilation, int64_t group,
                               const Format &format, const std::vector<int64_t> &pad_list) {
  set_out_channel(out_channel);
  set_kernel_size(kernel_size);
  set_mode(mode);
  set_pad_mode(pad_mode);
  set_pad(pad);
  set_stride(stride);
  set_dilation(dilation);
  set_group(group);
  set_format(format);
  set_pad_list(pad_list);
}

void Conv2DBackpropInput::set_out_channel(int64_t out_channel) {
  AddAttr(kOutChannel,
          MakeValue(CheckAndConvertUtils::CheckInteger(kOutChannel, out_channel, kGreaterThan, 0, name())));
}

void Conv2DBackpropInput::set_kernel_size(const std::vector<int64_t> &kernel_size) {
  AddAttr(kKernelSize, MakeValue(CheckAndConvertUtils::CheckPositiveVector(kKernelSize, kernel_size, name())));
}

void Conv2DBackpropInput::set_stride(const std::vector<int64_t> &stride) {
  AddAttr(kStride, MakeValue(CheckAndConvertUtils::CheckPositiveVector(kStride, stride, name(), true, true)));
}

void Conv2DBackpropInput::set_dilation(const std::vector<int64_t> &dilation) {
  AddAttr(kDilation, MakeValue(CheckAndConvertUtils::CheckPositiveVector(kDilation, dilation, name(), true, true)));
}

void Conv2DBackpropInput::set_pad_mode(const PadMode &pad_mode) {
  std::vector<int64_t> pad = get_pad();
  if (pad_mode == PAD) {
    for (auto item : pad) {
      CheckAndConvertUtils::Check(kPadItem, item, kGreaterEqual, "zeros_list", 0, name());
    }
  } else {
    CheckAndConvertUtils::Check(kPad, pad, kEqual, "zeros_list", {0, 0, 0, 0}, name());
  }
  int64_t swi = pad_mode;
  AddAttr(kPadMode, MakeValue(swi));
}

void Conv2DBackpropInput::set_pad(const std::vector<int64_t> &pad) {
  CheckAndConvertUtils::CheckInteger("pad_size", pad.size(), kEqual, 4, name());
  AddAttr(kPad, MakeValue(CheckAndConvertUtils::CheckPositiveVector(kPad, pad, name(), true, true)));
}

void Conv2DBackpropInput::set_mode(int64_t mode) {
  AddAttr(kMode, MakeValue(CheckAndConvertUtils::CheckInteger(kMode, mode, kEqual, 1, name())));
}

void Conv2DBackpropInput::set_group(int64_t group) {
  AddAttr(kGroup, MakeValue(CheckAndConvertUtils::CheckInteger(kGroup, group, kGreaterThan, 0, name())));
}

void Conv2DBackpropInput::set_format(const Format &format) {
  int64_t f = format;
  AddAttr(kFormat, MakeValue(f));
}

void Conv2DBackpropInput::set_pad_list(const std::vector<int64_t> &pad_list) {
  this->AddAttr(kPadList, MakeValue(pad_list));
}

int64_t Conv2DBackpropInput::get_out_channel() const {
  auto value_ptr = GetAttr(kOutChannel);
  return GetValue<int64_t>(value_ptr);
}

std::vector<int64_t> Conv2DBackpropInput::get_kernel_size() const {
  auto value_ptr = GetAttr(kKernelSize);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

std::vector<int64_t> Conv2DBackpropInput::get_stride() const {
  auto value_ptr = GetAttr(kStride);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

std::vector<int64_t> Conv2DBackpropInput::get_dilation() const {
  auto value_ptr = GetAttr(kDilation);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

PadMode Conv2DBackpropInput::get_pad_mode() const {
  auto value_ptr = GetAttr(kPadMode);
  return PadMode(GetValue<int64_t>(value_ptr));
}

std::vector<int64_t> Conv2DBackpropInput::get_pad() const {
  auto value_ptr = GetAttr(kPad);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

int64_t Conv2DBackpropInput::get_mode() const {
  auto value_ptr = GetAttr(kMode);
  return GetValue<int64_t>(value_ptr);
}

int64_t Conv2DBackpropInput::get_group() const {
  auto value_ptr = GetAttr(kGroup);
  return GetValue<int64_t>(value_ptr);
}

Format Conv2DBackpropInput::get_format() const {
  auto value_ptr = GetAttr(kFormat);
  return Format(GetValue<int64_t>(value_ptr));
}

std::vector<int64_t> Conv2DBackpropInput::get_pad_list() const {
  auto value_ptr = GetAttr(kPadList);
  return GetValue<std::vector<int64_t>>(value_ptr);
}
REGISTER_PRIMITIVE_C(kNameConv2DBackpropInput, Conv2DBackpropInput);
}  // namespace ops
}  // namespace mindspore
