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

#include "ops/depthwise_conv2d.h"
#include <string>
#include <memory>
#include <vector>
#include <algorithm>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"

namespace mindspore {
namespace ops {
void DepthWiseConv2D::Init(const int64_t channel_multiplier, const std::vector<int64_t> &kernel_size,
                           const int64_t mode, const PadMode &pad_mode, const std::vector<int64_t> &pad,
                           const std::vector<int64_t> &stride, const std::vector<int64_t> &dilation,
                           const int64_t group) {
  auto prim_name = this->name();
  this->set_format(NCHW);
  this->AddAttr("offset_a", MakeValue(0));
  this->set_mode(CheckAndConvertUtils::CheckInteger("mode", mode, kEqual, 3, prim_name));

  this->set_kernel_size(CheckAndConvertUtils::CheckPositiveVector(kKernelSize, kernel_size, prim_name));
  auto strides = CheckAndConvertUtils::CheckPositiveVector(kStride, stride, this->name(), false, false);
  if (strides[0] != strides[1]) {
    MS_EXCEPTION(ValueError) << "The height and width of stride should be equal, but got height " << strides[0]
                             << ", width " << strides[1];
  }
  this->set_stride(strides);
  auto dilations = CheckAndConvertUtils::CheckPositiveVector(kDilation, dilation, this->name(), false, false);
  if (dilations[0] != dilations[1]) {
    MS_EXCEPTION(ValueError) << "The height and width of dilation should be equal, but got height " << dilations[0]
                             << ", width " << dilations[1];
  }
  this->set_dilation(dilations);
  this->set_pad_mode(pad_mode);

  CheckAndConvertUtils::CheckInteger("pad_size", pad.size(), kEqual, 4, prim_name);
  if (pad_mode == PAD) {
    for (auto item : pad) {
      CheckAndConvertUtils::Check("pad_item", item, kGreaterEqual, "zeros_list", 0, prim_name);
    }
  } else {
    CheckAndConvertUtils::Check(kPad, pad, kEqual, "zeros_list", {0, 0, 0, 0}, prim_name);
  }
  this->set_pad(CheckAndConvertUtils::CheckPositiveVector(kPad, pad, this->name(), true, true));

  this->set_out_channel(
    CheckAndConvertUtils::CheckInteger("channel_multiplier", channel_multiplier, kGreaterThan, 0, prim_name));
  this->set_group(CheckAndConvertUtils::CheckInteger("group", group, kGreaterThan, 0, prim_name));
}

std::vector<int64_t> DepthWiseConv2D::get_kernel_size() const {
  auto value_ptr = GetAttr(kKernelSize);
  return GetValue<std::vector<int64_t>>(value_ptr);
}
std::vector<int64_t> DepthWiseConv2D::get_stride() const {
  auto value_ptr = GetAttr(kStride);
  return GetValue<std::vector<int64_t>>(value_ptr);
}
std::vector<int64_t> DepthWiseConv2D::get_dilation() const {
  auto value_ptr = GetAttr(kDilation);
  return GetValue<std::vector<int64_t>>(value_ptr);
}
PadMode DepthWiseConv2D::get_pad_mode() const {
  auto value_ptr = this->GetAttr(kPadMode);
  return PadMode(GetValue<int64_t>(value_ptr));
}
std::vector<int64_t> DepthWiseConv2D::get_pad() const {
  auto value_ptr = this->GetAttr(kPad);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

std::vector<int64_t> DepthWiseConv2D::get_pads() const {
  auto value_ptr = this->GetAttr(kPads);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

int64_t DepthWiseConv2D::get_mode() const {
  auto value_ptr = this->GetAttr(kMode);
  return GetValue<int64_t>(value_ptr);
}

int64_t DepthWiseConv2D::get_group() const {
  auto value_ptr = this->GetAttr(kGroup);
  return GetValue<int64_t>(value_ptr);
}
int64_t DepthWiseConv2D::get_out_channel() const {
  auto value_ptr = this->GetAttr(kOutChannel);
  return GetValue<int64_t>(value_ptr);
}

void DepthWiseConv2D::set_kernel_size(const std::vector<int64_t> &kernel_size) {
  this->AddAttr(kKernelSize, MakeValue(kernel_size));
}

void DepthWiseConv2D::set_stride(const std::vector<int64_t> &stride) { this->AddAttr(kStride, MakeValue(stride)); }
void DepthWiseConv2D::set_dilation(const std::vector<int64_t> &dilation) {
  this->AddAttr(kDilation, MakeValue(dilation));
}
void DepthWiseConv2D::set_pad_mode(const PadMode &pad_mode) {
  int64_t swi = pad_mode;
  this->AddAttr(kPadMode, MakeValue(swi));
}
void DepthWiseConv2D::set_pad(const std::vector<int64_t> &pad) { this->AddAttr(kPad, MakeValue(pad)); }
void DepthWiseConv2D::set_mode(const int64_t mode) { this->AddAttr(kMode, MakeValue(mode)); }
void DepthWiseConv2D::set_group(const int64_t group) { this->AddAttr(kGroup, MakeValue(group)); }
void DepthWiseConv2D::set_out_channel(const int64_t out_channel) { this->AddAttr(kOutChannel, MakeValue(out_channel)); }
void DepthWiseConv2D::set_pads(const std::vector<int64_t> &pad_list) { this->AddAttr(kPads, MakeValue(pad_list)); }
void DepthWiseConv2D::set_format(const Format &format) {
  int64_t f = format;
  this->AddAttr(kFormat, MakeValue(f));
}

Format DepthWiseConv2D::get_format() const {
  auto value_ptr = GetAttr(kFormat);
  return Format(GetValue<int64_t>(value_ptr));
}

abstract::ShapePtr DepthWiseConv2DInferShape(const PrimitivePtr &primitive,
                                             const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto conv_prim = primitive->cast<PrimDepthWiseConv2DPtr>();
  MS_EXCEPTION_IF_NULL(conv_prim);
  auto prim_name = conv_prim->name();
  CheckAndConvertUtils::CheckInRange<size_t>("conv2d_Infer", input_args.size(), kIncludeBoth, {2, 3}, prim_name);
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShape("x_shape", input_args[0]->GetShapeTrack(), prim_name);
  auto w_shape = CheckAndConvertUtils::ConvertShapePtrToShape("w_shape", input_args[1]->GetShapeTrack(), prim_name);
  if (conv_prim->get_format() == NHWC) {
    x_shape = {x_shape[0], x_shape[3], x_shape[1], x_shape[2]};
    w_shape = {w_shape[0], w_shape[3], w_shape[1], w_shape[2]};
  }
  CheckAndConvertUtils::CheckInteger("weight_rank", w_shape.size(), kEqual, 4, prim_name);
  CheckAndConvertUtils::CheckInteger("x_rank", x_shape.size(), kEqual, 4, prim_name);
  CheckAndConvertUtils::Check("x_shape[1]", x_shape[1], kEqual, "w_shape[1]", w_shape[1], conv_prim->name());
  auto out_channel = conv_prim->get_out_channel();

  std::vector<int64_t> temp_w;
  std::copy(w_shape.begin() + 2, w_shape.end(), std::back_inserter(temp_w));
  CheckAndConvertUtils::Check("kernel_size", conv_prim->get_kernel_size(), kEqual, "w_shape[2:4]", temp_w,
                              conv_prim->name());

  auto kernel_size_n = w_shape[0];
  if (kernel_size_n != 1) {
    MS_EXCEPTION(ValueError) << "The batch of input weeight should be 1, but got " << kernel_size_n;
  }
  auto kernel_size_h = w_shape[2];
  auto kernel_size_w = w_shape[3];
  auto stride = conv_prim->get_stride();
  auto dilation = conv_prim->get_dilation();
  auto stride_h = stride[2];
  auto stride_w = stride[3];
  auto dilation_h = dilation[2];
  auto dilation_w = dilation[3];
  int64_t h_out = -1;
  int64_t w_out = -1;
  std::vector<int64_t> pad_list(4, 0);
  auto pad_mode = conv_prim->get_pad_mode();
  if (pad_mode == VALID) {
    h_out = ceil((x_shape[2] - dilation_h * (kernel_size_h - 1)) / stride_h);
    w_out = ceil((x_shape[3] - dilation_w * (kernel_size_w - 1)) / stride_w);
  } else if (pad_mode == SAME) {
    h_out = ceil(x_shape[2] / stride_h);
    w_out = ceil(x_shape[3] / stride_w);

    auto pad_needed_h =
      std::max(static_cast<int64_t>(0), (h_out - 1) * stride_h + dilation_h * (kernel_size_h - 1) + 1 - x_shape[2]);
    pad_list.emplace_back(floor(pad_needed_h / 2));
    pad_list.emplace_back(pad_needed_h / 2);
    auto pad_needed_w =
      std::max(static_cast<int64_t>(0), (w_out - 1) * stride_w + dilation_w * (kernel_size_w - 1) + 1 - x_shape[3]);
    auto pad_left = floor(pad_needed_w / 2);
    pad_list.emplace_back(pad_left);
    pad_list.emplace_back(pad_needed_h - pad_left);
  } else if (pad_mode == PAD) {
    std::copy(conv_prim->get_pad().begin(), conv_prim->get_pad().end(), std::back_inserter(pad_list));
    auto pad_top = conv_prim->get_pad()[0];
    auto pad_bottom = conv_prim->get_pad()[1];
    auto pad_right = conv_prim->get_pad()[2];
    auto pad_left = conv_prim->get_pad()[3];

    h_out = 1 + (x_shape[2] + pad_top + pad_bottom - kernel_size_h - (kernel_size_h - 1) * (dilation_h - 1)) / stride_h;
    w_out = 1 + (x_shape[3] + pad_left + pad_right - kernel_size_w - (kernel_size_w - 1) * (dilation_w - 1)) / stride_w;
    h_out = floor(h_out);
    w_out = floor(w_out);
  }
  conv_prim->set_pads(pad_list);
  std::vector<int64_t> out_shape = {x_shape[0], out_channel * x_shape[1], h_out, w_out};
  if (conv_prim->get_format() == NHWC) {
    out_shape = {x_shape[0], h_out, w_out, out_channel * x_shape[1]};
  }
  return std::make_shared<abstract::Shape>(out_shape);
}

TypePtr DepthWiseConv2DInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  CheckAndConvertUtils::CheckInRange<size_t>("", input_args.size(), kIncludeBoth, {2, 3}, prim->name());
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }

  std::map<std::string, TypePtr> types;
  types.emplace("x", input_args[0]->BuildType());
  types.emplace("w", input_args[1]->BuildType());
  auto infer_type = CheckAndConvertUtils::CheckTensorTypeSame(types, common_valid_types, prim->name());
  if (infer_type == kNumberTypeInt8) {
    return std::make_shared<TensorType>(TypeIdToType(kNumberTypeInt32));
  }
  return TypeIdToType(infer_type);
}

AbstractBasePtr DepthWiseConv2DInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) {
  return std::make_shared<abstract::AbstractTensor>(DepthWiseConv2DInferType(primitive, input_args),
                                                    DepthWiseConv2DInferShape(primitive, input_args)->shape());
}
REGISTER_PRIMITIVE_C(kNameDepthWiseConv2D, DepthWiseConv2D);
}  // namespace ops
}  // namespace mindspore
