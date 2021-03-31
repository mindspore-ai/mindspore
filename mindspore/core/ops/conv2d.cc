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

#include "ops/conv2d.h"
#include <string>
#include <algorithm>
#include <memory>
#include <set>
#include <vector>
#include "ir/dtype/tensor_type.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"
#include "ops/control_depend.h"

namespace mindspore {
namespace ops {
namespace {
std::vector<int64_t> SetPadList(const PrimitivePtr &primitive, const std::vector<int64_t> &w_shape,
                                const std::vector<int64_t> &x_shape, const int64_t &out_channel) {
  auto kernel_size_h = w_shape[2];
  auto kernel_size_w = w_shape[3];
  auto stride = GetValue<std::vector<int64_t>>(primitive->GetAttr(kStride));
  auto dilation = GetValue<std::vector<int64_t>>(primitive->GetAttr(kDilation));
  auto stride_h = stride[2];
  auto stride_w = stride[3];
  auto dilation_h = dilation[2];
  auto dilation_w = dilation[3];
  int64_t h_out = -1;
  int64_t w_out = -1;
  std::vector<int64_t> pad_list(4, 0);
  auto pad_mode = PadMode(GetValue<int64_t>(primitive->GetAttr(kPadMode)));
  if (pad_mode == VALID) {
    h_out = ceil((x_shape[2] - dilation_h * (kernel_size_h - 1)) / stride_h);
    w_out = ceil((x_shape[3] - dilation_w * (kernel_size_w - 1)) / stride_w);
  } else if (pad_mode == SAME) {
    h_out = ceil(x_shape[2] / stride_h);
    w_out = ceil(x_shape[3] / stride_w);

    auto pad_needed_h =
      std::max(static_cast<int64_t>(0), (h_out - 1) * stride_h + dilation_h * (kernel_size_h - 1) + 1 - x_shape[2]);
    pad_list[0] = floor(pad_needed_h / 2);
    pad_list[1] = pad_needed_h / 2;
    auto pad_needed_w =
      std::max(static_cast<int64_t>(0), (w_out - 1) * stride_w + dilation_w * (kernel_size_w - 1) + 1 - x_shape[3]);
    auto pad_left = floor(pad_needed_w / 2);
    pad_list[2] = pad_left;
    pad_list[3] = pad_needed_h - pad_left;
  } else if (pad_mode == PAD) {
    auto pad = GetValue<std::vector<int64_t>>(primitive->GetAttr(kPad));
    std::copy(pad.begin(), pad.end(), std::back_inserter(pad_list));
    auto pad_top = pad[0];
    auto pad_bottom = pad[1];
    auto pad_right = pad[2];
    auto pad_left = pad[3];

    h_out = 1 + (x_shape[2] + pad_top + pad_bottom - kernel_size_h - (kernel_size_h - 1) * (dilation_h - 1)) / stride_h;
    w_out = 1 + (x_shape[3] + pad_left + pad_right - kernel_size_w - (kernel_size_w - 1) * (dilation_w - 1)) / stride_w;
    h_out = floor(h_out);
    w_out = floor(w_out);
  }
  CheckAndConvertUtils::CheckInteger("pad_size", pad_list.size(), kEqual, 4, primitive->name());
  primitive->AddAttr(kPadList, MakeValue(CheckAndConvertUtils::CheckPositiveVector(kPad, pad_list, primitive->name())));
  std::vector<int64_t> out_shape = {x_shape[0], out_channel, h_out, w_out};
  return out_shape;
}
abstract::ShapePtr Conv2dInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  CheckAndConvertUtils::CheckInRange<size_t>("conv2d_infer", input_args.size(), kIncludeBoth, {2, 3}, prim_name);
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShape("x_shape", input_args[0]->BuildShape(), prim_name);
  auto w_shape = CheckAndConvertUtils::ConvertShapePtrToShape("w_shape", input_args[1]->BuildShape(), prim_name);
  auto format = Format(GetValue<int64_t>(primitive->GetAttr(kFormat)));
  if (format == NHWC) {
    x_shape = {x_shape[0], x_shape[3], x_shape[1], x_shape[2]};
    w_shape = {w_shape[0], w_shape[3], w_shape[1], w_shape[2]};
  }
  CheckAndConvertUtils::CheckInteger("weight rank", w_shape.size(), kEqual, 4, prim_name);
  CheckAndConvertUtils::CheckInteger("x rank", x_shape.size(), kEqual, 4, prim_name);
  CheckAndConvertUtils::Check("x_shape[1] / group", x_shape[1] / GetValue<int64_t>(primitive->GetAttr(kGroup)), kEqual,
                              "w_shape[1]", w_shape[1], prim_name);
  auto out_channel = GetValue<int64_t>(primitive->GetAttr(kOutChannel));
  CheckAndConvertUtils::Check("out_channel", out_channel, kEqual, "w_shape[0]", w_shape[0], prim_name);
  std::vector<int64_t> temp_w;
  std::copy(w_shape.begin() + 2, w_shape.end(), std::back_inserter(temp_w));
  CheckAndConvertUtils::Check("kernel_size", GetValue<std::vector<int64_t>>(primitive->GetAttr(kKernelSize)), kEqual,
                              "w_shape[2:4]", temp_w, prim_name);
  auto out_shape = SetPadList(primitive, w_shape, x_shape, out_channel);
  if (format == NHWC) {
    out_shape = {out_shape[0], out_shape[3], out_shape[1], out_shape[2]};
  }
  return std::make_shared<abstract::Shape>(out_shape);
}

TypePtr Conv2dInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  CheckAndConvertUtils::CheckInRange<size_t>("", input_args.size(), kIncludeBoth, {2, 3}, prim->name());
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  const std::set<TypeId> valid_types = {kNumberTypeInt8, kNumberTypeInt32, kNumberTypeInt64, kNumberTypeFloat16,
                                        kNumberTypeFloat32};
  std::map<std::string, TypePtr> types;
  types.emplace("x", input_args[0]->BuildType());
  types.emplace("w", input_args[1]->BuildType());
  auto infer_type = CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, prim->name());
  if (infer_type == kNumberTypeInt8) {
    return TypeIdToType(kNumberTypeInt32);
  }
  return TypeIdToType(infer_type);
}
}  // namespace
void Conv2D::Init(int64_t out_channel, const std::vector<int64_t> &kernel_size, int64_t mode, const PadMode &pad_mode,
                  const std::vector<int64_t> &pad, const std::vector<int64_t> &stride,
                  const std::vector<int64_t> &dilation, int64_t group, const Format &format) {
  set_kernel_size(kernel_size);
  set_stride(stride);
  set_dilation(dilation);
  set_pad(pad);
  set_pad_mode(pad_mode);
  set_mode(mode);
  set_out_channel(out_channel);
  set_group(group);
  set_format(format);
}

void Conv2D::set_out_channel(int64_t out_channel) {
  AddAttr(kOutChannel,
          MakeValue(CheckAndConvertUtils::CheckInteger(kOutChannel, out_channel, kGreaterThan, 0, name())));
}

void Conv2D::set_kernel_size(const std::vector<int64_t> &kernel_size) {
  AddAttr(kKernelSize, MakeValue(CheckAndConvertUtils::CheckPositiveVector(kKernelSize, kernel_size, name())));
}

void Conv2D::set_stride(const std::vector<int64_t> &stride) {
  AddAttr(kStride, MakeValue(CheckAndConvertUtils::CheckPositiveVector(kStride, stride, name(), true, true)));
}

void Conv2D::set_dilation(const std::vector<int64_t> &dilation) {
  AddAttr(kDilation, MakeValue(CheckAndConvertUtils::CheckPositiveVector(kDilation, dilation, name(), true, true)));
}

void Conv2D::set_pad_mode(const PadMode &pad_mode) {
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

void Conv2D::set_pad(const std::vector<int64_t> &pad) {
  CheckAndConvertUtils::CheckInteger("pad_size", pad.size(), kEqual, 4, name());
  AddAttr(kPad, MakeValue(CheckAndConvertUtils::CheckPositiveVector(kPad, pad, name(), true, true)));
}

void Conv2D::set_mode(int64_t mode) {
  AddAttr(kMode, MakeValue(CheckAndConvertUtils::CheckInteger(kMode, mode, kEqual, 1, name())));
}

void Conv2D::set_group(int64_t group) {
  AddAttr(kGroup, MakeValue(CheckAndConvertUtils::CheckInteger(kGroup, group, kGreaterThan, 0, name())));
}

void Conv2D::set_format(const Format &format) {
  int64_t f = format;
  AddAttr(kFormat, MakeValue(f));
}

int64_t Conv2D::get_out_channel() const {
  auto value_ptr = GetAttr(kOutChannel);
  return GetValue<int64_t>(value_ptr);
}

std::vector<int64_t> Conv2D::get_kernel_size() const {
  auto value_ptr = GetAttr(kKernelSize);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

std::vector<int64_t> Conv2D::get_stride() const {
  auto value_ptr = GetAttr(kStride);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

std::vector<int64_t> Conv2D::get_dilation() const {
  auto value_ptr = GetAttr(kDilation);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

PadMode Conv2D::get_pad_mode() const {
  auto value_ptr = GetAttr(kPadMode);
  return PadMode(GetValue<int64_t>(value_ptr));
}

std::vector<int64_t> Conv2D::get_pad() const {
  auto value_ptr = GetAttr(kPad);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

int64_t Conv2D::get_mode() const {
  auto value_ptr = GetAttr(kMode);
  return GetValue<int64_t>(value_ptr);
}

int64_t Conv2D::get_group() const {
  auto value_ptr = GetAttr(kGroup);
  return GetValue<int64_t>(value_ptr);
}

Format Conv2D::get_format() const {
  auto value_ptr = GetAttr(kFormat);
  return Format(GetValue<int64_t>(value_ptr));
}

AbstractBasePtr Conv2dInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                            const std::vector<AbstractBasePtr> &input_args) {
  return std::make_shared<abstract::AbstractTensor>(Conv2dInferType(primitive, input_args),
                                                    Conv2dInferShape(primitive, input_args)->shape());
}
REGISTER_PRIMITIVE_C(kNameConv2D, Conv2D);
}  // namespace ops
}  // namespace mindspore
