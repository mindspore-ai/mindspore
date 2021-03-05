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
#include <set>

#include "ops/conv2d_transpose.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr Conv2dTransposeInferShape(const PrimitivePtr &primitive,
                                             const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto conv2d_transpose_prim = primitive->cast<PrimConv2dTransposePtr>();
  MS_EXCEPTION_IF_NULL(conv2d_transpose_prim);
  auto prim_name = conv2d_transpose_prim->name();
  auto input_shape = CheckAndConvertUtils::ConvertShapePtrToShape("x_shape", input_args[3]->BuildShape(), prim_name);
  return std::make_shared<abstract::Shape>(input_shape);
}

TypePtr Conv2dTransposeInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  CheckAndConvertUtils::CheckInteger("conv2d_transpose_infer", input_args.size(), kEqual, 3, prim->name());
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  const std::set<TypeId> valid_types = {kNumberTypeInt8, kNumberTypeInt32, kNumberTypeFloat16, kNumberTypeFloat32};
  std::map<std::string, TypePtr> types;
  types.emplace("doutput_dtye", input_args[0]->BuildType());
  types.emplace("w_dtype", input_args[1]->BuildType());
  auto infer_type = CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, prim->name());
  return TypeIdToType(infer_type);
}
}  // namespace

void Conv2dTranspose::Init(int64_t in_channel, int64_t out_channel, const std::vector<int64_t> &kernel_size,
                           int64_t mode, const PadMode &pad_mode, const std::vector<int64_t> &pad,
                           const std::vector<int64_t> &stride, const std::vector<int64_t> &dilation, int64_t group,
                           const Format &format, const std::vector<int64_t> &pad_list) {
  set_in_channel(in_channel);
  set_out_channel(out_channel);
  set_kernel_size(kernel_size);
  set_mode(mode);
  set_pad(pad);
  set_pad_mode(pad_mode);
  set_stride(stride);
  set_dilation(dilation);
  set_group(group);
  set_format(format);
  set_pad_list(pad_list);
}

void Conv2dTranspose::set_in_channel(int64_t in_channel) {
  AddAttr(kInChannel, MakeValue(CheckAndConvertUtils::CheckInteger(kInChannel, in_channel, kGreaterThan, 0, name())));
}

void Conv2dTranspose::set_out_channel(int64_t out_channel) {
  AddAttr(kOutChannel,
          MakeValue(CheckAndConvertUtils::CheckInteger(kOutChannel, out_channel, kGreaterThan, 0, name())));
}

void Conv2dTranspose::set_kernel_size(const std::vector<int64_t> &kernel_size) {
  CheckAndConvertUtils::CheckInteger(kKernelSize, kernel_size.size(), kEqual, 2, name());
  for (int64_t item : kernel_size) {
    CheckAndConvertUtils::CheckInteger(kKernelSize, item, kGreaterEqual, 1, name());
  }
  AddAttr(kKernelSize, MakeValue(kernel_size));
}

void Conv2dTranspose::set_stride(const std::vector<int64_t> &stride) {
  CheckAndConvertUtils::CheckInteger(kStride, stride.size(), kEqual, 2, name());
  for (int64_t item : stride) {
    CheckAndConvertUtils::CheckInteger(kStride, item, kGreaterEqual, 1, name());
  }
  AddAttr(kStride, MakeValue(stride));
}

void Conv2dTranspose::set_dilation(const std::vector<int64_t> &dilation) {
  CheckAndConvertUtils::CheckInteger(kDilation, dilation.size(), kGreaterEqual, 2, name());
  AddAttr(kDilation, MakeValue(dilation));
}

void Conv2dTranspose::set_pad_mode(const PadMode &pad_mode) {
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

void Conv2dTranspose::set_pad(const std::vector<int64_t> &pad) {
  CheckAndConvertUtils::CheckInteger("pad_size", pad.size(), kEqual, 4, name());
  AddAttr(kPad, MakeValue(CheckAndConvertUtils::CheckPositiveVector(kPad, pad, name(), true, true)));
}

void Conv2dTranspose::set_mode(int64_t mode) {
  AddAttr(kMode, MakeValue(CheckAndConvertUtils::CheckInteger(kMode, mode, kEqual, 1, name())));
}

void Conv2dTranspose::set_group(int64_t group) {
  AddAttr(kGroup, MakeValue(CheckAndConvertUtils::CheckInteger(kGroup, group, kGreaterThan, 0, name())));
}

void Conv2dTranspose::set_format(const Format &format) {
  int64_t f = format;
  AddAttr(kFormat, MakeValue(f));
}

void Conv2dTranspose::set_pad_list(const std::vector<int64_t> &pad_list) {
  CheckAndConvertUtils::CheckInteger(kPadList, pad_list.size(), kEqual, 4, name());
  this->AddAttr(kPadList, MakeValue(pad_list));
}

int64_t Conv2dTranspose::get_in_channel() const {
  auto value_ptr = GetAttr(kInChannel);
  return GetValue<int64_t>(value_ptr);
}

int64_t Conv2dTranspose::get_out_channel() const {
  auto value_ptr = GetAttr(kOutChannel);
  return GetValue<int64_t>(value_ptr);
}

std::vector<int64_t> Conv2dTranspose::get_kernel_size() const {
  auto value_ptr = GetAttr(kKernelSize);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

std::vector<int64_t> Conv2dTranspose::get_stride() const {
  auto value_ptr = GetAttr(kStride);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

std::vector<int64_t> Conv2dTranspose::get_dilation() const {
  auto value_ptr = GetAttr(kDilation);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

PadMode Conv2dTranspose::get_pad_mode() const {
  auto value_ptr = GetAttr(kPadMode);
  return PadMode(GetValue<int64_t>(value_ptr));
}

std::vector<int64_t> Conv2dTranspose::get_pad() const {
  auto value_ptr = GetAttr(kPad);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

int64_t Conv2dTranspose::get_mode() const {
  auto value_ptr = GetAttr(kMode);
  return GetValue<int64_t>(value_ptr);
}

int64_t Conv2dTranspose::get_group() const {
  auto value_ptr = GetAttr(kGroup);
  return GetValue<int64_t>(value_ptr);
}

Format Conv2dTranspose::get_format() const {
  auto value_ptr = GetAttr(kFormat);
  return Format(GetValue<int64_t>(value_ptr));
}

std::vector<int64_t> Conv2dTranspose::get_pad_list() const {
  auto value_ptr = GetAttr(kPadList);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

AbstractBasePtr Conv2dTransposeInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) {
  return std::make_shared<abstract::AbstractTensor>(Conv2dTransposeInferType(primitive, input_args),
                                                    Conv2dTransposeInferShape(primitive, input_args)->shape());
}
REGISTER_PRIMITIVE_C(kNameConv2dTranspose, Conv2dTranspose);
}  // namespace ops
}  // namespace mindspore
