/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include <set>
#include <map>
#include <vector>
#include <string>
#include <memory>

#include "ops/grad/conv3d_backprop_input.h"
#include "ops/grad/conv3d_backprop_filter.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
constexpr size_t kConv3DBackpropInputFilterIndex = 0;
constexpr size_t kConv3DBackpropInputDoutIndex = 1;
constexpr size_t kConv3DBackpropInputSizeIndex = 2;
constexpr int64_t kConv3DBackpropInputPadSize = 6;
constexpr int64_t kConv3DBackpropInputStrideSize = 5;
constexpr int64_t kConv3DBackpropInputDilationSize = 5;
}  // namespace

MIND_API_OPERATOR_IMPL(Conv3DBackpropInput, BaseOperator);
void Conv3DBackpropInput::Init(int64_t out_channel, const std::vector<int64_t> &kernel_size, int64_t mode,
                               const PadMode &pad_mode, const std::vector<int64_t> &pad,
                               const std::vector<int64_t> &stride, const std::vector<int64_t> &dilation, int64_t group,
                               const Format &format) {
  set_out_channel(out_channel);
  set_kernel_size(kernel_size);
  set_mode(mode);
  set_pad_mode(pad_mode);
  set_pad(pad);
  set_pad_list(pad);
  set_stride(stride);
  set_dilation(dilation);
  set_group(group);
  set_format(format);
}

void Conv3DBackpropInput::set_out_channel(const int64_t out_channel) {
  (void)AddAttr(kOutChannel,
                api::MakeValue(CheckAndConvertUtils::CheckInteger(kOutChannel, out_channel, kGreaterThan, 0, name())));
}

int64_t Conv3DBackpropInput::get_out_channel() const {
  auto value_ptr = GetAttr(kOutChannel);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<int64_t>(value_ptr);
}

void Conv3DBackpropInput::set_kernel_size(const std::vector<int64_t> &kernel_size) {
  const int64_t kernel_len = 3;
  (void)CheckAndConvertUtils::CheckInteger(kKernelSize, SizeToLong(kernel_size.size()), kEqual, kernel_len, name());
  for (int64_t item : kernel_size) {
    (void)CheckAndConvertUtils::CheckInteger(kKernelSize, item, kGreaterEqual, 1, name());
  }
  (void)AddAttr(kKernelSize, api::MakeValue(kernel_size));
}

std::vector<int64_t> Conv3DBackpropInput::get_kernel_size() const {
  auto value_ptr = GetAttr(kKernelSize);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

void Conv3DBackpropInput::set_pad_mode(const PadMode &pad_mode) {
  std::vector<int64_t> pad = get_pad();
  if (pad_mode == PadMode::PAD) {
    for (auto item : pad) {
      CheckAndConvertUtils::Check(kPadItem, item, kGreaterEqual, 0, name());
    }
  } else {
    CheckAndConvertUtils::Check(kPad, pad, kEqual, {0, 0, 0, 0, 0, 0}, name());
  }
  (void)AddAttr(kPadMode, api::MakeValue(pad_mode));
}

PadMode Conv3DBackpropInput::get_pad_mode() const {
  auto value_ptr = GetAttr(kPadMode);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return PadMode(GetValue<int64_t>(value_ptr));
}

void Conv3DBackpropInput::set_pad(const std::vector<int64_t> &pad) {
  (void)CheckAndConvertUtils::CheckInteger("pad_size", SizeToLong(pad.size()), kEqual, kConv3DBackpropInputPadSize,
                                           name());
  (void)AddAttr(kPad, api::MakeValue(CheckAndConvertUtils::CheckPositiveVector(kPad, pad, name())));
}

std::vector<int64_t> Conv3DBackpropInput::get_pad() const {
  auto value_ptr = GetAttr(kPad);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

void Conv3DBackpropInput::set_pad_list(const std::vector<int64_t> &pad_list) {
  (void)this->AddAttr(kPadList, api::MakeValue(pad_list));
}

std::vector<int64_t> Conv3DBackpropInput::get_pad_list() const {
  auto value_ptr = GetAttr(kPadList);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

void Conv3DBackpropInput::set_mode(int64_t mode) {
  (void)AddAttr(kMode, api::MakeValue(CheckAndConvertUtils::CheckInteger(kMode, mode, kEqual, 1, name())));
}

int64_t Conv3DBackpropInput::get_mode() const {
  auto value_ptr = GetAttr(kMode);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<int64_t>(value_ptr);
}

void Conv3DBackpropInput::set_stride(const std::vector<int64_t> &stride) {
  (void)CheckAndConvertUtils::CheckInteger(kStrides, SizeToLong(stride.size()), kEqual, kConv3DBackpropInputStrideSize,
                                           name());
  for (int64_t item : stride) {
    (void)CheckAndConvertUtils::CheckInteger(kStrides, item, kGreaterEqual, 1, name());
  }
  (void)AddAttr(kStride, api::MakeValue(stride));
}

std::vector<int64_t> Conv3DBackpropInput::get_stride() const {
  auto value_ptr = GetAttr(kStride);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

void Conv3DBackpropInput::set_dilation(const std::vector<int64_t> &dilation) {
  (void)CheckAndConvertUtils::CheckInteger(kDilations, SizeToLong(dilation.size()), kGreaterEqual,
                                           kConv3DBackpropInputDilationSize, name());
  (void)AddAttr(kDilations, api::MakeValue(dilation));
}

std::vector<int64_t> Conv3DBackpropInput::get_dilation() const {
  auto value_ptr = GetAttr(kDilation);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

void Conv3DBackpropInput::set_group(int64_t group) {
  (void)AddAttr(kGroup, api::MakeValue(CheckAndConvertUtils::CheckInteger(kGroup, group, kGreaterThan, 0, name())));
}

int64_t Conv3DBackpropInput::get_group() const {
  auto value_ptr = GetAttr(kGroup);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<int64_t>(value_ptr);
}

void Conv3DBackpropInput::set_format(const Format &format) {
  int64_t f = format;
  (void)AddAttr(kFormat, api::MakeValue(CheckAndConvertUtils::CheckInteger(kFormat, f, kEqual, Format::NCDHW, name())));
}

Format Conv3DBackpropInput::get_format() const {
  auto value_ptr = GetAttr(kFormat);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return Format(GetValue<int64_t>(value_ptr));
}

class Conv3DBackpropInputInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto forward_input_shape = GetShapeValue(primitive, input_args[kConv3DBackpropInputSizeIndex]);
    auto dout_shape =
      CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kConv3DBackpropInputDoutIndex]->BuildShape())[kShape];
    auto filter_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(
      input_args[kConv3DBackpropInputFilterIndex]->BuildShape())[kShape];
    if (IsDynamicRank(forward_input_shape) || IsDynamicRank(dout_shape) || IsDynamicRank(filter_shape)) {
      forward_input_shape = {abstract::Shape::kShapeRankAny};
      return std::make_shared<abstract::Shape>(forward_input_shape);
    }
    SetConv3DBackpropPadList(primitive, dout_shape, forward_input_shape);
    return std::make_shared<abstract::Shape>(forward_input_shape);
  }

  TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(prim);
    auto prim_name = prim->name();
    // check
    std::map<std::string, TypePtr> types;
    auto doutput_type = input_args[kConv3DBackpropInputDoutIndex]->BuildType();
    (void)types.emplace("filter", input_args[kConv3DBackpropInputFilterIndex]->BuildType());
    (void)types.emplace("doutput", doutput_type);
    std::set<TypePtr> valid_x_type = {kFloat16, kFloat32};
    CheckAndConvertUtils::CheckTensorTypeSame(types, valid_x_type, prim_name);
    return doutput_type;
  }
  std::set<int64_t> GetValueDependArgIndices() const override { return {kConv3DBackpropInputSizeIndex}; }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(Conv3DBackpropInput, prim::kPrimConv3DBackpropInput, Conv3DBackpropInputInfer, false);
}  // namespace ops
}  // namespace mindspore
