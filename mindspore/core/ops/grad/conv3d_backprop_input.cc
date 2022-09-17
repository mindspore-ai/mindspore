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

#include <map>
#include <vector>
#include <string>
#include <memory>
#include <set>

#include "ops/grad/conv3d_backprop_input.h"
#include "utils/check_convert_utils.h"
#include "ops/op_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
constexpr size_t kConv3DBackpropInputDoutIndex = 0;
constexpr size_t kConv3DBackpropInputFilterIndex = 1;
constexpr size_t kConv3DBackpropInputSizeIndex = 2;
constexpr int kConv3DBackpropPadHalf = 2;

void SetConv3DBackpropInputPadList(const PrimitivePtr &primitive, const std::vector<int64_t> &dout_shape_norm,
                                   const std::vector<int64_t> &x_size_v) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  // check
  auto kernel_size =
    CheckAndConvertUtils::CheckIntOrTupleInt("attribute[kernel_size]", primitive->GetAttr(kKernelSize), prim_name);
  auto stride = CheckAndConvertUtils::CheckIntOrTupleInt("attribute[stride]", primitive->GetAttr(kStride), prim_name);
  auto dilation =
    CheckAndConvertUtils::CheckIntOrTupleInt("attribute[dilation]", primitive->GetAttr(kDilation), prim_name);
  // default pad mode is valid
  int64_t pad_mode;
  CheckAndConvertUtils::GetPadModEnumValue(primitive->GetAttr(kPadMode), &pad_mode, false);
  ShapeVector pad_list = {0, 0, 0, 0, 0, 0};
  auto attr_pad_list_prt = primitive->GetAttr(kPadList);
  if ((attr_pad_list_prt != nullptr) && (!attr_pad_list_prt->isa<None>())) {
    pad_list = GetValue<ShapeVector>(attr_pad_list_prt);
  } else if (pad_mode == SAME) {
    auto stride_d = stride[2];
    auto stride_h = stride[3];
    auto stride_w = stride[4];
    auto kernel_d = kernel_size[0];
    auto kernel_h = kernel_size[1];
    auto kernel_w = kernel_size[2];
    auto dilation_d = dilation[2];
    auto dilation_h = dilation[3];
    auto dilation_w = dilation[4];
    int64_t pad_head = abstract::Shape::SHP_ANY;
    int64_t pad_tail = abstract::Shape::SHP_ANY;
    int64_t pad_top = abstract::Shape::SHP_ANY;
    int64_t pad_bottom = abstract::Shape::SHP_ANY;
    int64_t pad_left = abstract::Shape::SHP_ANY;
    int64_t pad_right = abstract::Shape::SHP_ANY;
    if (dout_shape_norm[kInputIndex2] != abstract::Shape::SHP_ANY &&
        x_size_v[kInputIndex2] != abstract::Shape::SHP_ANY) {
      auto pad_needed_d =
        (dout_shape_norm[kInputIndex2] - 1) * stride_d + dilation_d * (kernel_d - 1) + 1 - x_size_v[kInputIndex2];
      pad_needed_d = 0 > pad_needed_d ? 0 : pad_needed_d;
      pad_head = pad_needed_d / kConv3DBackpropPadHalf;
      pad_tail = pad_needed_d - pad_head;
    }
    if (dout_shape_norm[kInputIndex3] != abstract::Shape::SHP_ANY &&
        x_size_v[kInputIndex3] != abstract::Shape::SHP_ANY) {
      auto pad_needed_h =
        (dout_shape_norm[kInputIndex3] - 1) * stride_h + dilation_h * (kernel_h - 1) + 1 - x_size_v[kInputIndex3];
      pad_needed_h = 0 > pad_needed_h ? 0 : pad_needed_h;
      pad_top = pad_needed_h / kConv3DBackpropPadHalf;
      pad_bottom = pad_needed_h - pad_top;
    }
    if (dout_shape_norm[kInputIndex4] != abstract::Shape::SHP_ANY &&
        x_size_v[kInputIndex4] != abstract::Shape::SHP_ANY) {
      auto pad_needed_w =
        (dout_shape_norm[kInputIndex4] - 1) * stride_w + dilation_w * (kernel_w - 1) + 1 - x_size_v[kInputIndex4];
      pad_needed_w = pad_needed_w > 0L ? pad_needed_w : 0L;
      pad_left = pad_needed_w / kConv3DBackpropPadHalf;
      pad_right = pad_needed_w - pad_left;
    }
    pad_list = {pad_head, pad_tail, pad_top, pad_bottom, pad_left, pad_right};
  } else if (pad_mode == PAD) {
    pad_list = GetValue<std::vector<int64_t>>(primitive->GetAttr(kPad));
  }
  (void)primitive->AddAttr(kPadList, MakeValue(pad_list));
}
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
  set_stride(stride);
  set_dilation(dilation);
  set_group(group);
  set_format(format);
}

void Conv3DBackpropInput::set_out_channel(int64_t out_channel) {
  (void)AddAttr(kOutChannel,
                api::MakeValue(CheckAndConvertUtils::CheckInteger(kOutChannel, out_channel, kGreaterThan, 0, name())));
}

void Conv3DBackpropInput::set_kernel_size(const std::vector<int64_t> &kernel_size) {
  (void)AddAttr(kKernelSize,
                api::MakeValue(CheckAndConvertUtils::CheckPositiveVector(kKernelSize, kernel_size, name())));
}

void Conv3DBackpropInput::set_stride(const std::vector<int64_t> &stride) {
  (void)AddAttr(kStride, api::MakeValue(CheckAndConvertUtils::CheckPositiveVector(kStride, stride, name())));
}

void Conv3DBackpropInput::set_dilation(const std::vector<int64_t> &dilation) {
  (void)AddAttr(kDilation, api::MakeValue(CheckAndConvertUtils::CheckPositiveVector(kDilation, dilation, name())));
}

void Conv3DBackpropInput::set_pad_mode(const PadMode &pad_mode) {
  std::vector<int64_t> pad = get_pad();
  if (pad_mode == PAD) {
    for (auto item : pad) {
      CheckAndConvertUtils::Check(kPadItem, item, kGreaterEqual, 0, name());
    }
  } else {
    CheckAndConvertUtils::Check(kPad, pad, kEqual, {0, 0, 0, 0, 0, 0}, name());
  }
  int64_t swi = pad_mode;
  (void)AddAttr(kPadMode, api::MakeValue(swi));
}

void Conv3DBackpropInput::set_pad(const std::vector<int64_t> &pad) {
  const int64_t pad_size = 6;
  (void)CheckAndConvertUtils::CheckInteger("pad_size", SizeToLong(pad.size()), kEqual, pad_size, name());
  (void)AddAttr(kPad, api::MakeValue(CheckAndConvertUtils::CheckPositiveVector(kPad, pad, name())));
}

void Conv3DBackpropInput::set_mode(int64_t mode) {
  (void)AddAttr(kMode, api::MakeValue(CheckAndConvertUtils::CheckInteger(kMode, mode, kEqual, 1, name())));
}

void Conv3DBackpropInput::set_group(int64_t group) {
  (void)AddAttr(kGroup, api::MakeValue(CheckAndConvertUtils::CheckInteger(kGroup, group, kGreaterThan, 0, name())));
}

void Conv3DBackpropInput::set_format(const Format &format) {
  (void)AddAttr(kFormat,
                api::MakeValue(CheckAndConvertUtils::CheckInteger(kFormat, format, kEqual, Format::NCDHW, name())));
}

void Conv3DBackpropInput::set_pad_list(const std::vector<int64_t> &pad_list) {
  (void)this->AddAttr(kPadList, api::MakeValue(pad_list));
}

int64_t Conv3DBackpropInput::get_out_channel() const {
  auto value_ptr = GetAttr(kOutChannel);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<int64_t>(value_ptr);
}

std::vector<int64_t> Conv3DBackpropInput::get_kernel_size() const {
  auto value_ptr = GetAttr(kKernelSize);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

std::vector<int64_t> Conv3DBackpropInput::get_stride() const {
  auto value_ptr = GetAttr(kStride);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

std::vector<int64_t> Conv3DBackpropInput::get_dilation() const {
  auto value_ptr = GetAttr(kDilation);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

PadMode Conv3DBackpropInput::get_pad_mode() const {
  auto value_ptr = GetAttr(kPadMode);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return PadMode(GetValue<int64_t>(value_ptr));
}

std::vector<int64_t> Conv3DBackpropInput::get_pad() const {
  auto value_ptr = GetAttr(kPad);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

int64_t Conv3DBackpropInput::get_mode() const {
  auto value_ptr = GetAttr(kMode);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<int64_t>(value_ptr);
}

int64_t Conv3DBackpropInput::get_group() const {
  auto value_ptr = GetAttr(kGroup);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<int64_t>(value_ptr);
}

Format Conv3DBackpropInput::get_format() const {
  auto value_ptr = GetAttr(kFormat);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return Format(GetValue<int64_t>(value_ptr));
}

std::vector<int64_t> Conv3DBackpropInput::get_pad_list() const {
  auto value_ptr = GetAttr(kPadList);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

class Conv3DBackpropInputInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto out_shape_v = GetShapeValue(primitive, input_args[kConv3DBackpropInputSizeIndex]);
    auto dout_shape =
      CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kConv3DBackpropInputDoutIndex]->BuildShape())[kShape];
    auto filter_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(
      input_args[kConv3DBackpropInputFilterIndex]->BuildShape())[kShape];
    if (IsDynamicRank(out_shape_v) || IsDynamicRank(dout_shape) || IsDynamicRank(filter_shape)) {
      out_shape_v = {UNKNOWN_RANK};
      return std::make_shared<abstract::Shape>(out_shape_v);
    }

    SetConv3DBackpropInputPadList(primitive, dout_shape, out_shape_v);
    auto ret_shape = std::make_shared<abstract::Shape>(out_shape_v);
    return ret_shape;
  }

  TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(prim);
    auto prim_name = prim->name();
    // check
    std::map<std::string, TypePtr> types;
    (void)types.emplace("filter", input_args[kConv3DBackpropInputFilterIndex]->BuildType());
    (void)types.emplace("doutput", input_args[kConv3DBackpropInputDoutIndex]->BuildType());
    std::set<TypePtr> valid_x_type = {kInt8, kInt32, kFloat16, kFloat32};
    return CheckAndConvertUtils::CheckTensorTypeSame(types, valid_x_type, prim_name);
  }
  std::set<int64_t> GetValueDependArgIndices() const override { return {kConv3DBackpropInputSizeIndex}; }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(Conv3DBackpropInput, prim::kPrimConv3DBackpropInput, Conv3DBackpropInputInfer, false);
}  // namespace ops
}  // namespace mindspore
