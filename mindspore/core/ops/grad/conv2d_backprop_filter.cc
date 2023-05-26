/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include <memory>

#include "ops/grad/conv2d_backprop_filter.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
constexpr size_t kConv2DBackpropFilterDoutIndex = 0;
constexpr size_t kConv2DBackpropFilterInputIndex = 1;
constexpr int64_t kNumOne = 1;
constexpr int64_t kNumTwo = 2;
constexpr int64_t kNumFour = 4;

void SetAttrPadList(const PrimitivePtr &primitive, const std::vector<int64_t> &dout_shape_norm,
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
  auto attr_pad_list_prt = primitive->GetAttr(kPadList);
  MS_EXCEPTION_IF_NULL(attr_pad_list_prt);
  int64_t pad_mode;
  CheckAndConvertUtils::GetPadModEnumValue(primitive->GetAttr(kPadMode), &pad_mode, true);

  ShapeVector pad_list = {0, 0, 0, 0};
  if (pad_mode == SAME) {
    int64_t stride_h, stride_w, kernel_h, kernel_w, dilation_h, dilation_w;
    if (stride.size() == kNumOne) {
      stride_h = stride[kIndex0];
      stride_w = stride[kIndex0];
    } else if (stride.size() == kNumTwo) {
      stride_h = stride[kIndex0];
      stride_w = stride[kIndex1];
    } else if (stride.size() == kNumFour) {
      stride_h = stride[kIndex2];
      stride_w = stride[kIndex3];
    } else {
      MS_EXCEPTION(ValueError) << "For primitive[" << prim_name << "], the size of stride must be 1, 2 or 4, but got "
                               << stride.size();
    }
    if (kernel_size.size() == kNumOne) {
      kernel_h = kernel_size[kIndex0];
      kernel_w = kernel_size[kIndex0];
    } else if (kernel_size.size() == kNumTwo) {
      kernel_h = kernel_size[kIndex0];
      kernel_w = kernel_size[kIndex1];
    } else {
      MS_EXCEPTION(ValueError) << "For primitive[" << prim_name << "], the size of kernel_size must be 1 or 2, but got "
                               << kernel_size.size();
    }
    if (dilation.size() == kNumOne) {
      dilation_h = dilation[kIndex0];
      dilation_w = dilation[kIndex0];
    } else if (dilation.size() == kNumTwo) {
      dilation_h = dilation[kIndex0];
      dilation_w = dilation[kIndex1];
    } else if (dilation.size() == kNumFour) {
      dilation_h = dilation[kIndex2];
      dilation_w = dilation[kIndex3];
    } else {
      MS_EXCEPTION(ValueError) << "For primitive[" << prim_name << "], the size of dilation must be 1, 2 or 4, but got "
                               << dilation.size();
    }

    int64_t pad_top = abstract::Shape::kShapeDimAny;
    int64_t pad_bottom = abstract::Shape::kShapeDimAny;
    int64_t pad_left = abstract::Shape::kShapeDimAny;
    int64_t pad_right = abstract::Shape::kShapeDimAny;
    if (dout_shape_norm[kInputIndex2] != abstract::Shape::kShapeDimAny &&
        x_size_v[kInputIndex2] != abstract::Shape::kShapeDimAny) {
      auto pad_needed_h =
        (dout_shape_norm[kInputIndex2] - 1) * stride_h + dilation_h * (kernel_h - 1) + 1 - x_size_v[kInputIndex2];
      pad_needed_h = 0 > pad_needed_h ? 0 : pad_needed_h;
      pad_top = pad_needed_h / kNumTwo;
      pad_bottom = pad_needed_h - pad_top;
    }
    if (dout_shape_norm[kInputIndex3] != abstract::Shape::kShapeDimAny &&
        x_size_v[kInputIndex3] != abstract::Shape::kShapeDimAny) {
      auto pad_needed_w =
        (dout_shape_norm[kInputIndex3] - 1) * stride_w + dilation_w * (kernel_w - 1) + 1 - x_size_v[kInputIndex3];
      pad_needed_w = pad_needed_w > 0L ? pad_needed_w : 0L;
      pad_left = pad_needed_w / kNumTwo;
      pad_right = pad_needed_w - pad_left;
    }
    pad_list = {pad_top, pad_bottom, pad_left, pad_right};
  } else if (pad_mode == PAD) {
    pad_list = GetValue<std::vector<int64_t>>(primitive->GetAttr(kPad));
  }
  (void)primitive->AddAttr(kPadList, MakeValue(pad_list));
}

abstract::ShapePtr Conv2DBackpropFilterInferShape(const PrimitivePtr &primitive,
                                                  const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto input_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kConv2DBackpropFilterInputIndex]->BuildShape())[kShape];
  auto dout_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kConv2DBackpropFilterDoutIndex]->BuildShape())[kShape];
  auto format = CheckAndConvertUtils::GetAndCheckFormat(primitive->GetAttr(kFormat));
  ShapeVector tmp_shape = {dout_shape[0], dout_shape[2], dout_shape[3], dout_shape[1]};
  auto dout_shape_norm = format == Format::NCHW ? dout_shape : tmp_shape;
  SetAttrPadList(primitive, dout_shape_norm, input_shape);

  constexpr size_t kFilterSizeIndex = 2;
  auto filter_size = input_args[kFilterSizeIndex];
  auto out_shape = GetShapeValue(primitive, filter_size);
  auto ret_shape = std::make_shared<abstract::Shape>(out_shape);
  return ret_shape;
}

TypePtr Conv2DBackpropFilterInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();
  // check
  std::map<std::string, TypePtr> types;
  (void)types.emplace("x", input_args[kConv2DBackpropFilterInputIndex]->BuildType());
  (void)types.emplace("doutput", input_args[kConv2DBackpropFilterDoutIndex]->BuildType());
  std::set<TypePtr> valid_x_type = {kInt8, kInt32, kFloat16, kFloat32};
  return CheckAndConvertUtils::CheckTensorTypeSame(types, valid_x_type, prim_name);
}
}  // namespace

MIND_API_OPERATOR_IMPL(Conv2DBackpropFilter, BaseOperator);
void Conv2DBackpropFilter::Init(const int64_t out_channel, const std::vector<int64_t> &kernel_size,
                                const PadMode &pad_mode, const std::vector<int64_t> &pad_list, const int64_t mode,
                                const std::vector<int64_t> &stride, const std::vector<int64_t> &dilation,
                                const int64_t group, const Format &format) {
  set_out_channel(out_channel);
  set_kernel_size(kernel_size);
  set_pad_mode(pad_mode);
  set_pad_list(pad_list);
  set_mode(mode);
  constexpr size_t kStride4dSize = 4;
  if (stride.size() == kStride4dSize) {
    set_stride({stride[2], stride[3]});
  } else {
    set_stride(stride);
  }
  set_dilation(dilation);
  set_group(group);
  set_format(format);
}

void Conv2DBackpropFilter::set_out_channel(const int64_t out_channel) {
  (void)this->AddAttr(kOutChannel, api::MakeValue(out_channel));
}

int64_t Conv2DBackpropFilter::get_out_channel() const {
  auto value_ptr = GetAttr(kOutChannel);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<int64_t>(value_ptr);
}

void Conv2DBackpropFilter::set_kernel_size(const std::vector<int64_t> &kernel_size) {
  (void)this->AddAttr(kKernelSize, api::MakeValue(kernel_size));
}

std::vector<int64_t> Conv2DBackpropFilter::get_kernel_size() const {
  auto value_ptr = GetAttr(kKernelSize);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

void Conv2DBackpropFilter::set_pad_mode(const PadMode &pad_mode) {
  int64_t swi = pad_mode;
  (void)this->AddAttr(kPadMode, api::MakeValue(swi));
}

PadMode Conv2DBackpropFilter::get_pad_mode() const {
  auto value_ptr = GetAttr(kPadMode);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return PadMode(GetValue<int64_t>(value_ptr));
}

void Conv2DBackpropFilter::set_pad_list(const std::vector<int64_t> &pad_list) {
  (void)this->AddAttr(kPadList, api::MakeValue(pad_list));
}

std::vector<int64_t> Conv2DBackpropFilter::get_pad_list() const {
  auto value_ptr = GetAttr(kPadList);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

void Conv2DBackpropFilter::set_mode(const int64_t mode) { (void)this->AddAttr(kMode, api::MakeValue(mode)); }

int64_t Conv2DBackpropFilter::get_mode() const {
  auto value_ptr = GetAttr(kMode);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<int64_t>(value_ptr);
}

void Conv2DBackpropFilter::set_stride(const std::vector<int64_t> &stride) {
  (void)this->AddAttr(kStride, api::MakeValue(stride));
}

std::vector<int64_t> Conv2DBackpropFilter::get_stride() const {
  auto value_ptr = GetAttr(kStride);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

void Conv2DBackpropFilter::set_dilation(const std::vector<int64_t> &dilation) {
  (void)this->AddAttr(kDilation, api::MakeValue(dilation));
}

std::vector<int64_t> Conv2DBackpropFilter::get_dilation() const {
  auto value_ptr = GetAttr(kDilation);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

void Conv2DBackpropFilter::set_group(const int64_t group) { (void)this->AddAttr(kGroup, api::MakeValue(group)); }

int64_t Conv2DBackpropFilter::get_group() const {
  auto value_ptr = GetAttr(kGroup);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<int64_t>(value_ptr);
}

void Conv2DBackpropFilter::set_format(const Format &format) {
  int64_t swi = format;
  (void)this->AddAttr(kFormat, api::MakeValue(swi));
}

Format Conv2DBackpropFilter::get_format() const {
  auto value_ptr = GetAttr(kFormat);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return Format(GetValue<int64_t>(value_ptr));
}

AbstractBasePtr Conv2DBackpropFilterInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t input_num = 3;
  (void)CheckAndConvertUtils::CheckInteger("input size", SizeToLong(input_args.size()), kGreaterEqual, input_num,
                                           prim_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto res = std::make_shared<abstract::AbstractTensor>(Conv2DBackpropFilterInferType(primitive, input_args),
                                                        Conv2DBackpropFilterInferShape(primitive, input_args));
  return res;
}

// AG means auto generated
class MIND_API AGConv2DBackpropFilterInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return Conv2DBackpropFilterInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return Conv2DBackpropFilterInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return Conv2DBackpropFilterInfer(engine, primitive, input_args);
  }
  std::set<int64_t> GetValueDependArgIndices() const override { return {2}; }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(Conv2DBackpropFilter, prim::kPrimConv2DBackpropFilter, AGConv2DBackpropFilterInfer,
                                 false);
}  // namespace ops
}  // namespace mindspore
