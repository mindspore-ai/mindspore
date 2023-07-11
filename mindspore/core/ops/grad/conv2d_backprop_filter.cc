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

#include <map>
#include <memory>
#include <set>

#include "mindapi/src/helper.h"
#include "mindspore/core/ops/conv_pool_ops.h"
#include "ops/grad/conv2d_backprop_filter.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
namespace {
using abstract::Shape;
constexpr size_t kConv2DBackpropFilterDoutIndex = 0;
constexpr size_t kConv2DBackpropFilterInputIndex = 1;
constexpr size_t kConv2DBackpropFilterSizeIndex = 2;
constexpr auto kCon2dPadSize = 4;

ShapeVector CalcPadListForSameMode(const ShapeVector &dout_shape_norm, const ShapeVector &x_size_v,
                                   const ShapeVector &kernel_size, const ShapeVector &stride,
                                   const ShapeVector &dilation) {
  ShapeVector pad_list(kCon2dPadSize, Shape::kShapeDimAny);
  if (IsDynamicRank(dout_shape_norm) || IsDynamicRank(x_size_v)) {
    return pad_list;
  }
  const auto stride_h = stride[kIndex2];
  const auto stride_w = stride[kIndex3];
  const auto kernel_h = kernel_size[kIndex0];
  const auto kernel_w = kernel_size[kIndex1];
  const auto dilation_h = dilation[kIndex2];
  const auto dilation_w = dilation[kIndex3];
  constexpr auto pad_divisor = 2;
  if (dout_shape_norm[kInputIndex2] != Shape::kShapeDimAny && x_size_v[kInputIndex2] != Shape::kShapeDimAny) {
    auto pad_needed_h =
      (dout_shape_norm[kInputIndex2] - 1) * stride_h + dilation_h * (kernel_h - 1) + 1 - x_size_v[kInputIndex2];
    pad_needed_h = 0 > pad_needed_h ? 0 : pad_needed_h;
    pad_list[kIndex0] = pad_needed_h / pad_divisor;
    pad_list[kIndex1] = pad_needed_h - pad_list[kIndex0];
  }
  if (dout_shape_norm[kInputIndex3] != Shape::kShapeDimAny && x_size_v[kInputIndex3] != Shape::kShapeDimAny) {
    auto pad_needed_w =
      (dout_shape_norm[kInputIndex3] - 1) * stride_w + dilation_w * (kernel_w - 1) + 1 - x_size_v[kInputIndex3];
    pad_needed_w = pad_needed_w > 0L ? pad_needed_w : 0L;
    pad_list[kIndex2] = pad_needed_w / pad_divisor;
    pad_list[kIndex3] = pad_needed_w - pad_list[kIndex2];
  }
  return pad_list;
}

void SetConv2dPadList(const PrimitivePtr &primitive, const std::vector<int64_t> &dout_shape_norm,
                      const std::vector<int64_t> &x_size_v) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  // check
  auto kernel_size =
    CheckAndConvertUtils::CheckIntOrTupleInt("attribute[kernel_size]", primitive->GetAttr(kKernelSize), prim_name);
  auto stride = CheckAndConvertUtils::CheckIntOrTupleInt("attribute[stride]", primitive->GetAttr(kStride), prim_name);
  auto dilation =
    CheckAndConvertUtils::CheckIntOrTupleInt("attribute[dilation]", primitive->GetAttr(kDilation), prim_name);

  auto attr_pad_list_prt = primitive->GetAttr(kPadList);
  MS_EXCEPTION_IF_NULL(attr_pad_list_prt);
  int64_t pad_mode_value;
  CheckAndConvertUtils::GetPadModEnumValue(primitive->GetAttr(kPadMode), &pad_mode_value, true);
  PadMode pad_mode{pad_mode_value};

  ShapeVector pad_list(kCon2dPadSize, Shape::kShapeDimAny);
  auto is_valid_pad_attr = [&attr_pad_list_prt]() -> bool {
    if (attr_pad_list_prt->isa<None>()) {
      return false;
    }
    auto attr_pad_list = GetValue<ShapeVector>(attr_pad_list_prt);
    return std::all_of(attr_pad_list.begin(), attr_pad_list.end(), [](int64_t val) { return val >= 0; });
  };
  if (is_valid_pad_attr()) {
    pad_list = GetValue<ShapeVector>(attr_pad_list_prt);
  } else if (pad_mode == VALID) {
    pad_list.assign(pad_list.size(), 0);
  } else if (pad_mode == SAME) {
    pad_list = CalcPadListForSameMode(dout_shape_norm, x_size_v, kernel_size, stride, dilation);
  } else if (pad_mode == PAD) {
    pad_list = GetValue<std::vector<int64_t>>(primitive->GetAttr(kPad));
  }
  (void)primitive->AddAttr(kPadList, MakeValue(pad_list));
}

abstract::ShapePtr Conv2DBackpropFilterInferShape(const PrimitivePtr &primitive,
                                                  const std::vector<AbstractBasePtr> &input_args) {
  auto filter_size = input_args[kConv2DBackpropFilterSizeIndex];
  auto out_shape = GetShapeValue(primitive, filter_size);
  auto ret_shape = std::make_shared<abstract::Shape>(out_shape);

  auto dout_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kConv2DBackpropFilterDoutIndex]->BuildShape())[kShape];
  auto input_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kConv2DBackpropFilterInputIndex]->BuildShape())[kShape];

  auto format = CheckAndConvertUtils::GetAndCheckFormat(primitive->GetAttr(kFormat));
  // normalize shape to NCHW format
  auto normalize_shape = [&format](const ShapeVector &shape) -> ShapeVector {
    if (format == static_cast<int64_t>(Format::NCHW)) {
      return shape;
    }
    // convert NHWC to NCHW format
    return {shape[0], shape[3], shape[1], shape[2]};
  };
  SetConv2dPadList(primitive, normalize_shape(dout_shape), normalize_shape(input_shape));

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
