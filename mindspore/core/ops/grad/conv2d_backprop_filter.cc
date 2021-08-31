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

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr Conv2DBackpropFilterInferShape(const PrimitivePtr &primitive,
                                                  const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  // check
  auto w_size_v = input_args[kInputIndex2]->BuildValue();
  auto ret_shape = CheckAndConvertUtils::CheckAttrIntOrTupleInt("w_size", w_size_v, prim_name);
  return std::make_shared<abstract::Shape>(ret_shape);
}

TypePtr Conv2DBackpropFilterInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();
  // check
  std::map<std::string, TypePtr> types;
  (void)types.emplace("doutput", input_args[0]->BuildType());
  (void)types.emplace("x", input_args[1]->BuildType());
  std::set<TypePtr> valid_x_type = {kInt8, kInt32, kFloat16, kFloat32};
  return CheckAndConvertUtils::CheckTensorTypeSame(types, valid_x_type, prim_name);
}
}  // namespace

void Conv2DBackpropFilter::Init(const int64_t out_channel, const std::vector<int64_t> &kernel_size,
                                const PadMode &pad_mode, const std::vector<int64_t> &pad_list, const int64_t mode,
                                const std::vector<int64_t> &stride, const std::vector<int64_t> &dilation,
                                const int64_t group, const Format &format) {
  set_out_channel(out_channel);
  set_kernel_size(kernel_size);
  set_pad_mode(pad_mode);
  set_pad_list(pad_list);
  set_mode(mode);
  if (stride.size() == 4) {
    set_stride({stride[2], stride[3]});
  } else {
    set_stride(stride);
  }
  set_dilation(dilation);
  set_group(group);
  set_format(format);
}

void Conv2DBackpropFilter::set_out_channel(const int64_t out_channel) {
  (void)this->AddAttr(kOutChannel, MakeValue(out_channel));
}

int64_t Conv2DBackpropFilter::get_out_channel() const {
  auto value_ptr = GetAttr(kOutChannel);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<int64_t>(value_ptr);
}

void Conv2DBackpropFilter::set_kernel_size(const std::vector<int64_t> &kernel_size) {
  (void)this->AddAttr(kKernelSize, MakeValue(kernel_size));
}

std::vector<int64_t> Conv2DBackpropFilter::get_kernel_size() const {
  auto value_ptr = GetAttr(kKernelSize);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

void Conv2DBackpropFilter::set_pad_mode(const PadMode &pad_mode) {
  int64_t swi = pad_mode;
  (void)this->AddAttr(kPadMode, MakeValue(swi));
}

PadMode Conv2DBackpropFilter::get_pad_mode() const {
  auto value_ptr = GetAttr(kPadMode);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return PadMode(GetValue<int64_t>(value_ptr));
}

void Conv2DBackpropFilter::set_pad_list(const std::vector<int64_t> &pad_list) {
  (void)this->AddAttr(kPadList, MakeValue(pad_list));
}

std::vector<int64_t> Conv2DBackpropFilter::get_pad_list() const {
  auto value_ptr = GetAttr(kPadList);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

void Conv2DBackpropFilter::set_mode(const int64_t mode) { (void)this->AddAttr(kMode, MakeValue(mode)); }

int64_t Conv2DBackpropFilter::get_mode() const {
  auto value_ptr = GetAttr(kMode);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<int64_t>(value_ptr);
}

void Conv2DBackpropFilter::set_stride(const std::vector<int64_t> &stride) {
  (void)this->AddAttr(kStride, MakeValue(stride));
}

std::vector<int64_t> Conv2DBackpropFilter::get_stride() const {
  auto value_ptr = GetAttr(kStride);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

void Conv2DBackpropFilter::set_dilation(const std::vector<int64_t> &dilation) {
  (void)this->AddAttr(kDilation, MakeValue(dilation));
}

std::vector<int64_t> Conv2DBackpropFilter::get_dilation() const {
  auto value_ptr = GetAttr(kDilation);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

void Conv2DBackpropFilter::set_group(const int64_t group) { (void)this->AddAttr(kGroup, MakeValue(group)); }

int64_t Conv2DBackpropFilter::get_group() const {
  auto value_ptr = GetAttr(kGroup);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<int64_t>(value_ptr);
}

void Conv2DBackpropFilter::set_format(const Format &format) {
  int64_t swi = format;
  (void)this->AddAttr(kFormat, MakeValue(swi));
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
  return std::make_shared<abstract::AbstractTensor>(Conv2DBackpropFilterInferType(primitive, input_args),
                                                    Conv2DBackpropFilterInferShape(primitive, input_args));
}
REGISTER_PRIMITIVE_EVAL_IMPL(Conv2DBackpropFilter, prim::kPrimConv2DBackpropFilter, Conv2DBackpropFilterInfer, nullptr,
                             true);
}  // namespace ops
}  // namespace mindspore
