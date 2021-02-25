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
  auto out_put = input_args[2]->BuildValue();
  auto infer_shape = GetValue<std::vector<int64_t>>(out_put);
  return std::make_shared<abstract::Shape>(infer_shape);
}

TypePtr Conv2DBackpropFilterInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  const std::set<TypeId> valid_types = {kNumberTypeInt8, kNumberTypeInt32, kNumberTypeFloat16, kNumberTypeFloat32};
  std::map<std::string, TypePtr> types;
  types.emplace("drotput", input_args[0]->BuildType());
  types.emplace("input_x", input_args[1]->BuildType());
  auto infer_type = CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, prim->name());
  return TypeIdToType(infer_type);
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
  this->AddAttr(kOutChannel, MakeValue(out_channel));
}

int64_t Conv2DBackpropFilter::get_out_channel() const {
  auto value_ptr = GetAttr(kOutChannel);
  return GetValue<int64_t>(value_ptr);
}

void Conv2DBackpropFilter::set_kernel_size(const std::vector<int64_t> &kernel_size) {
  this->AddAttr(kKernelSize, MakeValue(kernel_size));
}

std::vector<int64_t> Conv2DBackpropFilter::get_kernel_size() const {
  auto value_ptr = GetAttr(kKernelSize);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

void Conv2DBackpropFilter::set_pad_mode(const PadMode &pad_mode) {
  int64_t swi = pad_mode;
  this->AddAttr(kPadMode, MakeValue(swi));
}

PadMode Conv2DBackpropFilter::get_pad_mode() const {
  auto value_ptr = GetAttr(kPadMode);
  return PadMode(GetValue<int64_t>(value_ptr));
}

void Conv2DBackpropFilter::set_pad_list(const std::vector<int64_t> &pad_list) {
  this->AddAttr(kPadList, MakeValue(pad_list));
}

std::vector<int64_t> Conv2DBackpropFilter::get_pad_list() const {
  auto value_ptr = GetAttr(kPadList);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

void Conv2DBackpropFilter::set_mode(const int64_t mode) { this->AddAttr(kMode, MakeValue(mode)); }

int64_t Conv2DBackpropFilter::get_mode() const {
  auto value_ptr = GetAttr(kMode);
  return GetValue<int64_t>(value_ptr);
}

void Conv2DBackpropFilter::set_stride(const std::vector<int64_t> &stride) { this->AddAttr(kStride, MakeValue(stride)); }

std::vector<int64_t> Conv2DBackpropFilter::get_stride() const {
  auto value_ptr = GetAttr(kStride);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

void Conv2DBackpropFilter::set_dilation(const std::vector<int64_t> &dilation) {
  this->AddAttr(kDilation, MakeValue(dilation));
}

std::vector<int64_t> Conv2DBackpropFilter::get_dilation() const {
  auto value_ptr = GetAttr(kDilation);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

void Conv2DBackpropFilter::set_group(const int64_t group) { this->AddAttr(kGroup, MakeValue(group)); }

int64_t Conv2DBackpropFilter::get_group() const {
  auto value_ptr = GetAttr(kGroup);
  return GetValue<int64_t>(value_ptr);
}

void Conv2DBackpropFilter::set_format(const Format &format) {
  int64_t swi = format;
  this->AddAttr(kFormat, MakeValue(swi));
}

Format Conv2DBackpropFilter::get_format() const {
  auto value_ptr = GetAttr(kFormat);
  return Format(GetValue<int64_t>(value_ptr));
}

AbstractBasePtr Conv2DBackpropFilterInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args) {
  return std::make_shared<abstract::AbstractTensor>(Conv2DBackpropFilterInferType(primitive, input_args),
                                                    Conv2DBackpropFilterInferShape(primitive, input_args)->shape());
}
REGISTER_PRIMITIVE_C(kNameConv2DBackpropFilter, Conv2DBackpropFilter);
}  // namespace ops
}  // namespace mindspore
