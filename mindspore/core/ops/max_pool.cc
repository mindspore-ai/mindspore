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

#include "ops/max_pool.h"
#include <string>
#include <algorithm>
#include <memory>
#include <set>
#include <vector>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"

namespace mindspore {
namespace ops {
void MaxPool::set_pad_mode(const PadMode &pad_mode) {
  int64_t swi = pad_mode;
  (void)this->AddAttr(kPadMode, MakeValue(swi));
}

PadMode MaxPool::get_pad_mode() const { return PadMode(GetValue<int64_t>(GetAttr(kPadMode))); }
void MaxPool::set_kernel_size(const std::vector<int64_t> &kernel_size) {
  (void)this->AddAttr(kKernelSize,
                      MakeValue(CheckAndConvertUtils::CheckPositiveVector(kKernelSize, kernel_size, this->name())));
}

std::vector<int64_t> MaxPool::get_kernel_size() const { return GetValue<std::vector<int64_t>>(GetAttr(kKernelSize)); }
void MaxPool::set_strides(const std::vector<int64_t> &strides) {
  (void)this->AddAttr(kStrides, MakeValue(CheckAndConvertUtils::CheckPositiveVector(kStrides, strides, this->name())));
}

std::vector<int64_t> MaxPool::get_strides() const { return GetValue<std::vector<int64_t>>(GetAttr(kStrides)); }

void MaxPool::set_format(const Format &format) {
  int64_t f = format;
  (void)this->AddAttr(kFormat, MakeValue(f));
}

Format MaxPool::get_format() const { return Format(GetValue<int64_t>(GetAttr(kFormat))); }

void MaxPool::set_pad(const std::vector<int64_t> &pad) { (void)this->AddAttr(kPad, MakeValue(pad)); }

std::vector<int64_t> MaxPool::get_pad() const {
  auto value_ptr = GetAttr(kPad);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

void MaxPool::set_round_mode(const RoundMode &round_mode) {
  int64_t swi = round_mode;
  (void)this->AddAttr(kRoundMode, MakeValue(swi));
}

RoundMode MaxPool::get_round_mode() const {
  auto value_ptr = GetAttr(kRoundMode);
  return RoundMode(GetValue<int64_t>(value_ptr));
}

void MaxPool::Init(const std::vector<int64_t> &kernel_size, const std::vector<int64_t> &stride, const PadMode &pad_mode,
                   const Format &format, const std::vector<int64_t> &pad, const RoundMode &round_mode) {
  this->set_pad_mode(pad_mode);
  this->set_kernel_size(kernel_size);
  this->set_strides(stride);
  this->set_format(format);
  this->set_pad(pad);
  this->set_round_mode(round_mode);
}

namespace {
abstract::ShapePtr InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  MS_EXCEPTION_IF_NULL(input_args[0]);
  auto in_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->GetShapeTrack())[kShape];
  auto format = Format(GetValue<int64_t>(primitive->GetAttr(kFormat)));
  if (format == NHWC) {
    in_shape = {in_shape[0], in_shape[3], in_shape[1], in_shape[2]};
  }
  const int64_t x_rank = 4;
  (void)CheckAndConvertUtils::CheckInteger("x_rank", SizeToLong(in_shape.size()), kEqual, x_rank, op_name);

  auto kernel_size = GetValue<std::vector<int64_t>>(primitive->GetAttr(kKernelSize));
  auto pad_mode_value = (primitive->GetAttr(kPadMode));
  auto pad_mode = PadMode(GetValue<int64_t>(pad_mode_value));
  auto batch = in_shape[0];
  auto channel = in_shape[1];
  auto in_h = in_shape[2];
  auto in_w = in_shape[3];
  auto strides = GetValue<std::vector<int64_t>>(primitive->GetAttr(kStrides));
  auto kernel_h = kernel_size[2];
  auto kernel_w = kernel_size[3];
  auto stride_h = strides[2];
  auto stride_w = strides[3];
  int64_t out_h = abstract::Shape::SHP_ANY;
  int64_t out_w = abstract::Shape::SHP_ANY;
  if (pad_mode == VALID) {
    out_h = static_cast<int64_t>(ceil((in_h - (kernel_h - 1)) + static_cast<float>(stride_h) - 1) /
                                 static_cast<float>(stride_h));
    out_w = static_cast<int64_t>(ceil((in_w - (kernel_w - 1)) + static_cast<float>(stride_w) - 1) /
                                 static_cast<float>(stride_w));
  } else if (pad_mode == SAME) {
    out_h = static_cast<int64_t>(ceil(in_h / static_cast<float>(stride_h)));
    out_w = static_cast<int64_t>(ceil(in_w / static_cast<float>(stride_w)));
  }
  std::vector<int64_t> out_shape = {batch, channel, out_h, out_w};
  if (format == NHWC) {
    out_shape = {batch, out_h, out_w, channel};
  }
  if (std::any_of(out_shape.begin(), out_shape.end(), [](int64_t a) { return a <= 0; })) {
    MS_LOG(EXCEPTION) << "Kernel size is not valid.";
  }
  return std::make_shared<abstract::Shape>(out_shape);
}

TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  if (std::any_of(input_args.begin(), input_args.end(), [](const AbstractBasePtr arg) { return arg == nullptr; })) {
    MS_LOG(EXCEPTION) << "nullptr";
  }
  auto name = prim->name();
  MS_LOG(DEBUG) << "Infer data type for : " << name;
  auto input_type = input_args[0]->BuildType();
  MS_EXCEPTION_IF_NULL(input_type);
  auto input_tensor_type = input_type->cast<TensorTypePtr>();
  if (input_tensor_type == nullptr) {
    MS_LOG_EXCEPTION << "The maxpool's input must be a tensor but got " << input_type->ToString();
  }
  return input_tensor_type->element();
}
}  // namespace

AbstractBasePtr MaxPoolInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                             const std::vector<AbstractBasePtr> &input_args) {
  return std::make_shared<abstract::AbstractTensor>(InferType(primitive, input_args),
                                                    InferShape(primitive, input_args)->shape());
}
REGISTER_PRIMITIVE_C(kNameMaxPool, MaxPool);
}  // namespace ops
}  // namespace mindspore
