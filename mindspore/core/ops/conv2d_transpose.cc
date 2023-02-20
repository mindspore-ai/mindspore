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

#include <vector>

#include "ops/conv2d_transpose.h"
#include "ops/grad/conv2d_backprop_input.h"
#include "utils/check_convert_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "base/base.h"
#include "ir/anf.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(Conv2DTranspose, BaseOperator);
void Conv2DTranspose::Init(int64_t in_channel, int64_t out_channel, const std::vector<int64_t> &kernel_size,
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

void Conv2DTranspose::set_in_channel(int64_t in_channel) {
  (void)AddAttr(kInChannel,
                api::MakeValue(CheckAndConvertUtils::CheckInteger(kInChannel, in_channel, kGreaterThan, 0, name())));
}

void Conv2DTranspose::set_out_channel(int64_t out_channel) {
  (void)AddAttr(kOutChannel,
                api::MakeValue(CheckAndConvertUtils::CheckInteger(kOutChannel, out_channel, kGreaterThan, 0, name())));
}

void Conv2DTranspose::set_kernel_size(const std::vector<int64_t> &kernel_size) {
  const int64_t kernel_len = 2;
  (void)CheckAndConvertUtils::CheckInteger(kKernelSize, SizeToLong(kernel_size.size()), kEqual, kernel_len, name());
  for (int64_t item : kernel_size) {
    (void)CheckAndConvertUtils::CheckInteger(kKernelSize, item, kGreaterEqual, 1, name());
  }
  (void)AddAttr(kKernelSize, api::MakeValue(kernel_size));
}

void Conv2DTranspose::set_stride(const std::vector<int64_t> &stride) {
  const int64_t stride_size = 2;
  (void)CheckAndConvertUtils::CheckInteger(kStride, SizeToLong(stride.size()), kEqual, stride_size, name());
  for (int64_t item : stride) {
    (void)CheckAndConvertUtils::CheckInteger(kStride, item, kGreaterEqual, 1, name());
  }
  (void)AddAttr(kStride, api::MakeValue(stride));
}

void Conv2DTranspose::set_dilation(const std::vector<int64_t> &dilation) {
  const int64_t dilation_size = 2;
  (void)CheckAndConvertUtils::CheckInteger(kDilation, SizeToLong(dilation.size()), kGreaterEqual, dilation_size,
                                           name());
  (void)AddAttr(kDilation, api::MakeValue(dilation));
}

void Conv2DTranspose::set_pad_mode(const PadMode &pad_mode) {
  std::vector<int64_t> pad = get_pad();
  if (pad_mode == PAD) {
    for (auto item : pad) {
      CheckAndConvertUtils::Check(kPadItem, item, kGreaterEqual, 0, name());
    }
  } else {
    CheckAndConvertUtils::Check(kPad, pad, kEqual, {0, 0, 0, 0}, name());
  }
  int64_t swi = pad_mode;
  (void)AddAttr(kPadMode, api::MakeValue(swi));
}

void Conv2DTranspose::set_pad(const std::vector<int64_t> &pad) {
  const int64_t pad_size = 4;
  (void)CheckAndConvertUtils::CheckInteger("pad_size", SizeToLong(pad.size()), kEqual, pad_size, name());
  (void)AddAttr(kPad, api::MakeValue(CheckAndConvertUtils::CheckPositiveVector(kPad, pad, name())));
}

void Conv2DTranspose::set_mode(int64_t mode) {
  (void)AddAttr(kMode, api::MakeValue(CheckAndConvertUtils::CheckInteger(kMode, mode, kEqual, 1, name())));
}

void Conv2DTranspose::set_group(int64_t group) {
  (void)AddAttr(kGroup, api::MakeValue(CheckAndConvertUtils::CheckInteger(kGroup, group, kGreaterThan, 0, name())));
}

void Conv2DTranspose::set_format(const Format &format) {
  int64_t f = format;
  (void)AddAttr(kFormat, api::MakeValue(f));
}

void Conv2DTranspose::set_pad_list(const std::vector<int64_t> &pad_list) {
  const int64_t pad_size = 4;
  (void)CheckAndConvertUtils::CheckInteger(kPadList, SizeToLong(pad_list.size()), kEqual, pad_size, name());
  (void)this->AddAttr(kPadList, api::MakeValue(pad_list));
}

int64_t Conv2DTranspose::get_in_channel() const {
  auto value_ptr = GetAttr(kInChannel);
  return GetValue<int64_t>(value_ptr);
}

int64_t Conv2DTranspose::get_out_channel() const {
  auto value_ptr = GetAttr(kOutChannel);
  return GetValue<int64_t>(value_ptr);
}

std::vector<int64_t> Conv2DTranspose::get_kernel_size() const {
  auto value_ptr = GetAttr(kKernelSize);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

std::vector<int64_t> Conv2DTranspose::get_stride() const {
  auto value_ptr = GetAttr(kStride);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

std::vector<int64_t> Conv2DTranspose::get_dilation() const {
  auto value_ptr = GetAttr(kDilation);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

PadMode Conv2DTranspose::get_pad_mode() const {
  auto value_ptr = GetAttr(kPadMode);
  return PadMode(GetValue<int64_t>(value_ptr));
}

std::vector<int64_t> Conv2DTranspose::get_pad() const {
  auto value_ptr = GetAttr(kPad);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

int64_t Conv2DTranspose::get_mode() const {
  auto value_ptr = GetAttr(kMode);
  return GetValue<int64_t>(value_ptr);
}

int64_t Conv2DTranspose::get_group() const {
  auto value_ptr = GetAttr(kGroup);
  return GetValue<int64_t>(value_ptr);
}

Format Conv2DTranspose::get_format() const {
  auto value_ptr = GetAttr(kFormat);
  return Format(GetValue<int64_t>(value_ptr));
}

std::vector<int64_t> Conv2DTranspose::get_pad_list() const {
  auto value_ptr = GetAttr(kPadList);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

class MIND_API Conv2DTransposeInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return Conv2DBackpropInputInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return Conv2DBackpropInputInferType(primitive, input_args);
  }

  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return Conv2DBackpropInputInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(Conv2DTranspose, prim::kPrimConv2DTranspose, Conv2DTransposeInfer, false);
}  // namespace ops
}  // namespace mindspore
