/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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
#include <string>
#include <vector>

#include "mindapi/src/helper.h"
#include "mindspore/core/ops/conv_pool_ops.h"
#include "ops/grad/conv3d_backprop_filter.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
namespace {
constexpr size_t kConv3DBackpropFilterInputIndex = 0;
constexpr size_t kConv3DBackpropFilterDoutIndex = 1;
constexpr size_t kConv3DBackpropFilterFilterSizeIndex = 2;
constexpr int64_t kConv3DBackpropFilterPadSize = 6;
constexpr int64_t kConv3DBackpropFilterStrideSize = 5;
constexpr int64_t kConv3DBackpropFilterDilationSize = 5;
constexpr int64_t kConv3DBackpropFilterArgsSizeTwo = 2;
constexpr int64_t kConv3DBackpropFilterArgsSizeThree = 3;

inline void Conv3dBackpropFilterInferCheck(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args, bool infer_shape) {
  MS_EXCEPTION_IF_NULL(primitive);
  for (auto item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  const int64_t input_num = infer_shape ? kConv3DBackpropFilterArgsSizeThree : kConv3DBackpropFilterArgsSizeTwo;
  (void)CheckAndConvertUtils::CheckInteger("Conv3dBackpropFilter infer check", SizeToLong(input_args.size()),
                                           kGreaterEqual, input_num, primitive->name());
}
}  // namespace

MIND_API_OPERATOR_IMPL(Conv3DBackpropFilter, BaseOperator);
void Conv3DBackpropFilter::Init(int64_t out_channel, const std::vector<int64_t> &kernel_size, int64_t mode,
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

void Conv3DBackpropFilter::set_out_channel(const int64_t out_channel) {
  (void)AddAttr(kOutChannel,
                api::MakeValue(CheckAndConvertUtils::CheckInteger(kOutChannel, out_channel, kGreaterThan, 0, name())));
}

int64_t Conv3DBackpropFilter::get_out_channel() const {
  auto value_ptr = GetAttr(kOutChannel);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<int64_t>(value_ptr);
}

void Conv3DBackpropFilter::set_kernel_size(const std::vector<int64_t> &kernel_size) {
  const int64_t kernel_len = 3;
  (void)CheckAndConvertUtils::CheckInteger(kKernelSize, SizeToLong(kernel_size.size()), kEqual, kernel_len, name());
  for (int64_t item : kernel_size) {
    (void)CheckAndConvertUtils::CheckInteger(kKernelSize, item, kGreaterEqual, 1, name());
  }
  (void)AddAttr(kKernelSize, api::MakeValue(kernel_size));
}

std::vector<int64_t> Conv3DBackpropFilter::get_kernel_size() const {
  auto value_ptr = GetAttr(kKernelSize);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

void Conv3DBackpropFilter::set_pad_mode(const PadMode &pad_mode) {
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

std::string Conv3DBackpropFilter::get_pad_mode() const {
  auto value_ptr = GetAttr(kPadMode);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<std::string>(value_ptr);
}

void Conv3DBackpropFilter::set_pad(const std::vector<int64_t> &pad) {
  (void)CheckAndConvertUtils::CheckInteger("pad_size", SizeToLong(pad.size()), kEqual, kConv3DBackpropFilterPadSize,
                                           name());
  (void)AddAttr(kPad, api::MakeValue(CheckAndConvertUtils::CheckPositiveVector(kPad, pad, name())));
}

std::vector<int64_t> Conv3DBackpropFilter::get_pad() const {
  auto value_ptr = GetAttr(kPad);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

void Conv3DBackpropFilter::set_pad_list(const std::vector<int64_t> &pad_list) {
  (void)AddAttr(kPadList, api::MakeValue(pad_list));
}

std::vector<int64_t> Conv3DBackpropFilter::get_pad_list() const {
  auto value_ptr = GetAttr(kPadList);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

void Conv3DBackpropFilter::set_mode(const int64_t mode) {
  (void)AddAttr(kMode, api::MakeValue(CheckAndConvertUtils::CheckInteger(kMode, mode, kEqual, 1, name())));
}

int64_t Conv3DBackpropFilter::get_mode() const {
  auto value_ptr = GetAttr(kMode);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<int64_t>(value_ptr);
}

void Conv3DBackpropFilter::set_stride(const std::vector<int64_t> &stride) {
  (void)CheckAndConvertUtils::CheckInteger(kStrides, SizeToLong(stride.size()), kEqual, kConv3DBackpropFilterStrideSize,
                                           name());
  for (int64_t item : stride) {
    (void)CheckAndConvertUtils::CheckInteger(kStrides, item, kGreaterEqual, 1, name());
  }
  (void)AddAttr(kStride, api::MakeValue(stride));
}

std::vector<int64_t> Conv3DBackpropFilter::get_stride() const {
  auto value_ptr = GetAttr(kStride);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

void Conv3DBackpropFilter::set_dilation(const std::vector<int64_t> &dilation) {
  (void)CheckAndConvertUtils::CheckInteger(kDilations, SizeToLong(dilation.size()), kGreaterEqual,
                                           kConv3DBackpropFilterDilationSize, name());
  (void)AddAttr(kDilations, api::MakeValue(dilation));
}

std::vector<int64_t> Conv3DBackpropFilter::get_dilation() const {
  auto value_ptr = GetAttr(kDilation);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

void Conv3DBackpropFilter::set_group(const int64_t group) {
  (void)AddAttr(kGroup, api::MakeValue(CheckAndConvertUtils::CheckInteger(kGroup, group, kGreaterThan, 0, name())));
}

int64_t Conv3DBackpropFilter::get_group() const {
  auto value_ptr = GetAttr(kGroup);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<int64_t>(value_ptr);
}

void Conv3DBackpropFilter::set_format(const Format &format) {
  int64_t f = format;
  (void)AddAttr(kFormat, api::MakeValue(CheckAndConvertUtils::CheckInteger(kFormat, f, kEqual, Format::NCDHW, name())));
}

Format Conv3DBackpropFilter::get_format() const {
  auto value_ptr = GetAttr(kFormat);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return Format(GetValue<int64_t>(value_ptr));
}

class Conv3DBackpropFilterInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    Conv3dBackpropFilterInferCheck(primitive, input_args, true);
    auto filter_size_v = GetShapeValue(primitive, input_args[kConv3DBackpropFilterFilterSizeIndex]);
    auto dout_shape =
      CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kConv3DBackpropFilterDoutIndex]->BuildShape())[kShape];
    auto input_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(
      input_args[kConv3DBackpropFilterInputIndex]->BuildShape())[kShape];
    if (IsDynamicRank(filter_size_v) || IsDynamicRank(input_shape) || IsDynamicRank(dout_shape)) {
      std::vector<int64_t> out_shape = {abstract::Shape::kShapeRankAny};
      return std::make_shared<abstract::Shape>(out_shape);
    }
    return std::make_shared<abstract::Shape>(filter_size_v);
  }

  TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const override {
    Conv3dBackpropFilterInferCheck(prim, input_args, false);
    auto prim_name = prim->name();
    // check
    std::map<std::string, TypePtr> types;
    (void)types.emplace("x", input_args[kConv3DBackpropFilterInputIndex]->BuildType());
    (void)types.emplace("doutput", input_args[kConv3DBackpropFilterDoutIndex]->BuildType());
    std::set<TypePtr> valid_x_type = {kFloat16, kFloat32};
    (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_x_type, prim_name);
    return kFloat32;
  }
  std::set<int64_t> GetValueDependArgIndices() const override { return {kConv3DBackpropFilterFilterSizeIndex}; }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(Conv3DBackpropFilter, prim::kPrimConv3DBackpropFilter, Conv3DBackpropFilterInfer,
                                 false);
}  // namespace ops
}  // namespace mindspore
