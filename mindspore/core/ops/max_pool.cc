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
#include <vector>
#include <map>

#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "utils/ms_context.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/primitive.h"
#include "ir/value.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
constexpr size_t kSizeFour = 4;
constexpr size_t kIndex0 = 0;
constexpr size_t kIndex1 = 1;
constexpr size_t kIndex2 = 2;
constexpr size_t kIndex3 = 3;
constexpr auto kAttrPrimitiveTarget = "primitive_target";

void ConvertShapeNHWCToNCHW(std::vector<int64_t> *const nhwc_shape) {
  if (nhwc_shape->empty()) {
    return;
  }
  if (nhwc_shape->size() != kSizeFour) {
    MS_EXCEPTION(ValueError) << "The size of shape should be 4, but got " << nhwc_shape->size();
  }
  int64_t tmp = (*nhwc_shape)[kIndex3];
  (*nhwc_shape)[kIndex3] = (*nhwc_shape)[kIndex2];
  (*nhwc_shape)[kIndex2] = (*nhwc_shape)[kIndex1];
  (*nhwc_shape)[kIndex1] = tmp;
}

int64_t CeilDiv(int64_t a, int64_t b) {
  if (b == 0) {
    MS_EXCEPTION(ValueError) << "The number can not be divided by zero.";
  }
  int64_t result = a / b;
  if (a % b != 0) {
    result += 1;
  }
  return result;
}

void CheckOutshapeValid(const PrimitivePtr &primitive, const std::vector<int64_t> &out_shape,
                        const std::vector<int64_t> &in_shape, const std::vector<int64_t> &kernel_size,
                        const std::vector<int64_t> &strides) {
  for (auto out : out_shape) {
    if (out <= 0 && out != -1) {
      MS_EXCEPTION(ValueError)
        << "For '" << primitive->name()
        << "', the each element of the output shape must be larger than 0, but got output shape: " << out_shape
        << ". The input shape: " << in_shape << ", kernel size: " << kernel_size << ", strides: " << strides
        << ". Please check the official api documents for more information about the output.";
    }
  }
}

string GetDeviceTarget(const PrimitivePtr &primitive) {
  string primitive_target;
  if (primitive->HasAttr(kAttrPrimitiveTarget)) {
    primitive_target = GetValue<std::string>(primitive->GetAttr(kAttrPrimitiveTarget));
  } else {
    auto ms_context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(ms_context);
    primitive_target = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  }
  return primitive_target;
}

void CheckDataFormat(const PrimitivePtr &primitive, Format data_format, const string &primitive_target) {
  if (data_format == NHWC) {
    if (primitive_target != kGPUDevice) {
      MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                               << "', the 'NHWC' format is only supported in GPU target.";
    }
  } else if (data_format != NCHW) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', the input format should be NCHW or NHWC, but got "
                             << data_format << ".";
  }
}

abstract::ShapePtr MaxPoolInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  std::vector<int64_t> kernel_size = GetValue<std::vector<int64_t>>(primitive->GetAttr(kKernelSize));
  std::vector<int64_t> strides = GetValue<std::vector<int64_t>>(primitive->GetAttr(kStrides));
  Format data_format = static_cast<Format>(CheckAndConvertUtils::GetAndCheckFormat(primitive->GetAttr(kFormat)));
  (void)CheckAndConvertUtils::CheckPositiveVector("kernel_size", kernel_size, op_name);
  (void)CheckAndConvertUtils::CheckPositiveVector("strides", strides, op_name);
  int64_t pad_mode_int = 0;
  CheckAndConvertUtils::GetPadModEnumValue(primitive->GetAttr(kPadMode), &pad_mode_int, true);
  PadMode pad_mode = static_cast<PadMode>(pad_mode_int);

  (void)CheckAndConvertUtils::CheckValue<size_t>("length of kernel_size", kernel_size.size(), kEqual, kSizeFour,
                                                 op_name);
  (void)CheckAndConvertUtils::CheckValue<size_t>("length of strides", strides.size(), kEqual, kSizeFour, op_name);

  auto shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape());
  if (shape_map.empty()) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', the input should exist, but missed.";
  }
  string device_target = GetDeviceTarget(primitive);
  CheckDataFormat(primitive, data_format, device_target);
  auto in_shape = shape_map[kShape];
  if (IsDynamicRank(in_shape)) {
    if (device_target == kAscendDevice) {
      MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                               << "', the Ascend platform hasn't support dynamic rank yet.";
    }
    std::vector<int64_t> out_shape = {-2};
    return std::make_shared<abstract::Shape>(out_shape);
  }
  (void)CheckAndConvertUtils::CheckValue<size_t>("length of input", in_shape.size(), kEqual, kSizeFour, op_name);

  if (data_format == NHWC) {
    ConvertShapeNHWCToNCHW(&in_shape);
  }

  int64_t out_h = 0, out_w = 0;
  if (pad_mode == PadMode::SAME) {
    out_h = in_shape[kIndex2] == -1 ? -1 : CeilDiv(in_shape[kIndex2], strides[kIndex2]);
    out_w = in_shape[kIndex3] == -1 ? -1 : CeilDiv(in_shape[kIndex3], strides[kIndex3]);
  } else if (pad_mode == PadMode::VALID) {
    out_h = in_shape[kIndex2] == -1 ? -1 : CeilDiv((in_shape[kIndex2] - (kernel_size[kIndex2] - 1)), strides[kIndex2]);
    out_w = in_shape[kIndex3] == -1 ? -1 : CeilDiv((in_shape[kIndex3] - (kernel_size[kIndex3] - 1)), strides[kIndex3]);
  } else {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', the pad_mode should be same or valid, but got "
                             << pad_mode << ".";
  }

  abstract::ShapePtr shape;
  std::vector<int64_t> out_shape;
  if (data_format == NHWC) {
    out_shape = {in_shape[kIndex0], out_h, out_w, in_shape[kIndex1]};
    shape = std::make_shared<abstract::Shape>(out_shape);
  } else {
    out_shape = {in_shape[kIndex0], in_shape[kIndex1], out_h, out_w};
    shape = std::make_shared<abstract::Shape>(out_shape);
  }

  CheckOutshapeValid(primitive, out_shape, in_shape, kernel_size, strides);
  return shape;
}

TypePtr MaxPoolInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  if (input_args.size() == 0) {
    MS_EXCEPTION(TypeError) << "For '" << primitive->name()
                            << "', the input args used for infer shape and type is necessary, but missing it.";
  }
  if (std::any_of(input_args.begin(), input_args.end(), [](const AbstractBasePtr &a) { return a == nullptr; })) {
    MS_EXCEPTION(TypeError) << "For '" << primitive->name()
                            << "', the input args used for infer shape and type is necessary, but missing it.";
  }
  auto type = CheckAndConvertUtils::GetTensorInputType(primitive->name(), input_args, 0);
  return type;
}
}  // namespace

void MaxPool::set_pad_mode(const PadMode &pad_mode) {
  int64_t swi = pad_mode;
  (void)this->AddAttr(kPadMode, api::MakeValue(swi));
}

PadMode MaxPool::get_pad_mode() const { return PadMode(GetValue<int64_t>(GetAttr(kPadMode))); }
void MaxPool::set_kernel_size(const std::vector<int64_t> &kernel_size) {
  (void)this->AddAttr(
    kKernelSize, api::MakeValue(CheckAndConvertUtils::CheckPositiveVector(kKernelSize, kernel_size, this->name())));
}

std::vector<int64_t> MaxPool::get_kernel_size() const { return GetValue<std::vector<int64_t>>(GetAttr(kKernelSize)); }
void MaxPool::set_strides(const std::vector<int64_t> &strides) {
  (void)this->AddAttr(kStrides,
                      api::MakeValue(CheckAndConvertUtils::CheckPositiveVector(kStrides, strides, this->name())));
}

std::vector<int64_t> MaxPool::get_strides() const { return GetValue<std::vector<int64_t>>(GetAttr(kStrides)); }

void MaxPool::set_format(const Format &format) {
  int64_t f = format;
  (void)this->AddAttr(kFormat, api::MakeValue(f));
}

Format MaxPool::get_format() const { return Format(GetValue<int64_t>(GetAttr(kFormat))); }

void MaxPool::set_pad(const std::vector<int64_t> &pad) { (void)this->AddAttr(kPad, api::MakeValue(pad)); }

std::vector<int64_t> MaxPool::get_pad() const {
  auto value_ptr = GetAttr(kPad);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

void MaxPool::set_round_mode(const RoundMode &round_mode) {
  int64_t swi = round_mode;
  (void)this->AddAttr(kRoundMode, api::MakeValue(swi));
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

MIND_API_OPERATOR_IMPL(MaxPool, BaseOperator);
abstract::AbstractBasePtr MaxPoolInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const std::vector<abstract::AbstractBasePtr> &input_args) {
  TypePtr type = MaxPoolInferType(primitive, input_args);
  abstract::ShapePtr shape = MaxPoolInferShape(primitive, input_args);
  return abstract::MakeAbstract(shape, type);
}

// AG means auto generated
class MIND_API AGMaxPoolInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return MaxPoolInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return MaxPoolInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return MaxPoolInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(MaxPool, prim::kPrimMaxPool, AGMaxPoolInfer, false);
}  // namespace ops
}  // namespace mindspore
