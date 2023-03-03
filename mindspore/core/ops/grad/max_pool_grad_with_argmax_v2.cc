/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "ops/grad/max_pool_grad_with_argmax_v2.h"

#include <algorithm>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "mindapi/src/helper.h"
#include "include/common/utils/utils.h"
#include "utils/ms_context.h"
#include "mindapi/ir/type.h"

namespace mindspore {
namespace ops {
void MaxPoolGradWithArgmaxV2::Init(const std::vector<int64_t> &kernel_size, const std::vector<int64_t> &strides,
                                   const std::vector<int64_t> &pads, const std::vector<int64_t> &dilation,
                                   bool ceil_mode, const TypeId &argmax_type) {
  set_kernel_size(kernel_size);
  set_strides(strides);
  set_pads(pads);
  set_dilation(dilation);
  set_ceil_mode(ceil_mode);
  set_argmax_type(argmax_type);
}

void MaxPoolGradWithArgmaxV2::set_kernel_size(const std::vector<int64_t> &kernel_size) {
  (void)AddAttr(kKernelSize,
                api::MakeValue(CheckAndConvertUtils::CheckPositiveVector(kKernelSize, kernel_size, name())));
}

void MaxPoolGradWithArgmaxV2::set_strides(const std::vector<int64_t> &strides) {
  (void)AddAttr(kStrides, api::MakeValue(CheckAndConvertUtils::CheckPositiveVector(kStrides, strides, name())));
}

void MaxPoolGradWithArgmaxV2::set_pads(const std::vector<int64_t> &pads) { (void)AddAttr(kPads, api::MakeValue(pads)); }

void MaxPoolGradWithArgmaxV2::set_dilation(const std::vector<int64_t> &dilation) {
  int64_t kMinDilationSize = 2;
  int64_t size = SizeToLong(dilation.size());
  (void)CheckAndConvertUtils::CheckInteger("dilation_shape", size, kGreaterThan, kMinDilationSize, name());
  std::vector<int64_t> d;
  for (int64_t i = size - kMinDilationSize; i < size; i++) {
    d.push_back(dilation[i]);
  }
  (void)AddAttr(kDilation, api::MakeValue(CheckAndConvertUtils::CheckPositiveVector(kDilation, d, name())));
}

void MaxPoolGradWithArgmaxV2::set_ceil_mode(bool ceil_mode) { (void)AddAttr(kCeilMode, api::MakeValue(ceil_mode)); }

void MaxPoolGradWithArgmaxV2::set_argmax_type(const TypeId &argmax_type) {
  int f = api::Type::GetType(argmax_type)->number_type() - kNumberTypeBool - 1;
  (void)AddAttr(kArgmaxType, api::MakeValue(f));
}

std::vector<int64_t> MaxPoolGradWithArgmaxV2::get_kernel_size() const {
  return GetValue<std::vector<int64_t>>(GetAttr(kKernelSize));
}

std::vector<int64_t> MaxPoolGradWithArgmaxV2::get_strides() const {
  return GetValue<std::vector<int64_t>>(GetAttr(kStrides));
}

std::vector<int64_t> MaxPoolGradWithArgmaxV2::get_pads() const {
  return GetValue<std::vector<int64_t>>(GetAttr(kPads));
}

std::vector<int64_t> MaxPoolGradWithArgmaxV2::get_dilation() const {
  return GetValue<std::vector<int64_t>>(GetAttr(kDilation));
}

bool MaxPoolGradWithArgmaxV2::get_ceil_mode() const { return GetValue<bool>(GetAttr(kCeilMode)); }

TypeId MaxPoolGradWithArgmaxV2::get_argmax_type() const {
  auto number_type = GetValue<int>(GetAttr(kArgmaxType));
  if (number_type == kNumberTypeInt32 - kNumberTypeBool - 1) {
    return kNumberTypeInt32;
  } else {
    return kNumberTypeInt64;
  }
}

TypePtr MaxPoolGradWithArgmaxV2InferType(const PrimitivePtr &prim,
                                         const std::vector<abstract::AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto op_name = prim->name();
  const int64_t input_num = 3;
  (void)CheckAndConvertUtils::CheckInteger("input size", SizeToLong(input_args.size()), kEqual, input_num, op_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (context->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kAscendDevice) {
    (void)CheckAndConvertUtils::CheckTensorTypeValid("argmax", input_args[kInputIndex2]->BuildType(), {kUInt16},
                                                     op_name);
  } else {
    (void)CheckAndConvertUtils::CheckTensorTypeValid("argmax", input_args[kInputIndex2]->BuildType(), {kInt64, kInt32},
                                                     op_name);
  }
  const std::set<TypePtr> valid_types = {kInt8,   kInt16,  kInt32,   kInt64,   kUInt8,  kUInt16,
                                         kUInt32, kUInt64, kFloat16, kFloat32, kFloat64};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", input_args[kInputIndex0]->BuildType(), valid_types, op_name);
  return input_args[0]->BuildType();
}

abstract::ShapePtr MaxPoolGradWithArgmaxV2InferShape(const PrimitivePtr &prim,
                                                     const std::vector<abstract::AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto op_name = prim->name();
  const int64_t input_num = 3;
  (void)CheckAndConvertUtils::CheckInteger("input size", SizeToLong(input_args.size()), kEqual, input_num, op_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  const int64_t kInputDims = 4;
  const size_t kAttrsSize = 4;
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  auto argmax_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->BuildShape())[kShape];
  if (IsDynamicRank(x_shape)) {
    return std::make_shared<abstract::Shape>(x_shape);
  }
  (void)CheckAndConvertUtils::CheckInteger("x_shape", x_shape.size(), kEqual, kInputDims, op_name);
  (void)CheckAndConvertUtils::CheckInteger("argmax_shape", argmax_shape.size(), kEqual, kInputDims, op_name);
  auto kernel_size = GetValue<std::vector<int64_t>>(prim->GetAttr(kKernelSize));
  (void)CheckAndConvertUtils::CheckInteger("kernel_size rank", SizeToLong(kernel_size.size()), kEqual, kAttrsSize,
                                           prim->name());
  auto strides = GetValue<std::vector<int64_t>>(prim->GetAttr(kStrides));
  (void)CheckAndConvertUtils::CheckInteger("strides rank", SizeToLong(strides.size()), kEqual, kAttrsSize,
                                           prim->name());
  auto pads = GetValue<std::vector<int64_t>>(prim->GetAttr(kPads));
  (void)CheckAndConvertUtils::CheckInteger("pads rank", SizeToLong(pads.size()), kEqual, kAttrsSize, prim->name());
  auto dilation = GetValue<std::vector<int64_t>>(prim->GetAttr(kDilation));
  (void)CheckAndConvertUtils::CheckInteger("dilation rank", SizeToLong(dilation.size()), kEqual, kAttrsSize,
                                           prim->name());
  return std::make_shared<abstract::Shape>(x_shape);
}

abstract::AbstractBasePtr MaxPoolGradWithArgmaxV2Infer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &prim,
                                                       const std::vector<abstract::AbstractBasePtr> &input_args) {
  auto infer_type = MaxPoolGradWithArgmaxV2InferType(prim, input_args);
  auto infer_shape = MaxPoolGradWithArgmaxV2InferShape(prim, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

MIND_API_OPERATOR_IMPL(MaxPoolGradWithArgmaxV2, BaseOperator);

// AG means auto generated
class MIND_API AGMaxPoolGradWithArgmaxV2Infer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const override {
    return MaxPoolGradWithArgmaxV2InferShape(prim, input_args);
  }

  TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const override {
    return MaxPoolGradWithArgmaxV2InferType(prim, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &prim,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return MaxPoolGradWithArgmaxV2Infer(engine, prim, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(MaxPoolGradWithArgmaxV2, prim::kPrimMaxPoolGradWithArgmaxV2,
                                 AGMaxPoolGradWithArgmaxV2Infer, false);
}  // namespace ops
}  // namespace mindspore
