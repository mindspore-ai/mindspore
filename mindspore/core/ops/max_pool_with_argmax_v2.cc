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

#include "ops/max_pool_with_argmax_v2.h"
#include <algorithm>
#include <set>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "mindapi/src/helper.h"
#include "include/common/utils/utils.h"
#include "utils/ms_context.h"
#include "mindapi/ir/type.h"

namespace mindspore {
namespace ops {
void MaxPoolWithArgmaxV2::Init(const std::vector<int64_t> &kernel_size, const std::vector<int64_t> &strides,
                               const std::vector<int64_t> &pads, const std::vector<int64_t> &dilation, bool ceil_mode,
                               const TypeId &argmax_type) {
  set_kernel_size(kernel_size);
  set_strides(strides);
  set_pads(pads);
  set_dilation(dilation);
  set_ceil_mode(ceil_mode);
  set_argmax_type(argmax_type);
}

void MaxPoolWithArgmaxV2::set_kernel_size(const std::vector<int64_t> &kernel_size) {
  (void)AddAttr(kKernelSize, api::MakeValue(kernel_size));
}

void MaxPoolWithArgmaxV2::set_strides(const std::vector<int64_t> &strides) {
  (void)AddAttr(kStrides, api::MakeValue(strides));
}

void MaxPoolWithArgmaxV2::set_pads(const std::vector<int64_t> &pads) { (void)AddAttr(kPads, api::MakeValue(pads)); }

void MaxPoolWithArgmaxV2::set_dilation(const std::vector<int64_t> &dilation) {
  int64_t kMinDilationSize = 2;
  int64_t size = SizeToLong(dilation.size());
  (void)CheckAndConvertUtils::CheckInteger("dilation_shape", size, kGreaterThan, kMinDilationSize, name());
  std::vector<int64_t> d;
  for (int64_t i = size - kMinDilationSize; i < size; i++) {
    d.push_back(dilation[i]);
  }
  (void)AddAttr(kDilation, api::MakeValue(d));
}

void MaxPoolWithArgmaxV2::set_ceil_mode(bool ceil_mode) { (void)AddAttr(kCeilMode, api::MakeValue(ceil_mode)); }

void MaxPoolWithArgmaxV2::set_argmax_type(const TypeId &argmax_type) {
  int f = api::Type::GetType(argmax_type)->number_type() - kNumberTypeBool - 1;
  (void)AddAttr(kArgmaxType, api::MakeValue(f));
}

std::vector<int64_t> MaxPoolWithArgmaxV2::get_kernel_size() const {
  return GetValue<std::vector<int64_t>>(GetAttr(kKernelSize));
}

std::vector<int64_t> MaxPoolWithArgmaxV2::get_strides() const {
  return GetValue<std::vector<int64_t>>(GetAttr(kStrides));
}

std::vector<int64_t> MaxPoolWithArgmaxV2::get_pads() const { return GetValue<std::vector<int64_t>>(GetAttr(kPads)); }

std::vector<int64_t> MaxPoolWithArgmaxV2::get_dilation() const {
  return GetValue<std::vector<int64_t>>(GetAttr(kDilation));
}

bool MaxPoolWithArgmaxV2::get_ceil_mode() const { return GetValue<bool>(GetAttr(kCeilMode)); }

TypeId MaxPoolWithArgmaxV2::get_argmax_type() const {
  auto number_type = GetValue<int>(GetAttr(kArgmaxType));
  if (number_type == kAiCoreNumTypeInt32) {
    return kNumberTypeInt32;
  } else {
    return kNumberTypeInt64;
  }
}

TuplePtr MaxPoolWithArgmaxV2InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  const std::set<TypePtr> valid_types = {kInt8,   kInt16,  kInt32,   kInt64,   kUInt8,  kUInt16,
                                         kUInt32, kUInt64, kFloat16, kFloat32, kFloat64};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("input", input_args[0]->BuildType(), valid_types, prim->name());
  auto output_dtype = input_args[0]->BuildType();
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  TypePtr argmax_dtype;
  if (context->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kAscendDevice) {
    argmax_dtype = std::make_shared<TensorType>(kUInt16);
  } else {
    auto target_max = GetValue<int64_t>(prim->GetAttr(kArgmaxType));
    if (target_max == kAiCoreNumTypeInt32) {
      argmax_dtype = std::make_shared<TensorType>(kInt32);
    } else if (target_max == kAiCoreNumTypeInt64) {
      argmax_dtype = std::make_shared<TensorType>(kInt64);
    } else {
      MS_EXCEPTION(TypeError) << "For " << prim->name() << ", the type of argmax should be int32 or int64.";
    }
  }
  std::vector<TypePtr> type_list = {output_dtype, argmax_dtype};
  return std::make_shared<Tuple>(type_list);
}

abstract::TupleShapePtr MaxPoolWithArgmaxV2InferShape(const PrimitivePtr &prim,
                                                      const std::vector<AbstractBasePtr> &input_args) {
  const size_t kAttrH = 2;
  const size_t kAttrW = 3;
  const size_t kInputShapeSize = 4;
  const size_t kAttrsSize = 4;
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  if (IsDynamicRank(x_shape)) {
    std::vector<abstract::BaseShapePtr> shape_list = {std::make_shared<abstract::Shape>(std::vector<int64_t>{
                                                        abstract::Shape::kShapeDimAny, abstract::Shape::kShapeDimAny,
                                                        abstract::Shape::kShapeDimAny, abstract::Shape::kShapeDimAny}),
                                                      std::make_shared<abstract::Shape>(std::vector<int64_t>{
                                                        abstract::Shape::kShapeDimAny, abstract::Shape::kShapeDimAny,
                                                        abstract::Shape::kShapeDimAny, abstract::Shape::kShapeDimAny})};
    return std::make_shared<abstract::TupleShape>(shape_list);
  }
  (void)CheckAndConvertUtils::CheckInteger("input x rank", SizeToLong(x_shape.size()), kEqual, kInputShapeSize,
                                           prim->name());
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
  if (IsDynamic(x_shape)) {
    std::vector<abstract::BaseShapePtr> shape_list = {
      std::make_shared<abstract::Shape>(
        std::vector<int64_t>{x_shape[0], x_shape[1], abstract::Shape::kShapeDimAny, abstract::Shape::kShapeDimAny}),
      std::make_shared<abstract::Shape>(
        std::vector<int64_t>{x_shape[0], x_shape[1], abstract::Shape::kShapeDimAny, abstract::Shape::kShapeDimAny})};
    return std::make_shared<abstract::TupleShape>(shape_list);
  }
  (void)CheckAndConvertUtils::CheckPositiveVectorExcludeZero(kKernelSize, kernel_size, prim->name());
  (void)CheckAndConvertUtils::CheckPositiveVectorExcludeZero(kStrides, strides, prim->name());
  (void)CheckAndConvertUtils::CheckPositiveVector(kPads, pads, prim->name());
  (void)CheckAndConvertUtils::CheckPositiveVectorExcludeZero(kDilation, dilation, prim->name());

  auto H_in = x_shape[kIndex2];
  auto W_in = x_shape[kIndex3];
  auto H_out = 0;
  auto W_out = 0;
  auto factor = 2.0;
  auto H_out_d =
    ((H_in + factor * pads[kAttrH] - dilation[kAttrH] * (kernel_size[kAttrH] - 1) - 1) / strides[kAttrH]) + 1;
  auto W_out_d =
    ((W_in + factor * pads[kAttrW] - dilation[kAttrW] * (kernel_size[kAttrW] - 1) - 1) / strides[kAttrW]) + 1;
  if (GetValue<bool>(prim->GetAttr(kCeilMode))) {
    H_out = static_cast<int>(ceil(H_out_d));
    W_out = static_cast<int>(ceil(W_out_d));
    // Whether the last pooling starts inside the image or not.
    if ((H_out - 1) * strides[kAttrH] >= H_in + pads[kAttrH]) {
      --H_out;
    }
    if ((W_out - 1) * strides[kAttrW] >= W_in + pads[kAttrW]) {
      --W_out;
    }
  } else {
    H_out = static_cast<int>(floor(H_out_d));
    W_out = static_cast<int>(floor(W_out_d));
  }
  ShapeVector output_shape = {x_shape[0], x_shape[1], H_out, W_out};
  if (H_out <= 0 || W_out <= 0) {
    MS_EXCEPTION(ValueError) << "The shape of input is [" << x_shape[0] << ", " << x_shape[1] << ", " << H_in << ", "
                             << W_in << "], which is invalid for the attributes of " << prim->name();
  }
  ShapeVector argmax_shape = output_shape;
  std::vector<abstract::BaseShapePtr> shape_list = {std::make_shared<abstract::Shape>(output_shape),
                                                    std::make_shared<abstract::Shape>(argmax_shape)};
  return std::make_shared<abstract::TupleShape>(shape_list);
}

AbstractBasePtr MaxPoolWithArgmaxV2Infer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) {
  const int64_t input_num = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infer_type = MaxPoolWithArgmaxV2InferType(primitive, input_args);
  auto infer_shape = MaxPoolWithArgmaxV2InferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
MIND_API_OPERATOR_IMPL(MaxPoolWithArgmaxV2, BaseOperator);

// AG means auto generated
class MIND_API AGMaxPoolWithArgmaxV2Infer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return MaxPoolWithArgmaxV2InferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return MaxPoolWithArgmaxV2InferType(primitive, input_args);
  }

  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return MaxPoolWithArgmaxV2Infer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(MaxPoolWithArgmaxV2, prim::kPrimMaxPoolWithArgmaxV2, AGMaxPoolWithArgmaxV2Infer,
                                 false);
}  // namespace ops
}  // namespace mindspore
