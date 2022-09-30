/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "ops/grad/adaptive_max_pool_3d_grad.h"

#include "ops/op_utils.h"
#include "mindapi/src/helper.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore {
namespace ops {
namespace {
constexpr auto kNameAdaptiveMaxPool3DGrad = "AdaptiveMaxPool3DGrad";

bool AdaptiveMaxPool3DGradIsDynamic(const ShapeVector &shape) {
  if (std::find(shape.begin(), shape.end(), -1) != shape.end()) {
    return true;
  }
  return false;
}

abstract::ShapePtr AdaptiveMaxPool3DGradInferShape(const PrimitivePtr &,
                                                   const std::vector<AbstractBasePtr> &input_args) {
  auto input_grad_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape())[kShape];
  auto argmax_shape_ptr = CheckAndConvertUtils::GetTensorInputShape("AdaptiveMaxPool3DGrad", input_args, 2);
  MS_EXCEPTION_IF_NULL(argmax_shape_ptr);

  const int64_t input_grad_dims = SizeToLong(input_grad_shape.size());
  const int64_t x_dims = SizeToLong(x_shape.size());
  CheckAndConvertUtils::CheckInRange("input_grad_dim", input_grad_dims, kIncludeBoth, {4, 5},
                                     kNameAdaptiveMaxPool3DGrad);
  CheckAndConvertUtils::CheckInRange("x_dim", x_dims, kIncludeBoth, {4, 5}, kNameAdaptiveMaxPool3DGrad);
  (void)CheckAndConvertUtils::CheckInteger("input_grad_dims", input_grad_dims, kEqual, x_dims,
                                           kNameAdaptiveMaxPool3DGrad);
  auto argmax_shape = argmax_shape_ptr->shape();
  if (!AdaptiveMaxPool3DGradIsDynamic(argmax_shape)) {
    const int64_t argmax_dim = SizeToLong(argmax_shape.size());
    CheckAndConvertUtils::CheckInRange("argmax_dim", argmax_dim, kIncludeBoth, {4, 5}, kNameAdaptiveMaxPool3DGrad);
    (void)CheckAndConvertUtils::CheckInteger("argmax_dim", argmax_dim, kEqual, x_dims, kNameAdaptiveMaxPool3DGrad);
    if (input_grad_shape != argmax_shape) {
      MS_LOG(EXCEPTION) << "Input grad shape must be same with argmax shape.";
    }
  } else {
    for (int64_t i = 1; i < x_dims; ++i) {
      argmax_shape[LongToSize(i)] = abstract::Shape::kShapeDimAny;
    }
  }
  return std::make_shared<abstract::Shape>(x_shape);
}

TypePtr AdaptiveMaxPool3DGradInferType(const PrimitivePtr &, const std::vector<AbstractBasePtr> &input_args) {
  auto input_grad_dtype = input_args[0]->BuildType();
  auto x_dtype = input_args[1]->BuildType();
  auto argmax_dtype = input_args[2]->BuildType();
  const std::set<TypePtr> real_number_types = {kInt8,   kInt16,  kInt32,   kInt64,   kUInt8,  kUInt16,
                                               kUInt32, kUInt64, kFloat16, kFloat32, kFloat64};
  const std::set<TypePtr> argmax_valid_types = {kInt32};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("input_grad_dtype", input_grad_dtype, real_number_types,
                                                   kNameAdaptiveMaxPool3DGrad);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x_dtype", x_dtype, real_number_types, kNameAdaptiveMaxPool3DGrad);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("argmax_dtype", argmax_dtype, argmax_valid_types,
                                                   kNameAdaptiveMaxPool3DGrad);
  return x_dtype;
}
}  // namespace

MIND_API_OPERATOR_IMPL(AdaptiveMaxPool3DGrad, BaseOperator);
AbstractBasePtr AdaptiveMaxPool3DGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  constexpr int64_t input_num = 3;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto types = AdaptiveMaxPool3DGradInferType(primitive, input_args);
  auto shapes = AdaptiveMaxPool3DGradInferShape(primitive, input_args);
  return abstract::MakeAbstract(shapes, types);
}
REGISTER_PRIMITIVE_EVAL_IMPL(AdaptiveMaxPool3DGrad, prim::kPrimAdaptiveMaxPool3DGrad, AdaptiveMaxPool3DGradInfer,
                             nullptr, true);
}  // namespace ops
}  // namespace mindspore
