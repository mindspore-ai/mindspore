/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "ops/grad/adaptive_avg_pool_3d_grad.h"

#include <set>

#include "abstract/param_validator.h"
#include "mindapi/src/helper.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
namespace {
constexpr int64_t kMaxShapeAdaptiveAvgPool3DGrap = 100;
abstract::ShapePtr InferShapeAdaptiveAvgPool3DGrad(const PrimitivePtr &primitive,
                                                   const std::vector<AbstractBasePtr> &input_args) {
  auto input_grad_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  auto orig_input_shape_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape())[kShape];

  const int64_t input_grad_dims = SizeToLong(input_grad_shape.size());
  const int64_t orig_input_shape_dims = SizeToLong(orig_input_shape_shape.size());
  CheckAndConvertUtils::CheckInRange("input_grad_dim", input_grad_dims, kIncludeBoth, {4, 5},
                                     kNameAdaptiveAvgPool3DGrad);
  (void)CheckAndConvertUtils::CheckInteger("orig_input_shape_dims", orig_input_shape_dims, kEqual, 1,
                                           kNameAdaptiveAvgPool3DGrad);
  CheckAndConvertUtils::CheckInRange("orig_input_shape_elem", orig_input_shape_shape[0], kIncludeBoth, {4, 5},
                                     kNameAdaptiveAvgPool3DGrad);
  std::vector<int64_t> orig_input_shape_value_vec(input_grad_dims);
  auto orig_input_shape = input_args[1];
  MS_EXCEPTION_IF_NULL(orig_input_shape);
  bool gen_value_succ = false;
  if (orig_input_shape->isa<abstract::AbstractTensor>()) {
    auto orig_input_shape_value = orig_input_shape->BuildValue();
    MS_EXCEPTION_IF_NULL(orig_input_shape_value);
    if (!orig_input_shape_value->isa<None>() && !orig_input_shape_value->isa<AnyValue>()) {
      auto orig_input_shape_tensor = orig_input_shape_value->cast<tensor::TensorPtr>();
      auto value = reinterpret_cast<int *>(orig_input_shape_tensor->data_c());
      MS_EXCEPTION_IF_NULL(value);
      for (int64_t i = 0; i < input_grad_dims; ++i) {
        orig_input_shape_value_vec[i] = value[i] > 0 ? static_cast<int64_t>(value[i]) : static_cast<int64_t>(1);
      }
      gen_value_succ = true;
    }
  }
  if (!gen_value_succ) {
    MS_LOG(WARNING) << "Output_size tensor is not a const tensor";
    ShapeVector dynamic_shape(input_grad_shape), min_shape(input_grad_shape), max_shape(input_grad_shape);
    for (int64_t i = 1; i <= input_grad_dims; ++i) {
      dynamic_shape.end()[-i] = abstract::Shape::kShapeDimAny;
      min_shape.end()[-i] = 0;
      max_shape.end()[-i] = kMaxShapeAdaptiveAvgPool3DGrap;
    }
    return std::make_shared<abstract::Shape>(dynamic_shape, min_shape, max_shape);
  } else {
    std::vector<int64_t> output_shape = input_grad_shape;
    for (int i = 1; i <= orig_input_shape_shape[0]; i++) {
      output_shape.end()[-i] = orig_input_shape_value_vec.end()[-i];
    }
    return std::make_shared<abstract::Shape>(output_shape);
  }
}

TypePtr InferTypeAdaptiveAvgPool3DGrad(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto input_grad_dtype = input_args[0]->BuildType();
  auto orig_input_shape_dtype = input_args[1]->BuildType();
  const std::set<TypePtr> input_grad_valid = {kInt8, kInt16, kInt32, kInt64, kUInt8, kFloat16, kFloat32, kFloat64};
  const std::set<TypePtr> orig_inputs_valid = {kInt32};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("input_grad", input_grad_dtype, input_grad_valid,
                                                   kNameAdaptiveAvgPool3DGrad);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("orig_input_shape", orig_input_shape_dtype, orig_inputs_valid,
                                                   kNameAdaptiveAvgPool3DGrad);
  return input_grad_dtype;
}
}  // namespace

MIND_API_OPERATOR_IMPL(AdaptiveAvgPool3DGrad, BaseOperator);
AbstractBasePtr AdaptiveAvgPool3DGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  constexpr int64_t input_num = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto types = InferTypeAdaptiveAvgPool3DGrad(primitive, input_args);
  auto shapes = InferShapeAdaptiveAvgPool3DGrad(primitive, input_args);
  return abstract::MakeAbstract(shapes, types);
}

REGISTER_PRIMITIVE_EVAL_IMPL(AdaptiveAvgPool3DGrad, prim::kPrimAdaptiveAvgPool3DGrad, AdaptiveAvgPool3DGradInfer,
                             nullptr, true);
}  // namespace ops
}  // namespace mindspore
