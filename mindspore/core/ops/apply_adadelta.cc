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

#include "ops/apply_adadelta.h"

#include <algorithm>
#include <set>

#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"
#include "utils/tensor_construct_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::TupleShapePtr ApplyAdadeltaInferShape(const PrimitivePtr &primitive,
                                                const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();

  auto var_shape = input_args[kInputIndex0]->BuildShape();
  auto accum_shape = input_args[kInputIndex1]->BuildShape();
  auto accum_update_shape = input_args[kInputIndex2]->BuildShape();
  auto lr_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex3]->BuildShape())[kShape];
  auto rho_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex4]->BuildShape())[kShape];
  auto epsilon_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex5]->BuildShape())[kShape];
  auto grad_shape = input_args[kInputIndex6]->BuildShape();
  auto var_shape_ptr = var_shape->cast<abstract::ShapePtr>();
  auto accum_shape_ptr = accum_shape->cast<abstract::ShapePtr>();
  auto accum_update_shape_ptr = accum_update_shape->cast<abstract::ShapePtr>();
  auto grad_shape_ptr = grad_shape->cast<abstract::ShapePtr>();
  // var and accum must have the same shape when is not dynamic
  if (!var_shape_ptr->IsDynamic() && !accum_shape_ptr->IsDynamic()) {
    if (*var_shape != *accum_shape) {
      MS_EXCEPTION(ValueError) << prim_name << " evaluator arg accum shape " << accum_shape->ToString()
                               << " are not consistent with var shape " << var_shape->ToString();
    }
  }
  // var and accum update must have the same shape when is not dynamic
  if (!var_shape_ptr->IsDynamic() && !accum_update_shape_ptr->IsDynamic()) {
    if (*var_shape != *accum_update_shape) {
      MS_EXCEPTION(ValueError) << prim_name << " evaluator arg accum update shape " << accum_update_shape->ToString()
                               << " are not consistent with var shape " << var_shape->ToString();
    }
  }
  // var and grad must have the same shape when is not dynamic
  if (!var_shape_ptr->IsDynamic() && !grad_shape_ptr->IsDynamic()) {
    if (*var_shape != *grad_shape) {
      MS_EXCEPTION(ValueError) << prim_name << " evaluator arg grad shape " << grad_shape->ToString()
                               << " are not consistent with var shape " << var_shape->ToString();
    }
  }
  const int64_t kShapeSize = 1;
  auto lr_shape_size = lr_shape.size();
  (void)CheckAndConvertUtils::CheckInteger("lr's rank'", lr_shape_size, kLessEqual, kShapeSize, primitive->name());
  if (lr_shape_size == 1) {
    (void)CheckAndConvertUtils::CheckInteger("lr_shape[0]", lr_shape[0], kEqual, kShapeSize, primitive->name());
  }

  auto rho_shape_size = rho_shape.size();
  (void)CheckAndConvertUtils::CheckInteger("rho's rank'", rho_shape_size, kLessEqual, kShapeSize, primitive->name());
  if (rho_shape_size == 1) {
    (void)CheckAndConvertUtils::CheckInteger("rho_shape[0]", rho_shape[0], kEqual, kShapeSize, primitive->name());
  }

  auto epsilon_shape_size = epsilon_shape.size();
  (void)CheckAndConvertUtils::CheckInteger("epsilon's rank'", epsilon_shape_size, kLessEqual, kShapeSize,
                                           primitive->name());
  if (epsilon_shape_size == 1) {
    (void)CheckAndConvertUtils::CheckInteger("epsilon_shape[0]", epsilon_shape[0], kEqual, kShapeSize,
                                             primitive->name());
  }

  return std::make_shared<abstract::TupleShape>(
    std::vector<abstract::BaseShapePtr>{var_shape, accum_shape, accum_update_shape});
}

TuplePtr ApplyAdadeltaInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  auto var_type = input_args[kInputIndex0]->BuildType();
  auto accum_type = input_args[kInputIndex1]->BuildType();
  auto accum_update_type = input_args[kInputIndex2]->BuildType();
  auto lr_type = input_args[kInputIndex3]->BuildType();
  auto rho_type = input_args[kInputIndex4]->BuildType();
  auto epsilon_type = input_args[kInputIndex5]->BuildType();
  auto grad_type = input_args[kInputIndex6]->BuildType();

  const std::set<TypePtr> valid_types = {kFloat16, kFloat32};
  std::map<std::string, TypePtr> args;
  (void)args.insert({"var_type", var_type});
  (void)args.insert({"accum_type", accum_type});
  (void)args.insert({"accum_update_type", accum_update_type});
  (void)args.insert({"grad_type", grad_type});
  (void)CheckAndConvertUtils::CheckTensorTypeSame(args, valid_types, prim_name);

  std::map<std::string, TypePtr> args_lr;
  (void)args_lr.insert({"lr_type", lr_type});
  (void)CheckAndConvertUtils::CheckScalarOrTensorTypesSame(args_lr, valid_types, prim_name);

  std::map<std::string, TypePtr> args_rho;
  (void)args_rho.insert({"rho_type", rho_type});
  (void)CheckAndConvertUtils::CheckScalarOrTensorTypesSame(args_rho, valid_types, prim_name);

  std::map<std::string, TypePtr> args_epsilon;
  (void)args_epsilon.insert({"epsilon_type", epsilon_type});
  (void)CheckAndConvertUtils::CheckScalarOrTensorTypesSame(args_epsilon, valid_types, prim_name);

  return std::make_shared<Tuple>(std::vector<TypePtr>{var_type, accum_type, accum_update_type});
}
}  // namespace

MIND_API_BASE_IMPL(ApplyAdadelta, PrimitiveC, BaseOperator);
AbstractBasePtr ApplyAdadeltaInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) {
  auto infer_type = ApplyAdadeltaInferType(primitive, input_args);
  auto infer_shape = ApplyAdadeltaInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(ApplyAdadelta, prim::kPrimApplyAdadelta, ApplyAdadeltaInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
