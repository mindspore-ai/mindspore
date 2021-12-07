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

#include "ops/apply_ada_max.h"

#include <algorithm>
#include <set>
#include "abstract/primitive_infer_map.h"
#include "ops/op_utils.h"
#include "utils/tensor_construct_utils.h"

namespace mindspore {
namespace ops {
namespace {
abstract::TupleShapePtr ApplyAdaMaxInferShape(const PrimitivePtr &primitive,
                                              const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputNum = 9;
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kGreaterEqual, kInputNum,
                                           primitive->name());
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto prim_name = primitive->name();
  auto var_shape = input_args[kInputIndex0]->BuildShape();
  auto m_shape = input_args[kInputIndex1]->BuildShape();
  auto v_shape = input_args[kInputIndex2]->BuildShape();
  auto var_shape_ptr = var_shape->cast<abstract::ShapePtr>();
  auto m_shape_ptr = m_shape->cast<abstract::ShapePtr>();
  auto v_shape_ptr = v_shape->cast<abstract::ShapePtr>();
  auto beta1_power_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex3]->BuildShape())[kShape];
  auto lr_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex4]->BuildShape())[kShape];
  auto beta1_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex5]->BuildShape())[kShape];
  auto beta2_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex6]->BuildShape())[kShape];
  auto epsilon_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex7]->BuildShape())[kShape];
  auto grad_shape = input_args[kInputIndex8]->BuildShape();
  auto grad_shape_ptr = grad_shape->cast<abstract::ShapePtr>();
  // beta1_power,lr,beta1,beta2,epsilon must be scalar
  const int64_t kInputShape = 1;
  (void)CheckAndConvertUtils::CheckInteger("beta1 power's rank", beta1_power_shape.size(), kLessEqual, kInputShape,
                                           prim_name);
  if (beta1_power_shape.size() == 1) {
    (void)CheckAndConvertUtils::CheckInteger("beta1_power_shape[0]", beta1_power_shape.size(), kEqual, kInputShape,
                                             prim_name);
  }
  (void)CheckAndConvertUtils::CheckInteger("lr's rank", lr_shape.size(), kLessEqual, kInputShape, prim_name);
  if (lr_shape.size() == 1) {
    (void)CheckAndConvertUtils::CheckInteger("lr_shape[0]", lr_shape.size(), kEqual, kInputShape, prim_name);
  }
  (void)CheckAndConvertUtils::CheckInteger("beta1's rank", beta1_shape.size(), kLessEqual, kInputShape, prim_name);
  if (beta1_shape.size() == 1) {
    (void)CheckAndConvertUtils::CheckInteger("beta1_shape[0]", beta1_shape.size(), kEqual, kInputShape, prim_name);
  }
  (void)CheckAndConvertUtils::CheckInteger("beta2's rank", beta2_shape.size(), kLessEqual, kInputShape, prim_name);
  if (beta2_shape.size() == 1) {
    (void)CheckAndConvertUtils::CheckInteger("beta2_shape[0]", beta2_shape.size(), kEqual, kInputShape, prim_name);
  }
  (void)CheckAndConvertUtils::CheckInteger("epsilon's rank", epsilon_shape.size(), kLessEqual, kInputShape, prim_name);
  if (epsilon_shape.size() == 1) {
    (void)CheckAndConvertUtils::CheckInteger("epsilon_shape[0]", epsilon_shape.size(), kEqual, kInputShape, prim_name);
  }

  // var, m,v and grad must have the same shape
  std::map<std::string, abstract::BaseShapePtr> same_shape_args_map;
  same_shape_args_map.insert({"m", m_shape});
  same_shape_args_map.insert({"v", v_shape});
  same_shape_args_map.insert({"grad", grad_shape});
  if (!var_shape_ptr->IsDynamic() && !m_shape_ptr->IsDynamic()) {
    if (*m_shape != *var_shape) {
      MS_EXCEPTION(ValueError) << primitive->name() << " evaluator arg m shape " << m_shape->ToString()
                               << " are not consistent with var shape " << var_shape->ToString();
    }
  }
  if (!v_shape_ptr->IsDynamic() && !var_shape_ptr->IsDynamic()) {
    if (*v_shape != *var_shape) {
      MS_EXCEPTION(ValueError) << primitive->name() << " evaluator arg v shape " << v_shape->ToString()
                               << " are not consistent with var shape " << var_shape->ToString();
    }
  }
  if (!grad_shape_ptr->IsDynamic() && !var_shape_ptr->IsDynamic()) {
    if (*grad_shape != *var_shape) {
      MS_EXCEPTION(ValueError) << primitive->name() << " evaluator arg grad shape " << grad_shape->ToString()
                               << " are not consistent with var shape " << var_shape->ToString();
    }
  }

  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{var_shape, m_shape, v_shape});
}

TuplePtr ApplyAdaMaxInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();
  const int64_t kInputNum = 9;
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kGreaterEqual, kInputNum,
                                           prim_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto var_type = input_args[kInputIndex0]->BuildType();
  auto m_type = input_args[kInputIndex1]->BuildType();
  auto v_type = input_args[kInputIndex2]->BuildType();
  auto beta1_power_type = input_args[kInputIndex3]->BuildType();
  auto lr_type = input_args[kInputIndex4]->BuildType();
  auto beta1_type = input_args[kInputIndex5]->BuildType();
  auto beta2_type = input_args[kInputIndex6]->BuildType();
  auto epsilon_type = input_args[kInputIndex7]->BuildType();
  auto grad_type = input_args[kInputIndex8]->BuildType();
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32};
  // m v grad must have the same type as var
  std::map<std::string, TypePtr> args;
  (void)args.insert({"var_type", var_type});
  (void)args.insert({"m_type", m_type});
  (void)args.insert({"v_type", v_type});
  (void)args.insert({"grad_type", grad_type});
  (void)CheckAndConvertUtils::CheckTensorTypeSame(args, valid_types, prim_name);

  std::map<std::string, TypePtr> args_beta1_power;
  std::map<std::string, TypePtr> args_lr;
  std::map<std::string, TypePtr> args_beta1;
  std::map<std::string, TypePtr> args_beta2;
  std::map<std::string, TypePtr> args_epsilon;

  (void)args_beta1_power.insert({"beta1_power_type", beta1_power_type});
  (void)args_lr.insert({"lr_type", lr_type});
  (void)args_beta1.insert({"beta1_type", beta1_type});
  (void)args_beta2.insert({"beta2_type", beta2_type});
  (void)args_epsilon.insert({"epsilon_type", epsilon_type});

  // beta1_power,lr,beta1,beta2,epsilon must be a scalar or zero dimension tensor type
  (void)CheckAndConvertUtils::CheckScalarOrTensorTypesSame(args_beta1_power, valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckScalarOrTensorTypesSame(args_lr, valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckScalarOrTensorTypesSame(args_beta1, valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckScalarOrTensorTypesSame(args_beta2, valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckScalarOrTensorTypesSame(args_epsilon, valid_types, prim_name);

  return std::make_shared<Tuple>(std::vector<TypePtr>{var_type, m_type, v_type});
}
}  // namespace

AbstractBasePtr ApplyAdaMaxInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                 const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto infer_type = ApplyAdaMaxInferType(primitive, input_args);
  auto infer_shape = ApplyAdaMaxInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

REGISTER_PRIMITIVE_EVAL_IMPL(ApplyAdaMax, prim::kPrimApplyAdaMax, ApplyAdaMaxInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
