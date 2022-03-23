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

#include "ops/apply_add_sign.h"
#include <algorithm>
#include <set>
#include "abstract/primitive_infer_map.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/tensor_construct_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::TupleShapePtr ApplyAddSignInferShape(const PrimitivePtr &primitive,
                                               const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t kInputNum = 7;
  (void)CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, kInputNum, primitive->name());
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto var_shape = input_args[kInputIndex0]->BuildShape();
  auto m_shape = input_args[kInputIndex1]->BuildShape();
  auto lr_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->BuildShape())[kShape];
  auto alpha_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex3]->BuildShape())[kShape];
  auto sign_decay_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex4]->BuildShape())[kShape];
  auto beta_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex5]->BuildShape())[kShape];
  auto grad_shape = input_args[kInputIndex6]->BuildShape();
  auto var_shape_ptr = var_shape->cast<abstract::ShapePtr>();
  auto m_shape_ptr = m_shape->cast<abstract::ShapePtr>();
  auto grad_shape_ptr = grad_shape->cast<abstract::ShapePtr>();
  if (!m_shape_ptr->IsDynamic() && !var_shape_ptr->IsDynamic()) {
    if (*m_shape != *var_shape) {
      MS_EXCEPTION(ValueError) << prim_name << " evaluator arg m shape " << m_shape->ToString()
                               << " are not consistent with var shape " << var_shape->ToString();
    }
  }
  if (!grad_shape_ptr->IsDynamic() && !var_shape_ptr->IsDynamic()) {
    if (*grad_shape != *var_shape) {
      MS_EXCEPTION(ValueError) << prim_name << " evaluator arg grad shape " << grad_shape->ToString()
                               << " are not consistent with var shape " << var_shape->ToString();
    }
  }
  const int64_t kShapeSize = 1;
  (void)CheckAndConvertUtils::CheckInteger("lr_shape_size", lr_shape.size(), kLessEqual, kShapeSize, prim_name);
  if (lr_shape.size() == 1) {
    (void)CheckAndConvertUtils::CheckInteger("lr_shape[0]", lr_shape[0], kEqual, kShapeSize, prim_name);
  }
  (void)CheckAndConvertUtils::CheckInteger("alpha_shape_size", alpha_shape.size(), kLessEqual, kShapeSize, prim_name);
  if (alpha_shape.size() == 1) {
    (void)CheckAndConvertUtils::CheckInteger("alpha_shape[0]", alpha_shape[0], kEqual, kShapeSize, prim_name);
  }
  (void)CheckAndConvertUtils::CheckInteger("sign_decay_shape_size", sign_decay_shape.size(), kLessEqual, kShapeSize,
                                           prim_name);
  if (sign_decay_shape.size() == 1) {
    (void)CheckAndConvertUtils::CheckInteger("sign_decay_shape[0]", sign_decay_shape[0], kEqual, kShapeSize, prim_name);
  }
  (void)CheckAndConvertUtils::CheckInteger("beta_shape_size", beta_shape.size(), kLessEqual, kShapeSize, prim_name);
  if (beta_shape.size() == 1) {
    (void)CheckAndConvertUtils::CheckInteger("beta_shape[0]", beta_shape[0], kEqual, kShapeSize, prim_name);
  }
  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{var_shape, m_shape});
}

TuplePtr ApplyAddSignInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();
  const int64_t kInputNum = 7;
  (void)CheckAndConvertUtils::CheckInteger("Input number", SizeToLong(input_args.size()), kGreaterEqual, kInputNum,
                                           prim_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto var_type = input_args[kInputIndex0]->BuildType();
  auto m_type = input_args[kInputIndex1]->BuildType();
  auto lr_type = input_args[kInputIndex2]->BuildType();
  auto alpha_type = input_args[kInputIndex3]->BuildType();
  auto sign_decay_type = input_args[kInputIndex4]->BuildType();
  auto beta_type = input_args[kInputIndex5]->BuildType();
  auto grad_type = input_args[kInputIndex6]->BuildType();
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32};
  std::map<std::string, TypePtr> args;
  (void)args.insert({"var_type", var_type});
  (void)args.insert({"m_type", m_type});
  (void)args.insert({"grad_type", grad_type});
  (void)CheckAndConvertUtils::CheckScalarOrTensorTypesSame(args, valid_types, prim_name);
  std::map<std::string, TypePtr> args_lr;
  (void)args_lr.insert({"lr_type", lr_type});
  (void)CheckAndConvertUtils::CheckScalarOrTensorTypesSame(args_lr, valid_types, prim_name);
  std::map<std::string, TypePtr> args_alpha;
  (void)args_alpha.insert({"alpha_type", alpha_type});
  (void)CheckAndConvertUtils::CheckScalarOrTensorTypesSame(args_alpha, valid_types, prim_name);
  std::map<std::string, TypePtr> args_sign_decay;
  (void)args_sign_decay.insert({"sign_decay_type", sign_decay_type});
  (void)CheckAndConvertUtils::CheckScalarOrTensorTypesSame(args_sign_decay, valid_types, prim_name);
  std::map<std::string, TypePtr> args_beta;
  (void)args_beta.insert({"beta_type", beta_type});
  (void)CheckAndConvertUtils::CheckScalarOrTensorTypesSame(args_beta, valid_types, prim_name);
  return std::make_shared<Tuple>(std::vector<TypePtr>{var_type, m_type});
}
}  // namespace

MIND_API_BASE_IMPL(ApplyAddSign, PrimitiveC, BaseOperator);
AbstractBasePtr ApplyAddSignInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto infer_type = ApplyAddSignInferType(primitive, input_args);
  auto infer_shape = ApplyAddSignInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

REGISTER_PRIMITIVE_EVAL_IMPL(ApplyAddSign, prim::kPrimApplyAddSign, ApplyAddSignInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
