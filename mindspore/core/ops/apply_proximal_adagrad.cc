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

#include "ops/apply_proximal_adagrad.h"

#include <algorithm>
#include <set>

#include "ops/op_utils.h"
#include "abstract/primitive_infer_map.h"
#include "utils/tensor_construct_utils.h"
#include "utils/check_convert_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::TupleShapePtr ApplyProximalAdagradInferShape(const PrimitivePtr &primitive,
                                                       const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t kInputNum = 6;
  (void)CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, kInputNum, primitive->name());
  auto var_shape_ptr = input_args[kInputIndex0]->BuildShape();
  auto accum_shape_ptr = input_args[kInputIndex1]->BuildShape();
  auto lr_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->BuildShape())[kShape];
  auto l1_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex3]->BuildShape())[kShape];
  auto l2_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex4]->BuildShape())[kShape];
  auto grad_shape_ptr = input_args[kInputIndex5]->BuildShape();
  // lr, l1, l2 should be scalar or size equal with 1
  (void)CheckAndConvertUtils::CheckInteger("lr_shape size", lr_shape.size(), kLessEqual, 1, prim_name);
  (void)CheckAndConvertUtils::CheckInteger("l1_shape size", l1_shape.size(), kLessEqual, 1, prim_name);
  (void)CheckAndConvertUtils::CheckInteger("l2_shape size", l2_shape.size(), kLessEqual, 1, prim_name);
  if (lr_shape.size() == 1) {
    (void)CheckAndConvertUtils::CheckInteger("lr_shape's first rank must be 1", lr_shape[0], kEqual, 1, prim_name);
  }
  if (l1_shape.size() == 1) {
    (void)CheckAndConvertUtils::CheckInteger("l1_shape's first rank must be 1", l1_shape[0], kEqual, 1, prim_name);
  }
  if (l2_shape.size() == 1) {
    (void)CheckAndConvertUtils::CheckInteger("l2_shape's first rank must be 1", l2_shape[0], kEqual, 1, prim_name);
  }
  if (grad_shape_ptr->IsDynamic()) {
    return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{var_shape_ptr, accum_shape_ptr});
  }
  // var, accum and grad must have the same shape
  std::map<std::string, abstract::BaseShapePtr> same_shape_args_map;
  same_shape_args_map.insert({"accum", accum_shape_ptr});
  same_shape_args_map.insert({"grad", grad_shape_ptr});
  for (auto &elem : same_shape_args_map) {
    if (*elem.second != *var_shape_ptr) {
      MS_EXCEPTION(ValueError) << prim_name << " evaluator arg " << elem.first << " shape " << elem.second->ToString()
                               << " are not consistent with var shape " << var_shape_ptr->ToString();
    }
  }
  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{var_shape_ptr, accum_shape_ptr});
}

TuplePtr ApplyProximalAdagradInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t kInputNum = 6;
  (void)CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, kInputNum, primitive->name());
  auto var_type = input_args[kInputIndex0]->BuildType();
  auto accum_type = input_args[kInputIndex1]->BuildType();
  auto lr_type = input_args[kInputIndex2]->BuildType();
  auto l1_type = input_args[kInputIndex3]->BuildType();
  auto l2_type = input_args[kInputIndex4]->BuildType();
  auto grad_type = input_args[kInputIndex5]->BuildType();
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32};
  // var, accum and grad must have the same type
  std::map<std::string, TypePtr> args;
  args.insert({"var", var_type});
  args.insert({"accum", accum_type});
  args.insert({"grad", grad_type});
  (void)CheckAndConvertUtils::CheckTensorTypeSame(args, valid_types, prim_name);
  // lr, l1, l2 type must be valid
  std::map<std::string, TypePtr> args_lr;
  args_lr.insert({"lr", lr_type});
  std::map<std::string, TypePtr> args_l1;
  args_l1.insert({"l1", l1_type});
  std::map<std::string, TypePtr> args_l2;
  args_l2.insert({"l2", l2_type});
  (void)CheckAndConvertUtils::CheckScalarOrTensorTypesSame(args_lr, valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckScalarOrTensorTypesSame(args_l1, valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckScalarOrTensorTypesSame(args_l2, valid_types, prim_name);
  return std::make_shared<Tuple>(std::vector<TypePtr>{var_type, accum_type});
}
}  // namespace

MIND_API_BASE_IMPL(ApplyProximalAdagrad, PrimitiveC, BaseOperator);
AbstractBasePtr ApplyProximalAdagradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto infer_type = ApplyProximalAdagradInferType(primitive, input_args);
  auto infer_shape = ApplyProximalAdagradInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(ApplyProximalAdagrad, prim::kPrimApplyProximalAdagrad, ApplyProximalAdagradInfer, nullptr,
                             true);
}  // namespace ops
}  // namespace mindspore
