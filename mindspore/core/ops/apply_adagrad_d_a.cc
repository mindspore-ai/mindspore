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

#include "ops/apply_adagrad_d_a.h"

#include <algorithm>
#include <set>

#include "abstract/primitive_infer_map.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/tensor_construct_utils.h"

namespace mindspore {
namespace ops {
namespace {
abstract::TupleShapePtr InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 8;
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, input_num,
                                           primitive->name());
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto var_shape = input_args[0]->BuildShape();
  auto gradient_accumulator_shape = input_args[kInputIndex1]->BuildShape();
  auto gradient_squared_accumulator_shape = input_args[kInputIndex2]->BuildShape();
  auto lr_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex4]->BuildShape())[kShape];
  auto l1_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex5]->BuildShape())[kShape];
  auto l2_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex6]->BuildShape())[kShape];
  auto global_step_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex7]->BuildShape())[kShape];
  const int64_t input_nums = 0;
  (void)CheckAndConvertUtils::CheckInteger("lr_shape size", SizeToInt(lr_shape.size()), kEqual, input_nums,
                                           primitive->name());
  (void)CheckAndConvertUtils::CheckInteger("l1_shape size", SizeToInt(l1_shape.size()), kEqual, input_nums,
                                           primitive->name());
  (void)CheckAndConvertUtils::CheckInteger("l2_shape size", SizeToInt(l2_shape.size()), kEqual, input_nums,
                                           primitive->name());
  (void)CheckAndConvertUtils::CheckInteger("global_step_shape size", SizeToInt(global_step_shape.size()), kEqual,
                                           input_nums, primitive->name());
  return std::make_shared<abstract::TupleShape>(
    std::vector<abstract::BaseShapePtr>{var_shape, gradient_accumulator_shape, gradient_squared_accumulator_shape});
}

TuplePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();
  const int64_t input_num = 8;
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, input_num, prim_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto var_type = input_args[kInputIndex0]->BuildType();
  auto gradient_accumulator_type = input_args[kInputIndex1]->BuildType();
  auto gradient_squared_accumulator_type = input_args[kInputIndex2]->BuildType();
  auto grad_type = input_args[kInputIndex3]->BuildType();
  auto lr_type = input_args[kInputIndex4]->BuildType();
  auto l1_type = input_args[kInputIndex5]->BuildType();
  auto l2_type = input_args[kInputIndex6]->BuildType();
  auto global_step_type = input_args[kInputIndex7]->BuildType();
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32};
  // gradient_accumulator、gradient_squared_accumulator、grad must have the same type as var
  std::map<std::string, TypePtr> args;
  (void)args.insert(std::make_pair("var_type", var_type));
  (void)args.insert(std::make_pair("gradient_accumulator_type", gradient_accumulator_type));
  (void)args.insert(std::make_pair("gradient_squared_accumulator_type", gradient_squared_accumulator_type));
  (void)args.insert(std::make_pair("grad_type", grad_type));
  (void)CheckAndConvertUtils::CheckTensorTypeSame(args, valid_types, prim_name);
  // lr、l1、l2、global_step_type must be a scalar type
  std::map<std::string, TypePtr> args_lr;
  std::map<std::string, TypePtr> args_l1;
  std::map<std::string, TypePtr> args_l2;
  std::map<std::string, TypePtr> args_global_step;
  (void)args_lr.insert(std::make_pair("lr_type", lr_type));
  (void)CheckAndConvertUtils::CheckScalarOrTensorTypesSame(args_lr, valid_types, prim_name);
  (void)args_l1.insert(std::make_pair("l1_type", l1_type));
  (void)CheckAndConvertUtils::CheckScalarOrTensorTypesSame(args_l1, valid_types, prim_name);
  (void)args_l2.insert(std::make_pair("l2_type", l2_type));
  (void)CheckAndConvertUtils::CheckScalarOrTensorTypesSame(args_l2, valid_types, prim_name);
  (void)args_global_step.insert(std::make_pair("global_step_type", global_step_type));
  const std::set<TypePtr> valid_types1 = {kInt32, kInt64};
  (void)CheckAndConvertUtils::CheckScalarOrTensorTypesSame(args_global_step, valid_types1, prim_name);
  return std::make_shared<Tuple>(
    std::vector<TypePtr>{var_type, gradient_accumulator_type, gradient_squared_accumulator_type});
}
}  // namespace

AbstractBasePtr ApplyAdagradDAInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  return abstract::MakeAbstract(InferShape(primitive, input_args), InferType(primitive, input_args));
}

REGISTER_PRIMITIVE_EVAL_IMPL(ApplyAdagradDA, prim::kPrimApplyAdagradDA, ApplyAdagradDAInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
