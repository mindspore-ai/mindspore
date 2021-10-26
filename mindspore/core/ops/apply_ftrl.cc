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

#include "ops/apply_ftrl.h"

#include <algorithm>
#include <set>

#include "abstract/primitive_infer_map.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/tensor_construct_utils.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr ApplyFtrlInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t kInputNum = 8;
  (void)CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, kInputNum, prim_name);
  auto var_shape = input_args[kInputIndex0]->BuildShape();
  auto accum_shape = input_args[kInputIndex1]->BuildShape();
  auto linear_shape = input_args[kInputIndex2]->BuildShape();
  if (var_shape->IsDynamic() || accum_shape->IsDynamic() || linear_shape->IsDynamic()) {
    return var_shape->cast<abstract::ShapePtr>();
  }
  std::map<std::string, abstract::BaseShapePtr> same_shape_args_map;
  same_shape_args_map.insert({"accum", accum_shape});
  same_shape_args_map.insert({"linear", linear_shape});
  for (auto &elem : same_shape_args_map) {
    if (*elem.second != *var_shape) {
      MS_EXCEPTION(ValueError) << prim_name << " evaluator arg " << elem.first << " shape " << elem.second->ToString()
                               << " are not consistent with var shape " << var_shape->ToString();
    }
  }
  auto shape_ptr = var_shape->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(shape_ptr);
  return shape_ptr;
}
TypePtr ApplyFtrlInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();
  const int64_t kInputNum = 8;
  (void)CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, kInputNum, prim_name);
  auto var_type = input_args[kInputIndex0]->BuildType();
  auto accum_type = input_args[kInputIndex1]->BuildType();
  auto linear_type = input_args[kInputIndex2]->BuildType();
  auto grad_type = input_args[kInputIndex3]->BuildType();
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32};
  std::map<std::string, TypePtr> args;
  (void)args.insert({"var_type", var_type});
  (void)args.insert({"accum_type", accum_type});
  (void)args.insert({"linear_type", linear_type});
  (void)args.insert({"grad_type", grad_type});
  // var accum linear grad must have same dtypes
  (void)CheckAndConvertUtils::CheckTensorTypeSame(args, valid_types, prim_name);

  auto lr_type = input_args[kInputIndex4]->BuildType();
  auto l1_type = input_args[kInputIndex5]->BuildType();
  auto l2_type = input_args[kInputIndex6]->BuildType();
  auto lr_power_type = input_args[kInputIndex7]->BuildType();
  std::map<std::string, TypePtr> args_lr;
  std::map<std::string, TypePtr> args_l1;
  std::map<std::string, TypePtr> args_l2;
  std::map<std::string, TypePtr> args_lr_power;
  (void)args_lr.insert({"lr_type", lr_type});
  (void)args_l1.insert({"l1_type", l1_type});
  (void)args_l2.insert({"l2_type", l2_type});
  (void)args_lr_power.insert({"lr_power_type", lr_power_type});

  // lr, l1, l2, lr_power type must be float or scalar tensor with float
  (void)CheckAndConvertUtils::CheckScalarOrTensorTypesSame(args_lr, valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckScalarOrTensorTypesSame(args_l1, valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckScalarOrTensorTypesSame(args_l2, valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckScalarOrTensorTypesSame(args_lr_power, valid_types, prim_name);

  return var_type;
}
}  // namespace
AbstractBasePtr ApplyFtrlInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                               const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t kInputNum = 8;
  (void)CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, kInputNum, prim_name);
  auto infer_type = ApplyFtrlInferType(primitive, input_args);
  auto infer_shape = ApplyFtrlInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(ApplyFtrl, prim::kPrimApplyFtrl, ApplyFtrlInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
