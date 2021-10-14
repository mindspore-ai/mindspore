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

#include "ops/apply_keras_momentum.h"

#include <algorithm>
#include <set>

#include "ops/op_utils.h"
#include "abstract/primitive_infer_map.h"
#include "utils/tensor_construct_utils.h"

namespace mindspore {
namespace ops {
namespace {
abstract::TupleShapePtr InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, 5, prim_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto var_shape = input_args[0]->BuildShape();
  auto accum_shape = input_args[1]->BuildShape();
  auto lr_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[2]->GetShapeTrack())[kShape];
  auto grad_shape = input_args[3]->BuildShape();
  auto momentum_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[4]->GetShapeTrack())[kShape];
  // lr, momentum must be scalar
  (void)CheckAndConvertUtils::CheckInteger("lr_shape size", lr_shape.size(), kEqual, 0, prim_name);
  (void)CheckAndConvertUtils::CheckInteger("momentum_shape size", momentum_shape.size(), kEqual, 0, prim_name);
  // var, accum and grad must have the same shape
  std::map<std::string, abstract::BaseShapePtr> same_shape_args_map;
  same_shape_args_map.insert({"accum", accum_shape});
  same_shape_args_map.insert({"grad", grad_shape});
  for (auto &elem : same_shape_args_map) {
    if (*elem.second != *var_shape) {
      MS_EXCEPTION(ValueError) << prim_name << " evaluator arg " << elem.first << " shape " << elem.second->ToString()
                               << " are not consistent with var shape " << var_shape->ToString();
    }
  }
  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{var_shape, accum_shape});
}

TuplePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, 5, prim_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto var_type = input_args[0]->BuildType();
  auto accum_type = input_args[1]->BuildType();
  auto lr_type = input_args[2]->BuildType();
  auto grad_type = input_args[3]->BuildType();
  auto momentum_type = input_args[4]->BuildType();
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32};
  // var, accum and grad must have the same type
  std::map<std::string, TypePtr> args;
  args.insert({"var", var_type});
  args.insert({"accum", accum_type});
  args.insert({"grad", grad_type});
  (void)CheckAndConvertUtils::CheckTensorTypeSame(args, valid_types, prim_name);
  // lr, momentum type must be valid
  (void)CheckAndConvertUtils::CheckTensorTypeValid("lr_dtype", lr_type, valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("momentum_dtype", momentum_type, valid_types, prim_name);
  return std::make_shared<Tuple>(std::vector<TypePtr>{var_type, accum_type});
}
}  // namespace

AbstractBasePtr ApplyKerasMomentumInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                        const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  return abstract::MakeAbstract(InferShape(primitive, input_args), InferType(primitive, input_args));
}
REGISTER_PRIMITIVE_EVAL_IMPL(ApplyKerasMomentum, prim::kPrimApplyKerasMomentum, ApplyKerasMomentumInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
