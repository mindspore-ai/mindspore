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

#include "ops/apply_adagrad.h"

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
abstract::TupleShapePtr ApplyAdagradInferShape(const PrimitivePtr &primitive,
                                               const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t kInputNum = 4;
  (void)CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, kInputNum, primitive->name());
  auto lr_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->BuildShape())[kShape];
  auto var_shape_ptr = input_args[kInputIndex0]->BuildShape();
  auto accum_shape_ptr = input_args[kInputIndex1]->BuildShape();
  auto grad_shape_ptr = input_args[kInputIndex3]->BuildShape();
  // lr should be scalar or size equal with 1
  (void)CheckAndConvertUtils::CheckInteger("lr_shape size", lr_shape.size(), kLessEqual, 1, prim_name);
  if (lr_shape.size() == 1) {
    (void)CheckAndConvertUtils::CheckInteger("lr_shape's first rank must be 1", lr_shape[0], kEqual, 1, prim_name);
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

TuplePtr ApplyAdagradInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t kInputNum = 4;
  (void)CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, kInputNum, primitive->name());
  auto var_type = input_args[kInputIndex0]->BuildType();
  auto accum_type = input_args[kInputIndex1]->BuildType();
  auto lr_type = input_args[kInputIndex2]->BuildType();
  auto grad_type = input_args[kInputIndex3]->BuildType();
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32};
  // var, accum and grad must have the same type
  std::map<std::string, TypePtr> args;
  args.insert({"var", var_type});
  args.insert({"accum", accum_type});
  args.insert({"grad", grad_type});
  (void)CheckAndConvertUtils::CheckTensorTypeSame(args, valid_types, prim_name);
  // lr type must be valid
  std::map<std::string, TypePtr> args_lr;
  args_lr.insert({"lr", lr_type});
  (void)CheckAndConvertUtils::CheckScalarOrTensorTypesSame(args_lr, valid_types, prim_name);
  return std::make_shared<Tuple>(std::vector<TypePtr>{var_type, accum_type});
}
}  // namespace

MIND_API_BASE_IMPL(ApplyAdagrad, PrimitiveC, BaseOperator);
AbstractBasePtr ApplyAdagradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto infer_type = ApplyAdagradInferType(primitive, input_args);
  auto infer_shape = ApplyAdagradInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(ApplyAdagrad, prim::kPrimApplyAdagrad, ApplyAdagradInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
