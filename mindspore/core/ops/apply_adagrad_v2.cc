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

#include "ops/apply_adagrad_v2.h"

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
abstract::TupleShapePtr ApplyAdagradV2InferShape(const PrimitivePtr &primitive,
                                                 const std::vector<AbstractBasePtr> &input_args) {
  auto var_shape = input_args[kInputIndex0]->BuildShape();
  auto accum_shape = input_args[kInputIndex1]->BuildShape();
  auto lr_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->BuildShape())[kShape];
  auto grad_shape = input_args[kInputIndex3]->BuildShape();
  auto grad_shape_ptr = grad_shape->cast<abstract::ShapePtr>();
  // lr must be a scalar [Number, Tensor]
  const int64_t kShapeSize_ = 1;
  auto lr_shape_size = lr_shape.size();
  (void)CheckAndConvertUtils::CheckInteger("lr's rank'", lr_shape_size, kLessEqual, kShapeSize_, primitive->name());
  if (lr_shape_size == 1) {
    (void)CheckAndConvertUtils::CheckInteger("lr_shape[0]", lr_shape[0], kEqual, kShapeSize_, primitive->name());
  }
  // var, accum and grad must have the same shape
  if (grad_shape_ptr->IsDynamic()) {
    return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{var_shape, accum_shape});
  }
  std::map<std::string, abstract::BaseShapePtr> same_shape_args_map;
  same_shape_args_map.insert({"accum", accum_shape});
  same_shape_args_map.insert({"grad", grad_shape});
  for (auto &elem : same_shape_args_map) {
    if (*elem.second != *var_shape) {
      MS_EXCEPTION(ValueError) << primitive->name() << " evaluator arg " << elem.first << " shape "
                               << elem.second->ToString() << " are not consistent with var shape "
                               << var_shape->ToString();
    }
  }
  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{var_shape, accum_shape});
}
TuplePtr ApplyAdagradV2InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto var_type = input_args[kInputIndex0]->BuildType();
  auto accum_type = input_args[kInputIndex1]->BuildType();
  auto lr_type = input_args[kInputIndex2]->BuildType();
  auto grad_type = input_args[kInputIndex3]->BuildType();
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32};
  // var, accum, grad  must have the same type
  std::map<std::string, TypePtr> args;
  (void)args.insert({"var_type", var_type});
  (void)args.insert({"accum_type", accum_type});
  (void)args.insert({"grad_type", grad_type});
  (void)CheckAndConvertUtils::CheckTensorTypeSame(args, valid_types, prim->name());
  // lr mustr be a scalar
  std::map<std::string, TypePtr> args_lr;
  (void)args_lr.insert({"lr_type", lr_type});
  (void)CheckAndConvertUtils::CheckScalarOrTensorTypesSame(args_lr, valid_types, prim->name());
  return std::make_shared<Tuple>(std::vector<TypePtr>{var_type, accum_type});
}
}  // namespace

MIND_API_BASE_IMPL(ApplyAdagradV2, PrimitiveC, BaseOperator);
AbstractBasePtr ApplyAdagradV2Infer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputNum = 4;
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kGreaterEqual, kInputNum,
                                           primitive->name());
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto infer_type = ApplyAdagradV2InferType(primitive, input_args);
  auto infer_shape = ApplyAdagradV2InferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

REGISTER_PRIMITIVE_EVAL_IMPL(ApplyAdagradV2, prim::kPrimApplyAdagradV2, ApplyAdagradV2Infer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
