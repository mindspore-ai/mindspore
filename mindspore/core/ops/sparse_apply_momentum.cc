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

#include "ops/sparse_apply_momentum.h"

#include <algorithm>
#include <set>

#include "abstract/ops/primitive_infer_map.h"
#include "ops/op_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr SparseApplyMomentumInferShape(const PrimitivePtr &primitive,
                                                 const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto var_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  auto accum_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape())[kShape];
  auto lr_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[2]->GetShapeTrack())[kShape];
  auto grad_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[3]->BuildShape())[kShape];
  auto indices_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[4]->BuildShape())[kShape];
  auto momentum_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[5]->GetShapeTrack())[kShape];

  auto scalar_shape = 0;
  (void)CheckAndConvertUtils::CheckInteger("lr_shape size", SizeToLong(lr_shape.size()), kEqual, scalar_shape,
                                           prim_name);
  (void)CheckAndConvertUtils::CheckInteger("momentum_shape size", SizeToLong(momentum_shape.size()), kEqual,
                                           scalar_shape, prim_name);

  std::map<std::string, ShapeVector> same_shape_args_map;
  (void)same_shape_args_map.emplace("shape of accum", accum_shape);
  for (auto &elem : same_shape_args_map) {
    CheckAndConvertUtils::Check(elem.first, elem.second, kEqual, var_shape, prim_name);
  }

  // Var dimension must be equal or greater than 1.
  (void)CheckAndConvertUtils::CheckInteger("var dimension", SizeToLong(var_shape.size()), kGreaterEqual, 1, prim_name);

  if (var_shape.size() != grad_shape.size()) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name
                             << "', rank(grad) should be same as rank(var), but got rank(grad): " << grad_shape.size()
                             << ", rank(var): " << var_shape.size() << ".";
  }

  for (size_t i = 1; i < var_shape.size(); ++i) {
    if (var_shape[i] != grad_shape[i]) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name << "', the shape of var and grad must equal in dimension " << i
                               << ".";
    }
  }

  // Indices must be rank 1.
  (void)CheckAndConvertUtils::CheckInteger("indices dimension", SizeToLong(indices_shape.size()), kEqual, 1, prim_name);
  if (indices_shape[0] != grad_shape[0]) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name
                             << "', grad.shape[0] must be equal to indices.shape[0], but got grad.shape[0]: "
                             << grad_shape[0] << " indices.shape[0]: " << indices_shape[0] << ".";
  }

  return std::make_shared<abstract::Shape>(var_shape);
}

TypePtr SparseApplyMomentumInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto var_type = input_args[0]->BuildType();
  auto accum_type = input_args[1]->BuildType();
  auto lr_type = input_args[2]->BuildType();
  auto grad_type = input_args[3]->BuildType();
  auto indices_type = input_args[4]->BuildType();
  auto momentum_type = input_args[5]->BuildType();

  std::map<std::string, TypePtr> args;
  (void)args.emplace("var", var_type);
  (void)args.emplace("accum", accum_type);
  (void)args.emplace("grad", grad_type);
  (void)args.emplace("lr", lr_type);
  (void)args.emplace("momentum", momentum_type);
  (void)CheckAndConvertUtils::CheckScalarOrTensorTypesSame(args, common_valid_types, prim_name);

  const std::set<TypePtr> valid_types2 = {kInt32, kInt64};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("indices", indices_type, valid_types2, prim_name);
  return var_type;
}
}  // namespace

MIND_API_OPERATOR_IMPL(SparseApplyMomentum, BaseOperator);
void SparseApplyMomentum::Init(const bool use_locking, const bool use_nesterov) {
  this->set_use_locking(use_locking);
  this->set_use_nesterov(use_nesterov);
}

void SparseApplyMomentum::set_use_locking(const bool use_locking) {
  (void)this->AddAttr(kUseLocking, api::MakeValue(use_locking));
}

void SparseApplyMomentum::set_use_nesterov(const bool use_nesterov) {
  (void)this->AddAttr(kUseNesterov, api::MakeValue(use_nesterov));
}

bool SparseApplyMomentum::get_use_locking() const {
  auto value_ptr = GetAttr(kUseLocking);
  return GetValue<bool>(value_ptr);
}

bool SparseApplyMomentum::get_use_nesterov() const {
  auto value_ptr = GetAttr(kUseNesterov);
  return GetValue<bool>(value_ptr);
}

AbstractBasePtr SparseApplyMomentumInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t Inputs_num = 6;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, Inputs_num, primitive->name());
  auto infer_type = SparseApplyMomentumInferType(primitive, input_args);
  auto infer_shape = SparseApplyMomentumInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(SparseApplyMomentum, prim::kPrimSparseApplyMomentum, SparseApplyMomentumInfer, nullptr,
                             true);
}  // namespace ops
}  // namespace mindspore
