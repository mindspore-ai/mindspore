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

#include "ops/sparse_apply_adagrad_da.h"

#include <algorithm>
#include <set>

#include "abstract/ops/primitive_infer_map.h"
#include "ops/op_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr SparseApplyAdagradDAInferShape(const PrimitivePtr &primitive,
                                                  const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto var_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  auto grad_accum_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape())[kShape];
  auto grad_square_accum_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[2]->BuildShape())[kShape];
  auto grad_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[3]->BuildShape())[kShape];
  auto indices_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[4]->BuildShape())[kShape];
  auto lr_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[5]->GetShapeTrack())[kShape];
  auto l1_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[6]->GetShapeTrack())[kShape];
  auto l2_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[7]->GetShapeTrack())[kShape];
  auto global_step_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[8]->GetShapeTrack())[kShape];

  std::vector<ShapeVector> scalar_shapes = {lr_shape, l1_shape, l2_shape, global_step_shape};
  auto is_dynamic_scalar = std::any_of(scalar_shapes.begin(), scalar_shapes.end(), IsDynamic);
  if (!is_dynamic_scalar) {
    int64_t scalar_shape = 0;
    (void)CheckAndConvertUtils::CheckInteger("lr_shape size", lr_shape.size(), kEqual, scalar_shape, prim_name);
    (void)CheckAndConvertUtils::CheckInteger("l1_shape size", l1_shape.size(), kEqual, scalar_shape, prim_name);
    (void)CheckAndConvertUtils::CheckInteger("l2_shape size", l2_shape.size(), kEqual, scalar_shape, prim_name);
    (void)CheckAndConvertUtils::CheckInteger("global_step_shape size", global_step_shape.size(), kEqual, scalar_shape,
                                             prim_name);
  }

  std::vector<ShapeVector> check_tensor_shapes = {var_shape, grad_accum_shape, grad_square_accum_shape};
  auto is_dynamic_tensor = std::any_of(check_tensor_shapes.begin(), check_tensor_shapes.end(), IsDynamic);
  if (!is_dynamic_tensor) {
    std::map<std::string, ShapeVector> same_shape_args_map;
    (void)same_shape_args_map.emplace("shape of grad_accum", grad_accum_shape);
    (void)same_shape_args_map.emplace("shape of grad_square_accum", grad_square_accum_shape);
    for (auto &elem : same_shape_args_map) {
      CheckAndConvertUtils::Check(elem.first, elem.second, kEqual, var_shape, prim_name);
    }
  }

  // Var dimension must be equal or greater than 1.
  (void)CheckAndConvertUtils::CheckInteger("var dimension", SizeToLong(var_shape.size()), kGreaterEqual, 1, prim_name);
  // Indices must be rank 1.
  (void)CheckAndConvertUtils::CheckInteger("indices dimension", SizeToLong(indices_shape.size()), kEqual, 1, prim_name);
  auto is_dynamic = IsDynamic(var_shape) || IsDynamic(grad_shape) || IsDynamic(indices_shape);
  if (!is_dynamic) {
    (void)CheckAndConvertUtils::CheckInteger("rank(grad) and rank(var)", SizeToLong(grad_shape.size()), kEqual,
                                             SizeToLong(var_shape.size()), prim_name);
    (void)CheckAndConvertUtils::CheckInteger("grad.shape[0] and indices.shape[0]", SizeToLong(indices_shape[0]), kEqual,
                                             SizeToLong(grad_shape[0]), prim_name);
    for (size_t i = 1; i < var_shape.size(); ++i) {
      if (var_shape[i] != grad_shape[i]) {
        MS_EXCEPTION(ValueError) << "For '" << prim_name << "', the shape of var and grad must equal in dimension " << i
                                 << ".";
      }
    }
  }

  return std::make_shared<abstract::Shape>(var_shape);
}

TypePtr SparseApplyAdagradDAInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto var = input_args[0]->BuildType();
  auto grad_accum = input_args[1]->BuildType();
  auto grad_square_accum = input_args[2]->BuildType();
  auto grad = input_args[3]->BuildType();
  auto indices = input_args[4]->BuildType();
  auto lr = input_args[5]->BuildType();
  auto l1 = input_args[6]->BuildType();
  auto l2 = input_args[7]->BuildType();
  auto global_step = input_args[8]->BuildType();

  std::map<std::string, TypePtr> args;
  (void)args.emplace("var", var);
  (void)args.emplace("grad_accum", grad_accum);
  (void)args.emplace("grad_square_accum", grad_square_accum);
  (void)args.emplace("grad", grad);
  (void)args.emplace("lr", lr);
  (void)args.emplace("l1", l1);
  (void)args.emplace("l2", l2);
  (void)CheckAndConvertUtils::CheckScalarOrTensorTypesSame(args, common_valid_types, prim_name);

  const std::set<TypePtr> valids1 = {kInt32, kInt64};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("indices", indices, valids1, prim_name);

  std::map<std::string, TypePtr> args_global_step;
  (void)args_global_step.emplace("global_step", global_step);
  const std::set<TypePtr> valids2 = {kInt64};
  (void)CheckAndConvertUtils::CheckScalarOrTensorTypesSame(args_global_step, valids2, prim_name);
  return var;
}
}  // namespace

AbstractBasePtr SparseApplyAdagradDAInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int Inputs_num = 9;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, Inputs_num, primitive->name());
  auto infer_type = SparseApplyAdagradDAInferType(primitive, input_args);
  auto infer_shape = SparseApplyAdagradDAInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
MIND_API_OPERATOR_IMPL(SparseApplyAdagradDA, BaseOperator);
REGISTER_PRIMITIVE_EVAL_IMPL(SparseApplyAdagradDA, prim::kPrimSparseApplyAdagradDA, SparseApplyAdagradDAInfer, nullptr,
                             true);
}  // namespace ops
}  // namespace mindspore
