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

#include <algorithm>
#include <set>
#include <map>

#include "ops/sparse_apply_proximal_adagrad.h"
#include "abstract/ops/primitive_infer_map.h"
#include "ops/op_name.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
void SparseApplyProximalAdagradCheckTensorShapeAndSize(const ShapeVector &var_shape, const ShapeVector &accum_shape,
                                                       const ShapeVector &grad_shape, const ShapeVector &indices_shape,
                                                       const std::string &prim_name) {
  std::vector<ShapeVector> check_shapes = {var_shape, accum_shape, grad_shape, indices_shape};
  auto is_dynamic = std::any_of(check_shapes.begin(), check_shapes.end(), IsDynamic);
  auto is_dynamic_rank = std::any_of(check_shapes.begin(), check_shapes.end(), IsDynamicRank);
  // Var dimension must be equal or greater than 1.
  (void)CheckAndConvertUtils::CheckInteger("var dimension", var_shape.size(), kGreaterEqual, 1, prim_name);
  // Indices must be rank 1.
  (void)CheckAndConvertUtils::CheckInteger("indices dimension", indices_shape.size(), kEqual, 1, prim_name);

  if (!is_dynamic_rank) {
    if (var_shape.size() != accum_shape.size()) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name
                               << "', rank(accum) should be same as rank(var), but got rank(grad): "
                               << accum_shape.size() << ", rank(var): " << var_shape.size() << ".";
    }
    if (var_shape.size() != grad_shape.size()) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name
                               << "', rank(grad) should be same as rank(var), but got rank(grad): " << grad_shape.size()
                               << ", rank(var): " << var_shape.size() << ".";
    }
  }

  if (!is_dynamic) {
    if (indices_shape[0] != grad_shape[0]) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name
                               << "', grad.shape[0] must be equal to indices.shape[0], but got grad.shape[0]: "
                               << grad_shape[0] << ", indices.shape[0]: " << indices_shape[0] << ".";
    }
    const size_t kZeroNum = 0;
    for (size_t i = 0; i < var_shape.size(); ++i) {
      if (var_shape[i] != accum_shape[i]) {
        MS_EXCEPTION(ValueError) << "For '" << prim_name << "'. the shape of var and accum must equal in dimension "
                                 << i << ".";
      }
      if (i == kZeroNum) {
        continue;
      }
      if (var_shape[i] != grad_shape[i]) {
        MS_EXCEPTION(ValueError) << "For '" << prim_name << "'. the shape of var and grad must equal in dimension " << i
                                 << ".";
      }
    }
  }
}

abstract::TupleShapePtr SparseApplyProximalAdagradInferShape(const PrimitivePtr &primitive,
                                                             const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto var_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  auto accum_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape())[kShape];
  auto lr_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[2]->BuildShape())[kShape];
  auto l1_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[3]->BuildShape())[kShape];
  auto l2_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[4]->BuildShape())[kShape];
  auto grad_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[5]->BuildShape())[kShape];
  auto indices_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[6]->BuildShape())[kShape];

  SparseApplyProximalAdagradCheckTensorShapeAndSize(var_shape, accum_shape, grad_shape, indices_shape, prim_name);

  if (!(IsDynamic(lr_shape) || IsDynamic(l1_shape) || IsDynamic(l2_shape))) {
    const size_t scalar_shape = 0;
    (void)CheckAndConvertUtils::CheckInteger("lr_shape size", lr_shape.size(), kEqual, scalar_shape, prim_name);
    (void)CheckAndConvertUtils::CheckInteger("l1_shape size", l1_shape.size(), kEqual, scalar_shape, prim_name);
    (void)CheckAndConvertUtils::CheckInteger("l2_shape size", l2_shape.size(), kEqual, scalar_shape, prim_name);
  }

  abstract::ShapePtr var_shape_ptr = std::make_shared<abstract::Shape>(var_shape);
  abstract::ShapePtr accum_shape_ptr = std::make_shared<abstract::Shape>(accum_shape);
  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{var_shape_ptr, accum_shape_ptr});
}

TuplePtr SparseApplyProximalAdagradInferType(const PrimitivePtr &primitive,
                                             const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto var_type = input_args[0]->BuildType();
  auto accum_type = input_args[1]->BuildType();
  auto lr_type = input_args[2]->BuildType();
  auto l1_type = input_args[3]->BuildType();
  auto l2_type = input_args[4]->BuildType();
  auto grad_type = input_args[5]->BuildType();
  auto indices_type = input_args[6]->BuildType();

  std::set<TypePtr> tensor_valid_types = {kFloat16, kFloat32};
  std::map<std::string, TypePtr> args;
  (void)args.insert({"var", var_type});
  (void)args.insert({"accum", accum_type});
  (void)args.insert({"grad", grad_type});
  (void)CheckAndConvertUtils::CheckScalarOrTensorTypesSame(args, tensor_valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckScalarOrTensorTypesSame({{"lr", lr_type}}, tensor_valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckScalarOrTensorTypesSame({{"l1", l1_type}}, tensor_valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckScalarOrTensorTypesSame({{"l2", l2_type}}, tensor_valid_types, prim_name);

  const std::set<TypePtr> valid_types = {kInt32, kInt64};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("indices", indices_type, valid_types, prim_name);

  return std::make_shared<Tuple>(std::vector<TypePtr>{var_type, accum_type});
}
}  // namespace

void SparseApplyProximalAdagrad::Init(const bool use_locking) { this->set_use_locking(use_locking); }

void SparseApplyProximalAdagrad::set_use_locking(const bool use_locking) {
  (void)this->AddAttr(kUseLocking, api::MakeValue(use_locking));
}

bool SparseApplyProximalAdagrad::get_use_locking() const {
  auto value_ptr = GetAttr(kUseLocking);
  return GetValue<bool>(value_ptr);
}

AbstractBasePtr SparseApplyProximalAdagradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int Inputs_num = 7;
  (void)CheckAndConvertUtils::CheckInteger("infer", SizeToLong(CheckAndConvertUtils::GetRemoveMonadAbsNum(input_args)),
                                           kEqual, Inputs_num, primitive->name());
  auto infer_type = SparseApplyProximalAdagradInferType(primitive, input_args);
  auto infer_shape = SparseApplyProximalAdagradInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

MIND_API_OPERATOR_IMPL(SparseApplyProximalAdagrad, BaseOperator);

// AG means auto generated
class MIND_API AGSparseApplyProximalAdagradInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return SparseApplyProximalAdagradInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return SparseApplyProximalAdagradInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return SparseApplyProximalAdagradInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(SparseApplyProximalAdagrad, prim::kPrimSparseApplyProximalAdagrad,
                                 AGSparseApplyProximalAdagradInfer, false);
}  // namespace ops
}  // namespace mindspore
