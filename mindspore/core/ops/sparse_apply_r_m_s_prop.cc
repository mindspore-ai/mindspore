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

#include "ops/sparse_apply_r_m_s_prop.h"

#include <set>
#include <utility>
#include <map>
#include <memory>
#include <type_traits>

#include "abstract/ops/primitive_infer_map.h"
#include "utils/check_convert_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/container.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::TupleShapePtr SparseApplyRMSPropInferShape(const PrimitivePtr &primitive,
                                                     const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  MS_EXCEPTION_IF_NULL(primitive);
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, 6, primitive->name());
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto var_shape_ptr = input_args[0]->BuildShape();
  auto ms_shape_ptr = input_args[1]->BuildShape();
  auto mom_shape_ptr = input_args[2]->BuildShape();

  auto var_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(var_shape_ptr)[kShape];
  auto ms_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(ms_shape_ptr)[kShape];
  auto mom_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(mom_shape_ptr)[kShape];
  auto lr_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[3]->BuildShape())[kShape];
  auto lr_shape_rank = SizeToLong(lr_shape.size());
  auto grad_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[4]->BuildShape())[kShape];
  auto indices_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[5]->BuildShape())[kShape];

  // Args lr must be scalar
  const int64_t input_num = 0;
  if (!IsDynamic(lr_shape)) {
    (void)CheckAndConvertUtils::CheckInteger("size of lr_shape", lr_shape_rank, kEqual, input_num, primitive->name());
  }

  std::vector<ShapeVector> check_shapes = {ms_shape, mom_shape, grad_shape, var_shape};
  auto is_dynamic = std::any_of(check_shapes.begin(), check_shapes.end(), IsDynamic);
  if (!is_dynamic) {
    // Shape of var、ms、mom、grad must be same
    std::map<std::string, ShapeVector> same_shape_args_map;
    (void)same_shape_args_map.insert(std::make_pair("shape of ms ", ms_shape));
    (void)same_shape_args_map.insert(std::make_pair("shape of mom ", mom_shape));
    (void)same_shape_args_map.insert(std::make_pair("shape of grad ", grad_shape));
    for (auto &elem : same_shape_args_map) {
      CheckAndConvertUtils::Check(elem.first, elem.second, kEqual, var_shape, prim_name);
    }
  }

  // Indices must be rank 1
  const int64_t input_num1 = 1;
  (void)CheckAndConvertUtils::CheckInteger("indices dim", SizeToLong(indices_shape.size()), kEqual, input_num1,
                                           prim_name);

  // Dimension of var must be equal or greater than 1
  (void)CheckAndConvertUtils::CheckInteger("dimension of var", SizeToLong(var_shape.size()), kGreaterEqual, input_num1,
                                           prim_name);

  // Indices shape must be equal to the first dimension of var
  if (!(IsDynamic(indices_shape) || IsDynamic(var_shape))) {
    CheckAndConvertUtils::Check("indices shape", indices_shape[0], kEqual, var_shape[0], prim_name);
  }

  return std::make_shared<abstract::TupleShape>(
    std::vector<abstract::BaseShapePtr>{var_shape_ptr, ms_shape_ptr, mom_shape_ptr});
}

TuplePtr SparseApplyRMSPropInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();

  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, 6, prim_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }

  auto var_type = input_args[0]->BuildType();
  auto ms_type = input_args[1]->BuildType();
  auto mom_type = input_args[2]->BuildType();
  auto lr_type = input_args[3]->BuildType();
  auto grad_type = input_args[4]->BuildType();
  auto indices_type = input_args[5]->BuildType();
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32};

  // Args ms、mom、grad must have the same type as var
  std::map<std::string, TypePtr> args;
  (void)args.insert(std::make_pair("var", var_type));
  (void)args.insert(std::make_pair("ms", ms_type));
  (void)args.insert(std::make_pair("mom", mom_type));
  (void)args.insert(std::make_pair("grad", grad_type));
  (void)CheckAndConvertUtils::CheckTensorTypeSame(args, valid_types, prim_name);

  // Args lr must be a scalar type
  std::map<std::string, TypePtr> args2;
  (void)args2.insert(std::make_pair("lr", lr_type));
  (void)CheckAndConvertUtils::CheckScalarOrTensorTypesSame(args2, valid_types, prim_name);

  // Check indices type
  std::map<std::string, TypePtr> args3;
  (void)args3.insert(std::make_pair("indices", indices_type));
  const std::set<TypePtr> valid_types1 = {kInt32, kInt64};
  (void)CheckAndConvertUtils::CheckScalarOrTensorTypesSame(args3, valid_types1, prim_name);

  return std::make_shared<Tuple>(std::vector<TypePtr>{var_type, ms_type, mom_type});
}
}  // namespace

// SparseApplyRMSProp Rho getter method
float SparseApplyRMSProp::get_rho() const {
  auto value_ptr = this->GetAttr(kRho);
  return GetValue<float>(value_ptr);
}

// SparseApplyRMSProp Rho setter method
void SparseApplyRMSProp::set_rho(const float rho) { (void)this->AddAttr(kRho, api::MakeValue(rho)); }

// SparseApplyRMSProp Momentum getter method
float SparseApplyRMSProp::get_momentum() const {
  auto value_ptr = this->GetAttr(kMomentum);
  return GetValue<float>(value_ptr);
}

// SparseApplyRMSProp Momentum setter method
void SparseApplyRMSProp::set_momentum(const float momentum) {
  (void)this->AddAttr(kMomentum, api::MakeValue(momentum));
}

// SparseApplyRMSProp Epsilon getter method
float SparseApplyRMSProp::get_epsilon() const {
  auto value_ptr = this->GetAttr(kEpsilon);
  return GetValue<float>(value_ptr);
}

// SparseApplyRMSProp Epsilon setter method
void SparseApplyRMSProp::set_epsilon(const float epsilon) { (void)this->AddAttr(kEpsilon, api::MakeValue(epsilon)); }

// SparseApplyRMSProp Use_Locking getz`ter method
bool SparseApplyRMSProp::get_use_locking() const {
  auto value_ptr = this->GetAttr(kUseLocking);
  return GetValue<bool>(value_ptr);
}

// SparseApplyRMSProp Use_Locking setter method
void SparseApplyRMSProp::set_use_locking(const bool use_locking) {
  (void)this->AddAttr(kUseLocking, api::MakeValue(use_locking));
}

MIND_API_OPERATOR_IMPL(SparseApplyRMSProp, BaseOperator);
AbstractBasePtr SparseApplyRMSPropInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                        const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  return abstract::MakeAbstract(SparseApplyRMSPropInferShape(primitive, input_args),
                                SparseApplyRMSPropInferType(primitive, input_args));
}

// AG means auto generated
class MIND_API AGSparseApplyRMSPropInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return SparseApplyRMSPropInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return SparseApplyRMSPropInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return SparseApplyRMSPropInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(SparseApplyRMSProp, prim::kPrimSparseApplyRMSProp, AGSparseApplyRMSPropInfer, false);
}  // namespace ops
}  // namespace mindspore
