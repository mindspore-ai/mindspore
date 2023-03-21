/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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

#include "ops/apply_adam_with_amsgrad.h"

#include <string>
#include <set>
#include <map>
#include <utility>

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
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::TupleShapePtr ApplyAdamWithAmsgradInferShape(const PrimitivePtr &primitive,
                                                       const std::vector<AbstractBasePtr> &input_args) {
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto prim_name = primitive->name();
  auto var_shape = input_args[0]->BuildShape();
  auto m_shape = input_args[1]->BuildShape();
  auto v_shape = input_args[2]->BuildShape();
  auto vhat_shape = input_args[3]->BuildShape();
  auto grad_shape = input_args[7]->BuildShape();
  auto beta1_power_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[4]->BuildShape())[kShape];
  auto beta2_power_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[5]->BuildShape())[kShape];
  auto lr_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[6]->BuildShape())[kShape];

  int64_t batch_rank = 0;
  if (primitive->HasAttr(kBatchRank)) {
    auto value_ptr = primitive->GetAttr(kBatchRank);
    batch_rank = GetValue<int64_t>(value_ptr);
  }

  (void)CheckAndConvertUtils::CheckInteger("beta1_power_shape size", beta1_power_shape.size(), kGreaterEqual,
                                           batch_rank, prim_name);
  (void)CheckAndConvertUtils::CheckInteger("beta2_power_shape size", beta2_power_shape.size(), kGreaterEqual,
                                           batch_rank, prim_name);
  (void)CheckAndConvertUtils::CheckInteger("lr_shape size", lr_shape.size(), kGreaterEqual, batch_rank, prim_name);

  if (var_shape->IsDynamic() || m_shape->IsDynamic() || v_shape->IsDynamic() || vhat_shape->IsDynamic() ||
      grad_shape->IsDynamic()) {
    return std::make_shared<abstract::TupleShape>(
      std::vector<abstract::BaseShapePtr>{var_shape, m_shape, v_shape, vhat_shape});
  }

  // shape of var, m, v, vhat must be the same
  std::map<std::string, abstract::BaseShapePtr> same_shape_args_map;
  (void)same_shape_args_map.insert(std::make_pair("m", m_shape));
  (void)same_shape_args_map.insert(std::make_pair("v", v_shape));
  (void)same_shape_args_map.insert(std::make_pair("vhat", vhat_shape));
  (void)same_shape_args_map.insert(std::make_pair("grad", grad_shape));
  for (auto &elem : same_shape_args_map) {
    if (*elem.second != *var_shape) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name << "', evaluator arg '" << elem.first
                               << "' and 'var' must have the same shape. But got '" << elem.first
                               << "' shape: " << elem.second->ToString() << ", 'var' shape: " << var_shape->ToString()
                               << ".";
    }
  }
  return std::make_shared<abstract::TupleShape>(
    std::vector<abstract::BaseShapePtr>{var_shape, m_shape, v_shape, vhat_shape});
}

TuplePtr ApplyAdamWithAmsgradInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = prim->name();
  auto var_type = input_args[0]->BuildType();
  auto m_type = input_args[1]->BuildType();
  auto v_type = input_args[2]->BuildType();
  auto vhat_type = input_args[3]->BuildType();
  auto beta1_power_type = input_args[4]->BuildType();
  auto beta2_power_type = input_args[5]->BuildType();
  auto lr_type = input_args[6]->BuildType();
  auto grad_type = input_args[7]->BuildType();
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32};
  // var, m, v, vhat, grad valid and must has the same type
  std::map<std::string, TypePtr> args;
  (void)args.insert(std::make_pair("var_type", var_type));
  (void)args.insert(std::make_pair("m_type", m_type));
  (void)args.insert(std::make_pair("v_type", v_type));
  (void)args.insert(std::make_pair("vhat_type", vhat_type));
  (void)args.insert(std::make_pair("grad_type", grad_type));
  (void)CheckAndConvertUtils::CheckTensorTypeSame(args, valid_types, prim_name);
  // beta1_power, beta2_power, lr type valid
  (void)CheckAndConvertUtils::CheckTensorTypeValid("beta1_power_type", beta1_power_type, valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("beta2_power_type", beta2_power_type, valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("lr_type", lr_type, valid_types, prim_name);
  return std::make_shared<Tuple>(std::vector<TypePtr>{var_type, m_type, v_type, vhat_type});
}
}  // namespace

void ApplyAdamWithAmsgrad::set_beta1(const float beta1) { (void)this->AddAttr(kBeta1, api::MakeValue(beta1)); }

void ApplyAdamWithAmsgrad::set_beta2(const float beta2) { (void)this->AddAttr(kBeta2, api::MakeValue(beta2)); }

void ApplyAdamWithAmsgrad::set_epsilon(const float epsilon) { (void)this->AddAttr(kEpsilon, api::MakeValue(epsilon)); }

void ApplyAdamWithAmsgrad::set_use_locking(const bool use_locking) {
  (void)this->AddAttr(kUseLocking, api::MakeValue(use_locking));
}

float ApplyAdamWithAmsgrad::get_beta1() const {
  auto value_ptr = this->GetAttr(kBeta1);
  return GetValue<float>(value_ptr);
}

float ApplyAdamWithAmsgrad::get_beta2() const {
  auto value_ptr = this->GetAttr(kBeta2);
  return GetValue<float>(value_ptr);
}

float ApplyAdamWithAmsgrad::get_epsilon() const {
  auto value_ptr = this->GetAttr(kEpsilon);
  return GetValue<float>(value_ptr);
}

bool ApplyAdamWithAmsgrad::get_use_locking() const {
  auto value_ptr = this->GetAttr(kUseLocking);
  return GetValue<bool>(value_ptr);
}

MIND_API_OPERATOR_IMPL(ApplyAdamWithAmsgrad, BaseOperator);
AbstractBasePtr ApplyAdamWithAmsgradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 8;
  CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, input_num, primitive->name());
  auto infer_type = ApplyAdamWithAmsgradInferType(primitive, input_args);
  auto infer_shape = ApplyAdamWithAmsgradInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGApplyAdamWithAmsgradInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return ApplyAdamWithAmsgradInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return ApplyAdamWithAmsgradInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return ApplyAdamWithAmsgradInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(ApplyAdamWithAmsgrad, prim::kPrimApplyAdamWithAmsgrad, AGApplyAdamWithAmsgradInfer,
                                 false);
}  // namespace ops
}  // namespace mindspore
