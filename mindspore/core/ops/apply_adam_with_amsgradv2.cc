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

#include "ops/apply_adam_with_amsgradv2.h"

#include <map>
#include <set>
#include <string>
#include <utility>

#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/container.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/nn_optimizer_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace ops {
namespace {
abstract::TupleShapePtr ApplyAdamWithAmsgradV2InferShape(const PrimitivePtr &primitive,
                                                         const std::vector<AbstractBasePtr> &input_args) {
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto prim_name = primitive->name();
  auto var_shape = input_args[0]->BuildShape();
  auto m_shape = input_args[1]->BuildShape();
  auto v_shape = input_args[2]->BuildShape();
  auto vhat_shape = input_args[3]->BuildShape();
  auto beta1_power_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[4]->BuildShape())[kShape];
  auto beta2_power_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[5]->BuildShape())[kShape];
  auto lr_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[6]->BuildShape())[kShape];
  auto grad_shape = input_args[10]->BuildShape();

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

  // shape of var, m, v, vhat, grad must be the same
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

TuplePtr ApplyAdamWithAmsgradV2InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = prim->name();
  auto var_type = input_args[0]->BuildType();
  auto m_type = input_args[1]->BuildType();
  auto v_type = input_args[2]->BuildType();
  auto vhat_type = input_args[3]->BuildType();
  auto beta1_power_type = input_args[4]->BuildType();
  auto beta2_power_type = input_args[5]->BuildType();
  auto lr_type = input_args[6]->BuildType();
  auto beta1_type = input_args[7]->BuildType();
  auto beta2_type = input_args[8]->BuildType();
  auto epsilon_type = input_args[9]->BuildType();
  auto grad_type = input_args[10]->BuildType();
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kFloat64};
  // var, m, v, vhat, grad valid and must has the same type
  std::map<std::string, TypePtr> args;
  (void)args.insert(std::make_pair("var_type", var_type));
  (void)args.insert(std::make_pair("m_type", m_type));
  (void)args.insert(std::make_pair("v_type", v_type));
  (void)args.insert(std::make_pair("vhat_type", vhat_type));
  (void)args.insert(std::make_pair("grad_type", grad_type));
  (void)CheckAndConvertUtils::CheckTensorTypeSame(args, valid_types, prim_name);
  // beta1_power, beta2_power, lr type valid
  (void)CheckAndConvertUtils::CheckTypeValid("beta1_power_type", beta1_power_type, valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckTypeValid("beta2_power_type", beta2_power_type, valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckTypeValid("lr_type", lr_type, valid_types, prim_name);
  // beta1, beta2, epsilon must have the same type as beta1_power
  std::map<std::string, TypePtr> args2;
  (void)args2.insert(std::make_pair("beta1_power", beta1_power_type));
  (void)args2.insert(std::make_pair("beta1", beta1_type));
  (void)args2.insert(std::make_pair("beta2", beta2_type));
  (void)args2.insert(std::make_pair("epsilon", epsilon_type));
  (void)CheckAndConvertUtils::CheckScalarOrTensorTypesSame(args2, valid_types, prim_name, true);
  return std::make_shared<Tuple>(std::vector<TypePtr>{var_type, m_type, v_type, vhat_type});
}
}  // namespace

void ApplyAdamWithAmsgradV2::set_use_locking(const bool use_locking) {
  (void)this->AddAttr(kUseLocking, api::MakeValue(use_locking));
}

bool ApplyAdamWithAmsgradV2::get_use_locking() const {
  auto value_ptr = this->GetAttr(kUseLocking);
  return GetValue<bool>(value_ptr);
}

MIND_API_OPERATOR_IMPL(ApplyAdamWithAmsgradV2, BaseOperator);
AbstractBasePtr ApplyAdamWithAmsgradV2Infer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                            const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 11;
  CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, input_num, primitive->name());
  auto infer_type = ApplyAdamWithAmsgradV2InferType(primitive, input_args);
  auto infer_shape = ApplyAdamWithAmsgradV2InferShape(primitive, input_args);
  auto shape_tuple = infer_shape->cast_ptr<abstract::TupleShape>();
  auto type_tuple = infer_type->cast_ptr<Tuple>();
  AbstractBasePtrList ptr_list;
  for (size_t it = 0; it < shape_tuple->size(); ++it) {
    auto base_shape = (*shape_tuple)[it];
    auto base_type = (*type_tuple)[it];
    auto tensor_type = base_type->cast_ptr<TensorType>();
    auto element = std::make_shared<abstract::AbstractScalar>(kValueAny, tensor_type->element());
    auto tensor_it = std::make_shared<abstract::AbstractTensor>(element, base_shape);
    ptr_list.push_back(tensor_it);
  }
  auto tuple = std::make_shared<abstract::AbstractTuple>(ptr_list);
  return tuple;
}

// AG means auto generated
class MIND_API AGApplyAdamWithAmsgradV2Infer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return ApplyAdamWithAmsgradV2InferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return ApplyAdamWithAmsgradV2InferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return ApplyAdamWithAmsgradV2Infer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(ApplyAdamWithAmsgradV2, prim::kPrimApplyAdamWithAmsgradV2,
                                 AGApplyAdamWithAmsgradV2Infer, false);
}  // namespace ops
}  // namespace mindspore
