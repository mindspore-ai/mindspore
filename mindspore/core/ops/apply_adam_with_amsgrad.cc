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

#include "ops/apply_adam_with_amsgrad.h"

#include <string>
#include <set>
#include <map>

#include "ops/op_utils.h"
#include "abstract/primitive_infer_map.h"
#include "utils/tensor_construct_utils.h"
#include "utils/check_convert_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::TupleShapePtr ApplyAdamWithAmsgradInferShape(const PrimitivePtr &primitive,
                                                       const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto var_shape = input_args[0]->BuildShape();
  auto m_shape = input_args[1]->BuildShape();
  auto v_shape = input_args[2]->BuildShape();
  auto vhat_shape = input_args[3]->BuildShape();
  auto beta1_power_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[4]->GetShapeTrack())[kShape];
  auto beta2_power_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[5]->GetShapeTrack())[kShape];
  auto lr_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[6]->GetShapeTrack())[kShape];
  auto grad_shape = input_args[7]->BuildShape();
  // beta1_power, beta2_power, lr must be scalar
  (void)CheckAndConvertUtils::CheckInteger("beta1_power_shape size", beta1_power_shape.size(), kEqual, 0, prim_name);
  (void)CheckAndConvertUtils::CheckInteger("beta2_power_shape size", beta2_power_shape.size(), kEqual, 0, prim_name);
  (void)CheckAndConvertUtils::CheckInteger("lr_shape size", lr_shape.size(), kEqual, 0, prim_name);
  // shape of var, m, v, vhat must be the same
  std::map<std::string, abstract::BaseShapePtr> same_shape_args_map;
  same_shape_args_map.insert({"m", m_shape});
  same_shape_args_map.insert({"v", v_shape});
  same_shape_args_map.insert({"vhat", vhat_shape});
  same_shape_args_map.insert({"grad", grad_shape});
  for (auto &elem : same_shape_args_map) {
    if (*elem.second != *var_shape) {
      MS_EXCEPTION(ValueError) << prim_name << " evaluator arg " << elem.first << " shape " << elem.second->ToString()
                               << " are not consistent with var shape " << var_shape->ToString();
    }
  }
  return std::make_shared<abstract::TupleShape>(
    std::vector<abstract::BaseShapePtr>{var_shape, m_shape, v_shape, vhat_shape});
}

TuplePtr ApplyAdamWithAmsgradInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();
  // get all input_args' shape
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
  args.insert({"var_type", var_type});
  args.insert({"m_type", m_type});
  args.insert({"v_type", v_type});
  args.insert({"vhat_type", vhat_type});
  args.insert({"grad_type", grad_type});
  CheckAndConvertUtils::CheckTensorTypeSame(args, valid_types, prim_name);
  // beta1_power, beta2_power, lr type valid
  CheckAndConvertUtils::CheckTensorTypeValid("beta1_power_type", beta1_power_type, valid_types, prim_name);
  CheckAndConvertUtils::CheckTensorTypeValid("beta2_power_type", beta2_power_type, valid_types, prim_name);
  CheckAndConvertUtils::CheckTensorTypeValid("lr_type", lr_type, valid_types, prim_name);
  return std::make_shared<Tuple>(std::vector<TypePtr>{var_type, m_type, v_type, vhat_type});
}
}  // namespace

MIND_API_BASE_IMPL(ApplyAdamWithAmsgrad, PrimitiveC, BaseOperator);
AbstractBasePtr ApplyAdamWithAmsgradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 8;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infer_type = ApplyAdamWithAmsgradInferType(primitive, input_args);
  auto infer_shape = ApplyAdamWithAmsgradInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(ApplyAdamWithAmsgrad, prim::kPrimApplyAdamWithAmsgrad, ApplyAdamWithAmsgradInfer, nullptr,
                             true);
}  // namespace ops
}  // namespace mindspore
