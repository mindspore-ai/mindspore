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

#include "ops/apply_power_sign_d.h"

#include <vector>
#include <memory>
#include <set>
#include <map>
#include <string>

#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
const int64_t kApplyPowerSignDInputNum = 7;
abstract::TupleShapePtr ApplyPowerSignDInferShape(const PrimitivePtr &primitive,
                                                  const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kGreaterEqual,
                                           kApplyPowerSignDInputNum, prim_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto var_shape = input_args[kInputIndex0]->BuildShape();
  auto m_shape = input_args[kInputIndex1]->BuildShape();
  auto lr_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->GetShapeTrack())[kShape];
  auto logbase_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex3]->GetShapeTrack())[kShape];
  auto logbase_shape_rank = SizeToLong(logbase_shape.size());
  auto sign_decay_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex4]->GetShapeTrack())[kShape];
  auto sign_decay_shape_rank = SizeToLong(sign_decay_shape.size());
  auto beta_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex5]->GetShapeTrack())[kShape];
  auto grad_shape = input_args[kInputIndex6]->BuildShape();
  int64_t batch_rank = 0;
  if (primitive->HasAttr(kBatchRank)) {
    auto value_ptr = primitive->GetAttr(kBatchRank);
    batch_rank = GetValue<int64_t>(value_ptr);
  }

  if (batch_rank != 0) {
    (void)CheckAndConvertUtils::CheckInteger("lr_shape size", SizeToLong(lr_shape.size()), kEqual, batch_rank,
                                             prim_name);
    (void)CheckAndConvertUtils::CheckInteger("logbase_shape size", logbase_shape_rank, kEqual, batch_rank, prim_name);
    (void)CheckAndConvertUtils::CheckInteger("sign_decay_shape size", sign_decay_shape_rank, kEqual, batch_rank,
                                             prim_name);
    (void)CheckAndConvertUtils::CheckInteger("beta_shape size", SizeToLong(beta_shape.size()), kEqual, batch_rank,
                                             prim_name);
  } else {
    (void)CheckAndConvertUtils::CheckInteger("lr_shape size", SizeToLong(lr_shape.size()), kLessEqual, 1, prim_name);
    if (lr_shape.size() == 1) {
      (void)CheckAndConvertUtils::CheckInteger("lr_shape[0] size", lr_shape[0], kEqual, 1, prim_name);
    }
    (void)CheckAndConvertUtils::CheckInteger("logbase_shape rank", logbase_shape_rank, kLessEqual, 1, prim_name);
    if (logbase_shape_rank == 1) {
      (void)CheckAndConvertUtils::CheckInteger("logbase_shape[0] size", logbase_shape[0], kEqual, 1, prim_name);
    }
    (void)CheckAndConvertUtils::CheckInteger("sign_decay_shape rank", sign_decay_shape_rank, kLessEqual, 1, prim_name);
    if (sign_decay_shape_rank == 1) {
      (void)CheckAndConvertUtils::CheckInteger("sign_decay_shape[0] size", sign_decay_shape[0], kEqual, 1, prim_name);
    }
    (void)CheckAndConvertUtils::CheckInteger("beta_shape size", SizeToLong(beta_shape.size()), kLessEqual, 1,
                                             prim_name);
    if (beta_shape.size() == 1) {
      (void)CheckAndConvertUtils::CheckInteger("beta_shape[0] size", beta_shape[0], kEqual, 1, prim_name);
    }
  }

  if (grad_shape->IsDynamic() || var_shape->IsDynamic() || m_shape->IsDynamic()) {
    return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{var_shape, m_shape});
  }
  // var, m and grad must have the same shape
  std::map<std::string, abstract::BaseShapePtr> same_shape_args_map;
  (void)same_shape_args_map.insert(std::make_pair("m", m_shape));
  (void)same_shape_args_map.insert(std::make_pair("grad", grad_shape));
  for (auto &elem : same_shape_args_map) {
    if (*elem.second != *var_shape) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name << "', evaluator arg '" << elem.first
                               << "' must have the same shape as 'var'. But got '" << elem.first
                               << "' shape: " << elem.second->ToString() << ", 'var' shape: " << var_shape->ToString()
                               << ".";
    }
  }
  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{var_shape, m_shape});
}

TuplePtr ApplyPowerSignDInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kGreaterEqual,
                                           kApplyPowerSignDInputNum, prim_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto var_type = input_args[kInputIndex0]->BuildType();
  auto m_type = input_args[kInputIndex1]->BuildType();
  auto lr_type = input_args[kInputIndex2]->BuildType();
  auto logbase_type = input_args[kInputIndex3]->BuildType();
  auto sign_decay_type = input_args[kInputIndex4]->BuildType();
  auto beta_type = input_args[kInputIndex5]->BuildType();
  auto grad_type = input_args[kInputIndex6]->BuildType();
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kFloat64};
  std::map<std::string, TypePtr> args;
  (void)args.insert(std::make_pair("var", var_type));
  (void)args.insert(std::make_pair("m", m_type));
  (void)args.insert(std::make_pair("grad", grad_type));
  (void)CheckAndConvertUtils::CheckTensorTypeSame(args, valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("lr_dtype", lr_type, valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("logbase_dtype", logbase_type, valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("sign_decay_dtype", sign_decay_type, valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("beta_dtype", beta_type, valid_types, prim_name);
  return std::make_shared<Tuple>(std::vector<TypePtr>{var_type, m_type});
}
}  // namespace

MIND_API_OPERATOR_IMPL(ApplyPowerSign, BaseOperator);
AbstractBasePtr ApplyPowerSignDInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto infer_type = ApplyPowerSignDInferType(primitive, input_args);
  auto infer_shape = ApplyPowerSignDInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(ApplyPowerSign, prim::kPrimApplyPowerSignD, ApplyPowerSignDInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
