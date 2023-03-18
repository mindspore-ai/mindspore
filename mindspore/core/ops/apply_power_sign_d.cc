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

#include "ops/apply_power_sign_d.h"

#include <vector>
#include <memory>
#include <set>
#include <map>
#include <string>
#include <utility>

#include "utils/check_convert_utils.h"
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
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
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
  auto sign_decay_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex4]->GetShapeTrack())[kShape];
  auto beta_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex5]->GetShapeTrack())[kShape];
  auto grad_shape = input_args[kInputIndex6]->BuildShape();
  int64_t batch_rank = 0;
  if (primitive->HasAttr(kBatchRank)) {
    auto value_ptr = primitive->GetAttr(kBatchRank);
    batch_rank = GetValue<int64_t>(value_ptr);
  }
  std::vector<std::pair<std::string, std::vector<int64_t>>> check_inputs_shape{{"lr_shape", lr_shape},
                                                                               {"logbase_shape", logbase_shape},
                                                                               {"sign_decay_shape", sign_decay_shape},
                                                                               {"beta_shape", beta_shape}};
  if (batch_rank != 0) {
    for (auto item : check_inputs_shape) {
      (void)CheckAndConvertUtils::CheckInteger(item.first + " rank", SizeToLong(item.second.size()), kEqual, batch_rank,
                                               prim_name);
    }
  } else {
    for (auto item : check_inputs_shape) {
      (void)CheckAndConvertUtils::CheckInteger(item.first + " rank", SizeToLong(item.second.size()), kLessEqual, 1,
                                               prim_name);
      if (item.second.size() == 1 && item.second[0] > 0) {
        (void)CheckAndConvertUtils::CheckInteger(item.first + "[0]", item.second[0], kEqual, 1, prim_name);
      }
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
AbstractBasePtr ApplyPowerSignInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto infer_type = ApplyPowerSignDInferType(primitive, input_args);
  auto infer_shape = ApplyPowerSignDInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGApplyPowerSignInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return ApplyPowerSignDInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return ApplyPowerSignDInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return ApplyPowerSignInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(ApplyPowerSign, prim::kPrimApplyPowerSign, AGApplyPowerSignInfer, false);
}  // namespace ops
}  // namespace mindspore
