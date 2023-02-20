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
#include "ops/adam_weight_decay.h"

#include <string>
#include <vector>
#include <set>
#include <map>
#include <memory>

#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
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
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(AdamWeightDecay, BaseOperator);
class AdamWeightDecayInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto prim_name = primitive->name();
    const int64_t input_num = 9;
    (void)CheckAndConvertUtils::CheckInteger(
      "input number", SizeToLong(CheckAndConvertUtils::GetRemoveMonadAbsNum(input_args)), kEqual, input_num, prim_name);
    for (const auto &item : input_args) {
      MS_EXCEPTION_IF_NULL(item);
    }
    auto var_shape_ptr = input_args[kInputIndex0]->BuildShape();
    auto var_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(var_shape_ptr)[kShape];
    auto m_shape_ptr = input_args[kInputIndex0]->BuildShape();
    auto m_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(m_shape_ptr)[kShape];
    auto v_shape_ptr = input_args[kInputIndex1]->BuildShape();
    auto v_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(v_shape_ptr)[kShape];
    auto grad_shape_ptr = input_args[kInputIndex8]->BuildShape();
    auto grad_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(grad_shape_ptr)[kShape];
    bool is_dynamic = IsDynamic(var_shape) || IsDynamic(m_shape) || IsDynamic(v_shape) || IsDynamic(grad_shape);
    if (!is_dynamic) {
      CheckAndConvertUtils::Check("var_shape", var_shape, kEqual, m_shape, prim_name);
      CheckAndConvertUtils::Check("var_shape", var_shape, kEqual, v_shape, prim_name);
      CheckAndConvertUtils::Check("var_shape", var_shape, kEqual, grad_shape, prim_name);
    }
    return std::make_shared<abstract::TupleShape>(
      std::vector<abstract::BaseShapePtr>{var_shape_ptr, m_shape_ptr, v_shape_ptr});
  }

  TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(prim);
    auto prim_name = prim->name();
    const int64_t input_num = 9;
    (void)CheckAndConvertUtils::CheckInteger(
      "input number", SizeToLong(CheckAndConvertUtils::GetRemoveMonadAbsNum(input_args)), kEqual, input_num, prim_name);
    for (const auto &item : input_args) {
      MS_EXCEPTION_IF_NULL(item);
    }
    auto var_type = input_args[kInputIndex0]->BuildType();
    auto m_type = input_args[kInputIndex1]->BuildType();
    auto v_type = input_args[kInputIndex2]->BuildType();
    auto lr_type = input_args[kInputIndex3]->BuildType();
    auto beta1_type = input_args[kInputIndex4]->BuildType();
    auto beta2_type = input_args[kInputIndex5]->BuildType();
    auto epsilon_type = input_args[kInputIndex6]->BuildType();
    auto decay_type = input_args[kInputIndex7]->BuildType();
    auto grad_type = input_args[kInputIndex8]->BuildType();

    const std::set<TypePtr> number_type = {kInt8,   kInt16,   kInt32,   kInt64,   kUInt8,     kUInt16,   kUInt32,
                                           kUInt64, kFloat16, kFloat32, kFloat64, kComplex64, kComplex64};
    std::map<std::string, TypePtr> type_dict_var_grad;
    type_dict_var_grad.emplace("var", var_type);
    type_dict_var_grad.emplace("grad", grad_type);
    CheckAndConvertUtils::CheckTensorTypeSame(type_dict_var_grad, number_type, prim_name);

    std::map<std::string, TypePtr> type_dict_m_v;
    type_dict_m_v.emplace("m", m_type);
    type_dict_m_v.emplace("v", v_type);
    CheckAndConvertUtils::CheckTensorTypeSame(type_dict_m_v, number_type, prim_name);

    std::set<TypePtr> float32_set = {kFloat32};
    std::map<std::string, TypePtr> type_dict1;
    type_dict1.emplace("lr", lr_type);
    type_dict1.emplace("beta1", beta1_type);
    type_dict1.emplace("beta2", beta2_type);
    type_dict1.emplace("epsilon", epsilon_type);
    type_dict1.emplace("decay", decay_type);
    CheckAndConvertUtils::CheckScalarOrTensorTypesSame(type_dict1, float32_set, prim_name, true);
    return std::make_shared<Tuple>(std::vector<TypePtr>{var_type, m_type, v_type});
  }
};

void AdamWeightDecay::Init(const bool use_locking) { this->set_use_locking(use_locking); }

void AdamWeightDecay::set_use_locking(const bool use_locking) {
  (void)this->AddAttr(kUseLocking, api::MakeValue(use_locking));
}

bool AdamWeightDecay::get_use_locking() const {
  auto value_ptr = GetAttr(kUseLocking);
  return GetValue<bool>(value_ptr);
}

REGISTER_PRIMITIVE_OP_INFER_IMPL(AdamWeightDecay, prim::kPrimAdamWeightDecay, AdamWeightDecayInfer, false);
}  // namespace ops
}  // namespace mindspore
