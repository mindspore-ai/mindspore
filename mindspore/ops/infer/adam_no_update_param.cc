/**
 * Copyright 2021-2025 Huawei Technologies Co., Ltd
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
#include "infer/adam_no_update_param.h"

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "abstract/abstract_value.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "mindapi/helper.h"
#include "primitive/nn_optimizer_ops.h"
#include "primitive/op_name.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(AdamNoUpdateParam, BaseOperator);
class AdamNoUpdateParamInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto prim_name = primitive->name();
    const int64_t input_num = 9;
    const int64_t input_num_with_monad = 10;
    if (input_args.size() != input_num && input_args.size() != input_num_with_monad) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name << "', the input num should be equal to 9, but got "
                               << input_args.size();
    }
    for (const auto &item : input_args) {
      MS_EXCEPTION_IF_NULL(item);
    }
    auto grad_shape_ptr = input_args[kInputIndex8]->GetShape();
    auto grad_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(grad_shape_ptr)[kShape];
    auto m_shape_ptr = input_args[kInputIndex0]->GetShape();
    auto m_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(m_shape_ptr)[kShape];
    auto v_shape_ptr = input_args[kInputIndex1]->GetShape();
    auto v_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(v_shape_ptr)[kShape];
    bool is_dynamic = IsDynamic(grad_shape) || IsDynamic(m_shape) || IsDynamic(v_shape);
    if (!is_dynamic) {
      CheckAndConvertUtils::Check("grad_shape", grad_shape, kEqual, m_shape, prim_name);
      CheckAndConvertUtils::Check("grad_shape", grad_shape, kEqual, v_shape, prim_name);
    }
    return grad_shape_ptr;
  }

  TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(prim);
    auto prim_name = prim->name();
    const int64_t input_num = 9;
    const int64_t input_num_with_monad = 10;
    if (input_args.size() != input_num && input_args.size() != input_num_with_monad) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name << "', the input num should be equal to 9 , but got "
                               << input_args.size();
    }
    for (const auto &item : input_args) {
      MS_EXCEPTION_IF_NULL(item);
    }
    auto m_type = input_args[kInputIndex0]->GetType();
    auto v_type = input_args[kInputIndex1]->GetType();
    auto beta1_power_type = input_args[kInputIndex2]->GetType();
    auto beta2_power_type = input_args[kInputIndex3]->GetType();
    auto lr_type = input_args[kInputIndex4]->GetType();
    auto beta1_type = input_args[kInputIndex5]->GetType();
    auto beta2_type = input_args[kInputIndex6]->GetType();
    auto epsilon_type = input_args[kInputIndex7]->GetType();
    auto grad_type = input_args[kInputIndex8]->GetType();
    std::set<TypePtr> float32_set = {kFloat32};
    std::map<std::string, TypePtr> type_dict;
    type_dict.emplace("m", m_type);
    type_dict.emplace("v", v_type);
    type_dict.emplace("grad", grad_type);
    CheckAndConvertUtils::CheckTensorTypeSame(type_dict, float32_set, prim_name);
    std::map<std::string, TypePtr> type_dict1;
    type_dict1.emplace("beta1_power", beta1_power_type);
    type_dict1.emplace("beta2_power", beta2_power_type);
    type_dict1.emplace("lr", lr_type);
    type_dict1.emplace("beta1", beta1_type);
    type_dict1.emplace("beta2", beta2_type);
    type_dict1.emplace("epsilon", epsilon_type);
    CheckAndConvertUtils::CheckScalarOrTensorTypesSame(type_dict1, float32_set, prim_name, true);
    return grad_type;
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(AdamNoUpdateParam, prim::kPrimAdamNoUpdateParam, AdamNoUpdateParamInfer, false);
}  // namespace ops
}  // namespace mindspore
