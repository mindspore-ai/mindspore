/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include "ops/ops_func_impl/adamw.h"

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/container.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/math_ops.h"
#include "mindspore/core/ops/nn_optimizer_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "ops/auto_generate/gen_ops_name.h"

namespace mindspore {
namespace ops {
BaseShapePtr AdamWFuncImpl::InferShape(const PrimitivePtr &primitive,
                                       const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t adamw_size = 13;
  auto input_real_num = SizeToLong(CheckAndConvertUtils::GetRemoveMonadAbsNum(input_args));
  MS_CHECK_VALUE(input_real_num == adamw_size, CheckAndConvertUtils::FormatCheckIntegerMsg(
                                                 "input number", input_real_num, kEqual, adamw_size, primitive));
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto var_shape_ptr = input_args[kInputIndex0]->GetShape();
  auto var_shape = var_shape_ptr->GetShapeVector();
  auto m_shape_ptr = input_args[kInputIndex1]->GetShape();
  auto m_shape = m_shape_ptr->GetShapeVector();
  auto v_shape_ptr = input_args[kInputIndex2]->GetShape();
  auto v_shape = v_shape_ptr->GetShapeVector();
  auto max_v_shape_ptr = input_args[kInputIndex3]->GetShape();
  auto max_v_shape = max_v_shape_ptr->GetShapeVector();
  auto step_shape_ptr = input_args[kInputIndex5]->GetShape();
  auto step_shape = step_shape_ptr->GetShapeVector();
  auto grad_shape_ptr = input_args[kInputIndex4]->GetShape();
  auto grad_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(grad_shape_ptr)[kShape];
  bool is_dynamic = IsDynamic(var_shape) || IsDynamic(m_shape) || IsDynamic(v_shape) || IsDynamic(max_v_shape) ||
                    IsDynamic(step_shape) || IsDynamic(grad_shape);
  if (!is_dynamic) {
    MS_CHECK_VALUE(var_shape == m_shape,
                   CheckAndConvertUtils::FormatCheckMsg("var_shape", var_shape, kEqual, m_shape, primitive));
    MS_CHECK_VALUE(var_shape == v_shape,
                   CheckAndConvertUtils::FormatCheckMsg("var_shape", var_shape, kEqual, v_shape, primitive));
    MS_CHECK_VALUE(var_shape == grad_shape,
                   CheckAndConvertUtils::FormatCheckMsg("var_shape", var_shape, kEqual, grad_shape, primitive));
    MS_CHECK_VALUE(var_shape == max_v_shape,
                   CheckAndConvertUtils::FormatCheckMsg("var_shape", var_shape, kEqual, max_v_shape, primitive));
  }
  return std::make_shared<abstract::TupleShape>(
    std::vector<abstract::BaseShapePtr>{var_shape_ptr, m_shape_ptr, v_shape_ptr});
}

TypePtr AdamWFuncImpl::InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();
  const int64_t adamw_size = 13;
  auto input_real_num = SizeToLong(CheckAndConvertUtils::GetRemoveMonadAbsNum(input_args));
  MS_CHECK_VALUE(input_real_num == adamw_size,
                 CheckAndConvertUtils::FormatCheckIntegerMsg("input number", input_real_num, kEqual, adamw_size, prim));
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto var_type = input_args[kInputIndex0]->GetType();
  auto m_type = input_args[kInputIndex1]->GetType();
  auto v_type = input_args[kInputIndex2]->GetType();

  auto lr_type = input_args[kInputIndex6]->GetType();
  auto beta1_type = input_args[kInputIndex7]->GetType();
  auto beta2_type = input_args[kInputIndex8]->GetType();
  auto decay_type = input_args[kInputIndex9]->GetType();
  auto epsilon_type = input_args[kInputIndex10]->GetType();
  auto grad_type = input_args[kInputIndex4]->GetType();

  const std::set<TypePtr> number_type = {kInt8,   kInt16,   kInt32,   kInt64,   kUInt8,     kUInt16,    kUInt32,
                                         kUInt64, kFloat16, kFloat32, kFloat64, kComplex64, kComplex64, kBFloat16};
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
  type_dict1.emplace("decay", decay_type);
  type_dict1.emplace("eps", epsilon_type);
  CheckAndConvertUtils::CheckScalarOrTensorTypesSame(type_dict1, float32_set, prim_name, true);
  return std::make_shared<Tuple>(std::vector<TypePtr>{var_type, m_type, v_type});
}
}  // namespace ops
}  // namespace mindspore
