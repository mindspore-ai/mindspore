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

#include "ops/ops_func_impl/embedding_apply_ftrl.h"

#include <vector>
#include <set>
#include <map>
#include <string>
#include <memory>

#include "utils/ms_context.h"
#include "utils/check_convert_utils.h"
#include "ops/op_name.h"
#include "ops/op_utils.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr EmbeddingApplyFtrlFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                    const std::vector<AbstractBasePtr> &input_args) const {
  CheckTensorScalarRank(primitive, input_args[kInputIndex0], "var_handle");
  CheckTensorScalarRank(primitive, input_args[kInputIndex1], "lr");
  CheckTensorScalarRank(primitive, input_args[kInputIndex2], "lr_power");
  CheckTensorScalarRank(primitive, input_args[kInputIndex3], "lambda1");
  CheckTensorScalarRank(primitive, input_args[kInputIndex4], "lambda2");
  return std::make_shared<abstract::TensorShape>(ShapeVector{});
}

TypePtr EmbeddingApplyFtrlFuncImpl::InferType(const PrimitivePtr &primitive,
                                              const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  const auto &prim_name = primitive->name();

  auto var_handle_type = input_args[kInputIndex0]->GetType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("var_handle", var_handle_type, {kInt32}, prim_name);

  auto keys_type = input_args[kInputIndex6]->GetType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("keys", keys_type, {kInt64}, prim_name);

  const std::set<TypePtr> grad_types = {kFloat16, kFloat32};
  std::map<std::string, TypePtr> type_dict;
  auto grad_type = input_args[kInputIndex5]->GetType();
  type_dict.emplace("grad", grad_type);
  auto lr_type = input_args[kInputIndex1]->GetType();
  type_dict.emplace("lr", lr_type);
  auto lr_power_type = input_args[kInputIndex2]->GetType();
  type_dict.emplace("lr_power", lr_power_type);
  auto lambda1_type = input_args[kInputIndex3]->GetType();
  type_dict.emplace("lambda1", lambda1_type);
  auto lambda2_type = input_args[kInputIndex4]->GetType();
  type_dict.emplace("lambda2", lambda2_type);
  CheckAndConvertUtils::CheckTensorTypeSame(type_dict, grad_types, prim_name);

  return std::make_shared<TensorType>(kInt32);
}
}  // namespace ops
}  // namespace mindspore
