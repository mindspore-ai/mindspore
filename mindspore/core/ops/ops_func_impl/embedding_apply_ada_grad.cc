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

#include "ops/ops_func_impl/embedding_apply_ada_grad.h"

#include <vector>
#include <string>
#include <set>
#include <memory>

#include "ops/op_utils.h"
#include "utils/shape_utils.h"
#include "utils/check_convert_utils.h"
#include "ops/op_name.h"

namespace mindspore {
namespace ops {
BaseShapePtr EmbeddingApplyAdaGradFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                       const std::vector<AbstractBasePtr> &input_args) const {
  CheckTensorScalarRank(primitive, input_args[kInputIndex0], "var_handle");
  CheckTensorScalarRank(primitive, input_args[kInputIndex1], "lr");
  CheckTensorScalarRank(primitive, input_args[kInputIndex4], "global_step");

  return std::make_shared<abstract::Shape>(ShapeVector{});
}

TypePtr EmbeddingApplyAdaGradFuncImpl::InferType(const PrimitivePtr &primitive,
                                                 const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  const std::string &prim_name = primitive->name();

  auto var_handle_type = input_args[kInputIndex0]->GetType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("var_handle", var_handle_type, {kInt32}, prim_name);

  const std::set<TypePtr> valid_types = {kFloat16, kFloat32};
  auto lr_type = input_args[kInputIndex1]->GetType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("lr", lr_type, valid_types, prim_name);

  auto grad_type = input_args[kInputIndex2]->GetType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("grad", grad_type, valid_types, prim_name);

  auto keys_type = input_args[kInputIndex3]->GetType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("keys", keys_type, {kInt64}, prim_name);

  auto global_step_type = input_args[kInputIndex4]->GetType();
  const std::set<TypePtr> global_types = {kInt32, kInt64};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("global_step", global_step_type, global_types, prim_name);

  return std::make_shared<TensorType>(kInt32);
}
}  // namespace ops
}  // namespace mindspore
