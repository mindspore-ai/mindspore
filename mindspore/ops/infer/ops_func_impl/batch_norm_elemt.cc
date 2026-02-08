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

#include "infer/ops_func_impl/batch_norm_elemt.h"
#include <set>
#include <vector>
#include <memory>
#include "primitive/op_name.h"
#include "ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "infer/ops_func_impl/common_infer_fns.h"

namespace mindspore {
namespace ops {
BaseShapePtr BatchNormElemtFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                const std::vector<AbstractBasePtr> &input_args) const {
  auto input_shape = input_args[kInputIndex0]->GetShape();
  return input_shape->Clone();
}

TypePtr BatchNormElemtFuncImpl::InferType(const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args) const {
  auto op_name = primitive->name();
  const std::set<TypePtr> tensor_valid_types = {kFloat32, kFloat16, kBFloat16};
  TypePtr input_type = input_args[kInputIndex0]->GetType();
  (void)CheckAndConvertUtils::CheckTypeValid("input", input_type, tensor_valid_types, op_name);
  if (!IsOptionalInputNone(input_args[kInputIndex1])) {
    (void)CheckAndConvertUtils::CheckTypeValid("weight", input_args[kInputIndex1]->GetType(), tensor_valid_types,
                                               op_name);
  }
  if (!IsOptionalInputNone(input_args[kInputIndex2])) {
    (void)CheckAndConvertUtils::CheckTypeValid("bias", input_args[kInputIndex2]->GetType(), tensor_valid_types,
                                               op_name);
  }
  if (!IsOptionalInputNone(input_args[kInputIndex3])) {
    (void)CheckAndConvertUtils::CheckTypeValid("mean", input_args[kInputIndex3]->GetType(), tensor_valid_types,
                                               op_name);
  } else {
    MS_LOG(EXCEPTION) << "For '" << op_name
                      << "', the type of 'mean' must be Tensor[kFloat32, kFloat16, kBFloat16], but got None.";
  }
  if (!IsOptionalInputNone(input_args[kInputIndex4])) {
    (void)CheckAndConvertUtils::CheckTypeValid("invstd", input_args[kInputIndex4]->GetType(), tensor_valid_types,
                                               op_name);
  } else {
    MS_LOG(EXCEPTION) << "For '" << op_name
                      << "', the type of 'invstd' must be Tensor[kFloat32, kFloat16, kBFloat16], but got None.";
  }
  return input_type->Clone();
}
}  // namespace ops
}  // namespace mindspore
