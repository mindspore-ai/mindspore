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

#include "infer/ops_func_impl/communication/inner_comm_all_reduce.h"
#include <set>
#include <string>
#include "ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace ops {
BaseShapePtr InnerCommAllReduceFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                    const std::vector<AbstractBasePtr> &input_args) const {
  return input_args[kIndex0]->GetShape()->Clone();
}

TypePtr InnerCommAllReduceFuncImpl::InferType(const PrimitivePtr &primitive,
                                              const std::vector<AbstractBasePtr> &input_args) const {
  auto type = input_args[kIndex0]->GetType();
  const std::set<TypePtr> default_target_dtypes = {kInt8, kInt32, kFloat16, kFloat32, kBFloat16, kComplex64};
  if (MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET) != kAscendDevice) {
    (void)CheckAndConvertUtils::CheckTensorTypeValid("input", type, common_valid_types_with_bool, primitive->name());
  } else {
    (void)CheckAndConvertUtils::CheckTensorTypeValid("input", type, default_target_dtypes, primitive->name());
  }
  return type;
}
}  // namespace ops
}  // namespace mindspore
