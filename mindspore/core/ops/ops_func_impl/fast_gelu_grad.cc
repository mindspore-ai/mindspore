/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "ops/ops_func_impl/fast_gelu_grad.h"
#include "ops/op_utils.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace ops {
BaseShapePtr FastGeLUGradFuncImpl::InferShape(const PrimitivePtr &primitive,
                                              const std::vector<AbstractBasePtr> &input_args) const {
  return input_args[kInputIndex0]->GetShape()->Clone();
}

TypePtr FastGeLUGradFuncImpl::InferType(const PrimitivePtr &primitive,
                                        const std::vector<AbstractBasePtr> &input_args) const {
  std::vector<TypeId> valid_types = {kNumberTypeFloat16, kNumberTypeFloat32};
  auto tensor_type = input_args[0]->GetType()->cast<TensorTypePtr>();
  auto real_type = tensor_type->element()->type_id();
  if (std::find(valid_types.begin(), valid_types.end(), real_type) == valid_types.end()) {
    MS_EXCEPTION(TypeError) << "For '" << primitive->name() << "', input[0] type should be float16 or float32. but got "
                            << tensor_type->element()->ToString();
  }
  return input_args[kInputIndex0]->GetType()->Clone();
}
}  // namespace ops
}  // namespace mindspore
