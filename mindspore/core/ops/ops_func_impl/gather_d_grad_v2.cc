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

#include "ops/ops_func_impl/gather_d_grad_v2.h"
#include "ops/op_name.h"

namespace mindspore {
namespace ops {
BaseShapePtr GatherDGradV2FuncImpl::InferShape(const PrimitivePtr &primitive,
                                               const std::vector<AbstractBasePtr> &input_args) const {
  return input_args[kInputIndex0]->GetShape()->Clone();
}

TypePtr GatherDGradV2FuncImpl::InferType(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) const {
  return input_args[kInputIndex0]->GetType()->Clone();
}

int32_t GatherDGradV2FuncImpl::CheckValidation(const PrimitivePtr &primitive,
                                               const std::vector<AbstractBasePtr> &input_args) const {
  auto input_shape_vec = input_args[kInputIndex0]->GetShape()->GetShapeVector();
  auto index_shape_vec = input_args[kInputIndex2]->GetShape()->GetShapeVector();
  auto grad_shape_vec = input_args[kInputIndex3]->GetShape()->GetShapeVector();

  if (IsDynamicRank(input_shape_vec) || IsDynamicRank(index_shape_vec) || IsDynamicRank(grad_shape_vec)) {
    return OP_CHECK_RETRY;
  }

  if (input_shape_vec.size() != index_shape_vec.size() || input_shape_vec.size() != grad_shape_vec.size()) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                             << "', the dimension of grad and output must be the equal to the "
                             << "dimension of index: " << index_shape_vec.size()
                             << ", but got the dimension of grad: " << grad_shape_vec.size()
                             << ", the dimension of input/output: " << input_shape_vec.size();
  }
  return OP_CHECK_SUCCESS;
}
}  // namespace ops
}  // namespace mindspore
