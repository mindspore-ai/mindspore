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

#include "ops/ops_func_impl/binary_cross_entropy_grad.h"
#include <algorithm>
#include <memory>
#include <set>
#include <utility>
#include "ops/op_utils.h"
#include "utils/ms_context.h"
#include "utils/check_convert_utils.h"
#include "ops/ops_func_impl/simple_infer.h"

namespace mindspore {
namespace ops {
TypePtr BinaryCrossEntropyGradFuncImpl::InferType(const PrimitivePtr &primitive,
                                                  const std::vector<abstract::AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex1]);
  auto input_type = input_args[kInputIndex1]->GetType();
  return input_type;
}

BaseShapePtr BinaryCrossEntropyGradFuncImpl::InferShape(
  const PrimitivePtr &primitive, const std::vector<abstract::AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex1]);
  auto input_shape_ptr = input_args[kInputIndex1]->GetShape();

  return input_shape_ptr->Clone();
}

TypePtrList BinaryCrossEntropyGradFuncImpl::InferType(const PrimitivePtr &primitive,
                                                      const ValuePtrList &input_values) const {
  const auto &input_tensor = input_values[kInputIndex1]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(input_tensor);
  return {input_tensor->Dtype()};
}

ShapeArray BinaryCrossEntropyGradFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                      const ValuePtrList &input_values) const {
  const auto &input_tensor = input_values[kInputIndex1]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(input_tensor);
  auto input_shape = input_tensor->shape();
  return {input_shape};
}

REGISTER_SIMPLE_INFER(kNameBinaryCrossEntropyGrad, BinaryCrossEntropyGradFuncImpl)

}  // namespace ops
}  // namespace mindspore
