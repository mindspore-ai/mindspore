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
#include "infer/ops_func_impl/round.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "ops/ops_func_impl/simple_infer.h"
#include "primitive/auto_generate/gen_ops_primitive_r.h"

namespace mindspore {
namespace ops {
BaseShapePtr RoundFuncImpl::InferShape(const PrimitivePtr &primitive,
                                       const std::vector<AbstractBasePtr> &input_args) const {
  return input_args[kIndex0]->GetShape()->Clone();
}

TypePtr RoundFuncImpl::InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
  return input_args[kIndex0]->GetType();
}

TypePtrList RoundFuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[0]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  return {x_tensor->Dtype()};
}

ShapeArray RoundFuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[0]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  return {x_tensor->shape()};
}
REGISTER_SIMPLE_INFER(kNameRound, RoundFuncImpl)
}  // namespace ops
}  // namespace mindspore
