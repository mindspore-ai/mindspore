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

#include "infer/ops_func_impl/rotary_position_embedding_grad.h"
#include <algorithm>
#include <memory>
#include <set>
#include <utility>
#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/ms_context.h"
#include "utils/check_convert_utils.h"
#include "ops/ops_func_impl/simple_infer.h"
#include "primitive/auto_generate/gen_ops_primitive_r.h"

namespace mindspore {
namespace ops {
TypePtr RotaryPositionEmbeddingGradFuncImpl::InferType(const PrimitivePtr &primitive,
                                                       const std::vector<abstract::AbstractBasePtr> &input_args) const {
  const auto &dx_type = input_args[kInputIndex0]->GetType();
  const auto &cos_type = input_args[kInputIndex1]->GetType();
  const auto &sin_type = input_args[kInputIndex2]->GetType();
  return std::make_shared<Tuple>(std::vector{dx_type, cos_type, sin_type});
}

BaseShapePtr RotaryPositionEmbeddingGradFuncImpl::InferShape(
  const PrimitivePtr &primitive, const std::vector<abstract::AbstractBasePtr> &input_args) const {
  const auto &dx_shape_ptr = input_args[kInputIndex0]->GetShape();
  const auto &cos_shape_ptr = input_args[kInputIndex1]->GetShape();
  const auto &sin_shape_ptr = input_args[kInputIndex2]->GetShape();
  return std::make_shared<abstract::TupleShape>(
    std::vector<abstract::BaseShapePtr>{dx_shape_ptr, cos_shape_ptr, sin_shape_ptr});
}

TypePtrList RotaryPositionEmbeddingGradFuncImpl::InferType(const PrimitivePtr &primitive,
                                                           const ValuePtrList &input_values) const {
  const auto &dx_tensor = input_values[kInputIndex0]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(dx_tensor);
  const auto &cos_tensor = input_values[kInputIndex1]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(cos_tensor);
  const auto &sin_tensor = input_values[kInputIndex2]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(sin_tensor);
  return {dx_tensor->Dtype(), cos_tensor->Dtype(), sin_tensor->Dtype()};
}

ShapeArray RotaryPositionEmbeddingGradFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                           const ValuePtrList &input_values) const {
  const auto &dx_tensor = input_values[kInputIndex0]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(dx_tensor);
  const auto &cos_tensor = input_values[kInputIndex1]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(cos_tensor);
  const auto &sin_tensor = input_values[kInputIndex2]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(sin_tensor);
  return {dx_tensor->shape(), cos_tensor->shape(), sin_tensor->shape()};
}

REGISTER_SIMPLE_INFER(kNameRotaryPositionEmbeddingGrad, RotaryPositionEmbeddingGradFuncImpl)
}  // namespace ops
}  // namespace mindspore
