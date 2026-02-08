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

#include "infer/ops_func_impl/rotary_position_embedding.h"
#include <algorithm>
#include <memory>
#include <set>
#include <utility>
#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/ms_context.h"
#include "utils/check_convert_utils.h"
#include "mindapi/base/types.h"
#include "ops/ops_func_impl/simple_infer.h"
#include "primitive/auto_generate/gen_ops_primitive_r.h"

namespace mindspore {
namespace ops {
TypePtr RotaryPositionEmbeddingFuncImpl::InferType(const PrimitivePtr &primitive,
                                                   const std::vector<abstract::AbstractBasePtr> &input_args) const {
  return input_args[kInputIndex0]->GetType();
}

BaseShapePtr RotaryPositionEmbeddingFuncImpl::InferShape(
  const PrimitivePtr &primitive, const std::vector<abstract::AbstractBasePtr> &input_args) const {
  return input_args[kInputIndex0]->GetShape()->Clone();
}

TypePtrList RotaryPositionEmbeddingFuncImpl::InferType(const PrimitivePtr &primitive,
                                                       const ValuePtrList &input_values) const {
  const auto &x = input_values[kInputIndex0]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(x);
  return {x->Dtype()};
}

ShapeArray RotaryPositionEmbeddingFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                       const ValuePtrList &input_values) const {
  const auto &x = input_values[kInputIndex0]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(x);
  return {x->shape()};
}

REGISTER_SIMPLE_INFER(kNameRotaryPositionEmbedding, RotaryPositionEmbeddingFuncImpl)
}  // namespace ops
}  // namespace mindspore
