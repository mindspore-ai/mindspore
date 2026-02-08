/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "infer/ops_func_impl/inplace_stop_gradient.h"
#include <vector>
#include "primitive/auto_generate/gen_ops_name_i.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "ops/ops_func_impl/simple_infer.h"

namespace mindspore {
namespace ops {
BaseShapePtr InplaceStopGradientFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                     const std::vector<AbstractBasePtr> &input_args) const {
  return nullptr;
}

TypePtr InplaceStopGradientFuncImpl::InferType(const PrimitivePtr &primitive,
                                               const std::vector<AbstractBasePtr> &input_args) const {
  return nullptr;
}

TypePtrList InplaceStopGradientFuncImpl::InferType(const PrimitivePtr &primitive,
                                                   const ValuePtrList &input_values) const {
  return {};
}

ShapeArray InplaceStopGradientFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                   const ValuePtrList &input_values) const {
  return {};
}

REGISTER_SIMPLE_INFER(kNameInplaceStopGradient, InplaceStopGradientFuncImpl)
}  // namespace ops
}  // namespace mindspore
