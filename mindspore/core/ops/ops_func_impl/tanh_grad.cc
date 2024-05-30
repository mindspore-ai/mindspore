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

#include <memory>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "ops/ops_func_impl/simple_infer.h"
#include "ops/ops_func_impl/tanh_grad.h"

namespace mindspore {
namespace ops {
BaseShapePtr TanhGradFuncImpl::InferShape(const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args) const {
  return EltwiseGradInferShape(primitive, input_args);
}

TypePtr TanhGradFuncImpl::InferType(const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const {
  return EltwiseGradInferType(primitive, input_args);
}

ShapeArray TanhGradFuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  return EltwiseGradSimpleInferShape(primitive, input_values);
}

TypePtrList TanhGradFuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  return EltwiseGradSimpleInferType(primitive, input_values);
}

REGISTER_SIMPLE_INFER(kNameTanhGrad, TanhGradFuncImpl)

}  // namespace ops
}  // namespace mindspore
