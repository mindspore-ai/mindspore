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

#include <vector>
#include <memory>
#include "primitive/op_name.h"
#include "infer/ops_func_impl/prelu_grad.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "ops/ops_func_impl/simple_infer.h"
#include "utils/ms_context.h"
#include "primitive/auto_generate/gen_ops_primitive_p.h"

namespace mindspore::ops {
BaseShapePtr PReLUGradFuncImpl::InferShape(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) const {
  auto x_shape = input_args[kIndex1]->GetShape();
  auto w_shape = input_args[kIndex2]->GetShape();
  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{x_shape, w_shape});
}

TypePtr PReLUGradFuncImpl::InferType(const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) const {
  auto x_type = input_args[kIndex1]->GetType();
  auto w_type = input_args[kIndex2]->GetType();
  return std::make_shared<Tuple>(std::vector<TypePtr>{x_type, w_type});
}

TypePtrList PReLUGradFuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kIndex1]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  const auto &w_tensor = input_values[kIndex2]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(w_tensor);
  return {x_tensor->Dtype(), w_tensor->Dtype()};
}

ShapeArray PReLUGradFuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kIndex1]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  const auto &w_tensor = input_values[kIndex2]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(w_tensor);
  return {x_tensor->shape(), w_tensor->shape()};
}
REGISTER_SIMPLE_INFER(kNamePReLUGrad, PReLUGradFuncImpl)
}  // namespace mindspore::ops
