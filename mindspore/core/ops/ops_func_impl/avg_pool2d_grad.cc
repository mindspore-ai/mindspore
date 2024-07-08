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

#include "ops/ops_func_impl/avg_pool2d_grad.h"
#include "ops/op_name.h"
#include "ops/auto_generate/gen_ops_name.h"
#include "ops/ops_func_impl/simple_infer.h"

namespace mindspore {
namespace ops {
BaseShapePtr AvgPool2DGradFuncImpl::InferShape(const PrimitivePtr &primitive,
                                               const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(input_args.at(kIndex1));
  const auto &image_shape = input_args[kIndex1]->GetShape();
  return image_shape->Clone();
}

TypePtr AvgPool2DGradFuncImpl::InferType(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(input_args.at(kIndex0));
  auto dout_type = input_args[kIndex0]->GetType();
  return dout_type;
}

ShapeArray AvgPool2DGradFuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  MS_EXCEPTION_IF_NULL(input_values.at(kIndex1));
  const auto &image = input_values[kIndex1]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(image);
  return {image->shape()};
}

TypePtrList AvgPool2DGradFuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  MS_EXCEPTION_IF_NULL(input_values.at(kIndex0));
  const auto &input = input_values[kIndex0]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(input);
  return {input->Dtype()};
}

REGISTER_SIMPLE_INFER(kNameAvgPool2DGrad, AvgPool2DGradFuncImpl)
}  // namespace ops
}  // namespace mindspore
