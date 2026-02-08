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
#include "infer/ops_func_impl/uniform_ext.h"
#include <utility>
#include <memory>
#include "mindspore/ops/ops_utils/op_utils.h"
#include "ir/dtype.h"
#include "primitive/op_name.h"
#include "utils/check_convert_utils.h"
#include "ops/ops_func_impl/simple_infer.h"
#include "primitive/auto_generate/gen_ops_primitive_u.h"

namespace mindspore {
namespace ops {

BaseShapePtr UniformExtFuncImpl::InferShape(const PrimitivePtr &primitive,
                                            const std::vector<AbstractBasePtr> &input_args) const {
  return input_args[0]->GetShape();
}

TypePtr UniformExtFuncImpl::InferType(const PrimitivePtr &primitive,
                                      const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto x_type = input_args[0]->GetType();
  const std::set<TypePtr> valid_types = {kInt8,    kInt16,   kInt32,   kInt64, kUInt8,
                                         kFloat16, kFloat32, kFloat64, kBool,  kBFloat16};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x_type, valid_types, prim_name);
  return x_type;
}

TypePtrList UniformExtFuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kInputIndex0]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  return {x_tensor->Dtype()};
}

ShapeArray UniformExtFuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kInputIndex0]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  return {x_tensor->shape()};
}
REGISTER_SIMPLE_INFER(kNameUniformExt, UniformExtFuncImpl)
}  // namespace ops
}  // namespace mindspore
