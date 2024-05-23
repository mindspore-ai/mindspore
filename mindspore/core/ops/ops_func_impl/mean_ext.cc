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
#include "ops/ops_func_impl/mean_ext.h"
#include <memory>
#include "ops/ops_func_impl/reduce_arithmetic.h"
#include "ops/ops_func_impl/simple_infer.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr MeanExtFuncImpl::InferShape(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) const {
  return ReduceExtandInferShape(primitive, input_args);
}

TypePtr MeanExtFuncImpl::InferType(const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) const {
  if (input_args[kIndex3]->GetType()->isa<TypeNone>()) {
    return input_args[kIndex0]->GetType()->Clone();
  }
  auto dtype_opt = GetScalarValue<int64_t>(input_args[kIndex3]->GetValue());
  MS_CHECK_VALUE(dtype_opt.has_value(), primitive->name() + " error: dtype input should has valid value.");
  return std::make_shared<TensorType>(TypeIdToType(static_cast<TypeId>(dtype_opt.value())));
}

ShapeArray MeanExtFuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  return ReduceExtandSimpleInferShape(primitive, input_values);
}

TypePtrList MeanExtFuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  if (input_values[kIndex3] == mindspore::kNone) {
    const auto &input = input_values[kIndex0]->cast<tensor::BaseTensorPtr>();
    MS_EXCEPTION_IF_NULL(input);
    return {input->Dtype()};
  } else {
    const auto &dtype = input_values[kIndex3]->cast<Int64ImmPtr>();
    MS_EXCEPTION_IF_NULL(dtype);
    return {TypeIdToType(static_cast<TypeId>(dtype->value()))};
  }
}
REGISTER_SIMPLE_INFER(kNameMeanExt, MeanExtFuncImpl)
}  // namespace ops
}  // namespace mindspore
