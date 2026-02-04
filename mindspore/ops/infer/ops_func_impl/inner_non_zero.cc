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

#include <functional>
#include <memory>
#include "infer/ops_func_impl/inner_non_zero.h"
#include "ops/ops_frontend_func_impl.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "ops/ops_func_impl/simple_infer.h"
#include "primitive/auto_generate/gen_ops_primitive_i.h"

namespace mindspore {
namespace ops {

BaseShapePtr InnerNonZeroFuncImpl::InferShape(const PrimitivePtr &primitive,
                                              const std::vector<AbstractBasePtr> &input_args) const {
  const auto &x_shape = input_args[kInputIndex0]->GetShape()->GetShapeVector();

  MS_CHECK_VALUE(!IsDynamic(x_shape), primitive->name() + "error: shape should not has dynamic values");
  auto x_rank = SizeToLong(x_shape.size());
  auto x_num = std::accumulate(x_shape.begin(), x_shape.end(), int64_t(1), std::multiplies<int64_t>());
  return std::make_shared<abstract::Shape>(ShapeVector({x_rank, x_num}));
}

TypePtr InnerNonZeroFuncImpl::InferType(const PrimitivePtr &primitive,
                                        const std::vector<AbstractBasePtr> &input_args) const {
  return std::make_shared<TensorType>(kInt64);
}

int32_t InnerNonZeroFuncImpl::CheckValidation(const PrimitivePtr &primitive,
                                              const std::vector<AbstractBasePtr> &input_args) const {
  const std::set valid_types = {kBool,   kInt8,   kInt16,   kInt32,   kInt64,   kUInt8, kUInt16,
                                kUInt32, kUInt64, kFloat16, kFloat32, kFloat64, kFloat, kBFloat16};
  auto tensor_type = input_args[kInputIndex0]->GetType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", tensor_type, valid_types, primitive->name());
  return OP_CHECK_SUCCESS;
}

ShapeArray InnerNonZeroFuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kIndex0]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  auto x_shape = x_tensor->shape();
  auto x_rank = SizeToLong(x_shape.size());
  auto x_num = std::accumulate(x_shape.begin(), x_shape.end(), static_cast<int64_t>(1), std::multiplies<int64_t>());
  return {ShapeVector({x_rank, x_num})};
}

TypePtrList InnerNonZeroFuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kIndex0]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  return {kInt64};
}
REGISTER_SIMPLE_INFER(kNameInnerNonZero, InnerNonZeroFuncImpl)
}  // namespace ops
}  // namespace mindspore
