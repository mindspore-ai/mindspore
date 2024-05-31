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
#include "ops/ops_func_impl/anti_quant.h"
#include <memory>
#include "abstract/dshape.h"
#include "ops/op_name.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr AntiQuantFuncImpl::InferShape(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) const {
  // Get input tensor shape.
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex0]);
  auto x_shape = input_args[kInputIndex0]->GetShape();
  MS_EXCEPTION_IF_NULL(x_shape);

  return x_shape->Clone();
}

TypePtr AntiQuantFuncImpl::InferType(const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) const {
  auto dtype_ptr = GetScalarValue<int64_t>(input_args[kInputIndex4]->GetValue());
  auto type = TypeIdToType(static_cast<TypeId>(dtype_ptr.value()));
  MS_CHECK_VALUE(type == kFloat16 || type == kBFloat16, primitive->name() + " error: dst_type should be " +
                                                          kFloat16->ToString() + " or " + kBFloat16->ToString() +
                                                          " but got " + type->ToString());
  return std::make_shared<TensorType>(type);
}

}  // namespace ops
}  // namespace mindspore
