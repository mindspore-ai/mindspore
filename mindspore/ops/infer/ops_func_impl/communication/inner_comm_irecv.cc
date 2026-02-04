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

#include "infer/ops_func_impl/communication/inner_comm_irecv.h"
#include <memory>
#include "ops_utils/op_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr InnerCommIrecvFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                const std::vector<AbstractBasePtr> &input_args) const {
  auto shape_opt = GetArrayValue<int64_t>(input_args[kInputIndex2]);
  MS_CHECK_VALUE(shape_opt.has_value(), primitive->name() + " error: shape input should has valid value.");

  return std::make_shared<abstract::TensorShape>(shape_opt.value().ToVector());
}

TypePtr InnerCommIrecvFuncImpl::InferType(const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args) const {
  auto dtype_ptr = GetScalarValue<int64_t>(input_args[kInputIndex4]->GetValue());
  MS_CHECK_VALUE(dtype_ptr.has_value(), primitive->name() + " error: dtype input should has valid value.");
  auto type = TypeIdToType(static_cast<TypeId>(dtype_ptr.value()));
  return std::make_shared<TensorType>(type);
}
}  // namespace ops
}  // namespace mindspore
