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

#include "ops/ops_func_impl/scalar_cast.h"
#include <vector>
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr ScalarCastFuncImpl::InferShape(const PrimitivePtr &primitive,
                                            const std::vector<AbstractBasePtr> &input_args) const {
  return abstract::kNoShape;
}

TypePtr ScalarCastFuncImpl::InferType(const PrimitivePtr &primitive,
                                      const std::vector<AbstractBasePtr> &input_args) const {
  auto dtype_ptr = GetScalarValue<int64_t>(input_args[kInputIndex1]->GetValue());
  MS_CHECK_VALUE(dtype_ptr.has_value(), primitive->name() + " error: dtype input should has valid value.");
  auto type_id = static_cast<TypeId>(dtype_ptr.value());
  auto type = TypeIdToType(type_id);
  if (type_id != kNumberTypeInt64 && type_id != kNumberTypeFloat64 && type_id != kNumberTypeBool) {
    MS_EXCEPTION(ValueError)
      << "For '" << primitive->name()
      << "', the supported type is in the list: [mindspore.int64, mindspore.float64, mindspore.bool], but got "
      << type->ToString() << ".";
  }
  return type;
}
}  // namespace ops
}  // namespace mindspore
