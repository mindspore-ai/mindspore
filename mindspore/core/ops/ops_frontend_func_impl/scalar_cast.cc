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

#include "ops/op_utils.h"
#include "ops/ops_frontend_func_impl.h"
#include "abstract/abstract_value.h"

namespace mindspore {
namespace ops {
class ScalarCastFrontendFuncImpl : public OpFrontendFuncImpl {
 public:
  ValuePtr InferValue(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
    auto op_name = primitive->name();
    auto elem_x = input_args[kInputIndex0];

    auto x_value = elem_x->GetValue();
    if (x_value == nullptr || x_value->ContainsValueAny()) {
      return nullptr;
    }

    auto dtype_ptr = GetScalarValue<int64_t>(input_args[kInputIndex1]->GetValue());
    if (!dtype_ptr.has_value()) {
      return nullptr;
    }
    auto res_type = TypeIdToType(static_cast<TypeId>(dtype_ptr.value()));
    switch (res_type->type_id()) {
      case kNumberTypeInt64:
        return MakeValue(GetScalarCastValue<int64_t>(op_name, x_value));
      case kNumberTypeFloat32:
        return MakeValue(GetScalarCastValue<float>(op_name, x_value));
      case kNumberTypeFloat64:
        return MakeValue(GetScalarCastValue<double>(op_name, x_value));
      case kNumberTypeBool:
        return MakeValue(GetScalarCastValue<bool>(op_name, x_value));
      default: {
        MS_EXCEPTION(ValueError)
          << "For '" << op_name
          << "', the supported type is in the list: [mindspore.int64, mindspore.float64, mindspore.bool], but got "
          << res_type->ToString() << ".";
      }
    }
  }
};
REGISTER_PRIMITIVE_FUNCTION_FRONTEND_FUNC_IMPL("ScalarCast", ScalarCastFrontendFuncImpl);
}  // namespace ops
}  // namespace mindspore
