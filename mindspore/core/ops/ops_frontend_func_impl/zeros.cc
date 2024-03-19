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

#include "ops/ops_frontend_func_impl.h"
#include "ops/op_utils.h"
#include "utils/tensor_construct_utils.h"

namespace mindspore {
namespace ops {
class ZerosFrontendFuncImpl : public OpFrontendFuncImpl {
 public:
  ValuePtr InferValue(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto shape_value = GetArrayValue<int64_t>(input_args[kInputIndex0]);
    if (!shape_value.has_value() || shape_value.value().HasUnknownValue()) {
      return nullptr;
    }
    auto out_shape = shape_value.value().ToVector();
    if (SizeOf(out_shape) > INT_MAX) {
      MS_LOG(EXCEPTION) << "For '" << primitive->name() << "', the output elements num can not larger than " << INT_MAX
                        << "(INT_MAX), but got " << SizeOf(out_shape);
    }
    TypePtr out_type;
    auto dtype_type = input_args[kInputIndex1]->GetType();
    MS_EXCEPTION_IF_NULL(dtype_type);
    if (dtype_type->isa<TypeNone>()) {
      out_type = kFloat32;
    } else {
      auto dtype_ptr = input_args[kInputIndex1]->GetValue();
      MS_EXCEPTION_IF_NULL(dtype_ptr);
      auto val = GetValue<int64_t>(dtype_ptr);
      out_type = TypeIdToType(static_cast<TypeId>(val));
    }
    MS_EXCEPTION_IF_NULL(out_type);
    return TensorConstructUtils::CreateZerosTensor(out_type, out_shape);
  }
};

REGISTER_PRIMITIVE_FUNCTION_FRONTEND_FUNC_IMPL("Zeros", ZerosFrontendFuncImpl);
}  // namespace ops
}  // namespace mindspore
