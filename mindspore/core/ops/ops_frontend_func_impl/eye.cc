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

#include "ops/ops_func_impl/eye.h"
#include <memory>
#include "ops/op_utils.h"
#include "ir/dtype.h"
#include "utils/check_convert_utils.h"
#include "ops/ops_frontend_func_impl.h"

namespace mindspore {
namespace ops {
class EyeFrontendFuncImpl : public OpFrontendFuncImpl {
 public:
  ValuePtr InferValue(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    auto n_ptr = GetScalarValue<int64_t>(input_args[kInputIndex0]->GetValue());
    auto m_ptr = GetScalarValue<int64_t>(input_args[kInputIndex1]->GetValue());
    if (n_ptr.has_value() && m_ptr.has_value()) {
      auto n = n_ptr.value();
      auto m = m_ptr.value();
      if (m == 0 || n == 0) {
        MS_EXCEPTION_IF_NULL(primitive);
        auto prim_name = primitive->name();
        MS_CHECK_VALUE(n >= 0, prim_name + " error: n value can not be negative.");
        MS_CHECK_VALUE(m >= 0, prim_name + " error: m value can not be negative.");

        auto dtype_ptr = GetScalarValue<int64_t>(input_args[kInputIndex2]->GetValue());
        MS_CHECK_VALUE(dtype_ptr.has_value(), prim_name + " error: dtype input should has valid value.");
        auto dtype_id = static_cast<TypeId>(dtype_ptr.value());
        auto out_shape = ShapeVector{n, m};
        auto result_tensor = std::make_shared<tensor::Tensor>(dtype_id, out_shape);
        return result_tensor;
      }
    }
    return nullptr;
  }
};

REGISTER_PRIMITIVE_FUNCTION_FRONTEND_FUNC_IMPL("Eye", EyeFrontendFuncImpl);
}  // namespace ops
}  // namespace mindspore
