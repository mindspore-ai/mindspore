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
#include <string>
#include "ops/op_utils.h"
#include "ir/dtype.h"
#include "utils/check_convert_utils.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace ops {
BaseShapePtr EyeFuncImpl::InferShape(const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) const {
  int64_t n = abstract::Shape::kShapeDimAny;
  int64_t m = abstract::Shape::kShapeDimAny;
  auto n_ptr = GetScalarValue<int64_t>(input_args[0]->GetValue());
  auto m_ptr = GetScalarValue<int64_t>(input_args[1]->GetValue());

  if (n_ptr.has_value()) {
    n = n_ptr.value();
  }
  if (m_ptr.has_value()) {
    m = m_ptr.value();
  }

  return std::make_shared<abstract::TensorShape>(ShapeVector{n, m});
}

TypePtr EyeFuncImpl::InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
  auto dtype_ptr = GetScalarValue<int64_t>(input_args[kInputIndex2]->GetValue());
  MS_CHECK_VALUE(dtype_ptr.has_value(), primitive->name() + " error: dtype input should has valid value.");
  return std::make_shared<TensorType>(TypeIdToType(static_cast<TypeId>(dtype_ptr.value())));
}

int32_t EyeFuncImpl::CheckValidation(const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto n_ptr = GetScalarValue<int64_t>(input_args[kInputIndex0]->GetValue());
  auto m_ptr = GetScalarValue<int64_t>(input_args[kInputIndex1]->GetValue());
  auto dtype_ptr = GetScalarValue<int64_t>(input_args[kInputIndex2]->GetValue());

  if (!n_ptr.has_value() || !m_ptr.has_value() || !dtype_ptr.has_value()) {
    return OP_CHECK_RETRY;
  }

  MS_CHECK_VALUE(n_ptr.value() >= 0, prim_name + " error: n value can not be negative.");
  MS_CHECK_VALUE(m_ptr.value() >= 0, prim_name + " error: m value can not be negative.");

  // Ascend GE Eye unsupported float64/uint16/uint32/uint64/complex64/complex128
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  bool is_ascend = (context->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kAscendDevice);
  if (is_ascend) {
    auto dtype_value = TypeIdToType(static_cast<TypeId>(dtype_ptr.value()));
    std::set<TypePtr> valid_types = {{kInt8, kUInt8, kInt16, kInt32, kInt64, kFloat16, kFloat32, kBool}};
    (void)CheckAndConvertUtils::CheckSubClass("dtype", dtype_value, valid_types, prim_name);
  }

  return OP_CHECK_SUCCESS;
}
}  // namespace ops
}  // namespace mindspore
