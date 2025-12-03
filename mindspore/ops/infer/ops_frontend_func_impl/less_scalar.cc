/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include <algorithm>
#include <map>
#include <memory>
#include "ir/tensor_new.h"
#include "ops/ops_frontend_func_impl.h"
#include "ops_utils/op_utils.h"
#include "utils/log_adapter.h"
#include "abstract/abstract_value.h"
#include "ops_utils/op_constants.h"

namespace mindspore {
namespace ops {
namespace {
template <typename T>
T GetScalarValueImpl(const ValuePtr &y_value) {
  if (y_value->isa<Int32Imm>()) {
    return static_cast<T>(GetValue<int32_t>(y_value));
  } else if (y_value->isa<Int64Imm>()) {
    return static_cast<T>(GetValue<int64_t>(y_value));
  } else if (y_value->isa<FP32Imm>()) {
    return static_cast<T>(GetValue<float>(y_value));
  } else if (y_value->isa<FP64Imm>()) {
    return static_cast<T>(GetValue<double>(y_value));
  } else if (y_value->isa<BoolImm>()) {
    return static_cast<T>(GetValue<bool>(y_value));
  }
  MS_LOG(EXCEPTION) << "LessScalar unsupported scalar type: " << y_value->ToString();
  return static_cast<T>(0);
}

template <typename T1, typename CommonT>
void LessScalarImpl(const void *x_data, const ValuePtr &y_value, void *result, size_t size) {
  MS_EXCEPTION_IF_NULL(x_data);
  MS_EXCEPTION_IF_NULL(result);
  const T1 *x = static_cast<const T1 *>(x_data);
  bool *result_data = static_cast<bool *>(result);

  CommonT y = GetScalarValueImpl<CommonT>(y_value);

  for (size_t i = 0; i < size; ++i) {
    result_data[i] = (static_cast<CommonT>(x[i]) < y);
  }
}

#define LESS_SCALAR_PROMOTE_CASE(TYPE, TYPE_C)                                              \
  case kNumberType##TYPE: {                                                                 \
    LessScalarImpl<T1, TYPE_C>(x_tensor->data_c(), y_value, result_tensor->data_c(), size); \
    break;                                                                                  \
  }

template <typename T1>
void LessScalarFinalDispatch(const tensor::TensorPtr &x_tensor, const ValuePtr &y_value,
                             const tensor::TensorPtr &result_tensor, TypeId x_type_id, TypeId y_type_id) {
  auto size = x_tensor->DataSize();
  switch (PromoteTensorScalarType(x_type_id, y_type_id)) {
    LESS_SCALAR_PROMOTE_CASE(Float16, float16)
    LESS_SCALAR_PROMOTE_CASE(Float32, float)
    LESS_SCALAR_PROMOTE_CASE(Float64, double)
    LESS_SCALAR_PROMOTE_CASE(Int8, int8_t)
    LESS_SCALAR_PROMOTE_CASE(Int16, int16_t)
    LESS_SCALAR_PROMOTE_CASE(Int32, int32_t)
    LESS_SCALAR_PROMOTE_CASE(Int64, int64_t)
    LESS_SCALAR_PROMOTE_CASE(UInt8, uint8_t)
    LESS_SCALAR_PROMOTE_CASE(UInt16, uint16_t)
    LESS_SCALAR_PROMOTE_CASE(UInt32, uint32_t)
    LESS_SCALAR_PROMOTE_CASE(UInt64, uint64_t)
    LESS_SCALAR_PROMOTE_CASE(BFloat16, bfloat16)
    LESS_SCALAR_PROMOTE_CASE(Bool, bool)
    default:
      MS_LOG(EXCEPTION) << "For LessScalar, the output data type is not supported.";
  }
}
#undef LESS_SCALAR_PROMOTE_CASE

#define LESS_SCALAR_T1_CASE(TYPE, TYPE_C)                                                       \
  case kNumberType##TYPE: {                                                                     \
    LessScalarFinalDispatch<TYPE_C>(x_tensor, y, result_tensor, type_id, y->type()->type_id()); \
    break;                                                                                      \
  }

}  // namespace

class LessScalarFrontendFuncImpl : public OpFrontendFuncImpl {
 public:
  ValuePtr InferValue(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    auto x = input_args[kIndex0]->GetValue();
    auto y = input_args[kIndex1]->GetValue();
    if (x == nullptr || y == nullptr || x->isa<ValueAny>() || y->isa<ValueAny>()) {
      return nullptr;
    }

    auto x_tensor = x->cast<tensor::TensorPtr>();
    if (x_tensor == nullptr) {
      return nullptr;
    }

    if (!y->isa<Scalar>()) {
      return nullptr;
    }

    auto x_shape = input_args[kIndex0]->GetShape()->GetShapeVector();
    if (IsDynamic(x_shape)) {
      return nullptr;
    }

    auto type_id = x_tensor->data_type();
    auto result_tensor = tensor::from_spec(kNumberTypeBool, x_shape, device::DeviceType::kCPU);

    switch (type_id) {
      LESS_SCALAR_T1_CASE(Float16, float16)
      LESS_SCALAR_T1_CASE(Float32, float)
      LESS_SCALAR_T1_CASE(Float64, double)
      LESS_SCALAR_T1_CASE(Int8, int8_t)
      LESS_SCALAR_T1_CASE(Int16, int16_t)
      LESS_SCALAR_T1_CASE(Int32, int32_t)
      LESS_SCALAR_T1_CASE(Int64, int64_t)
      LESS_SCALAR_T1_CASE(UInt8, uint8_t)
      LESS_SCALAR_T1_CASE(UInt16, uint16_t)
      LESS_SCALAR_T1_CASE(UInt32, uint32_t)
      LESS_SCALAR_T1_CASE(UInt64, uint64_t)
      LESS_SCALAR_T1_CASE(BFloat16, bfloat16)
      LESS_SCALAR_T1_CASE(Bool, bool)
      default:
        MS_LOG(DEBUG) << "For '" << primitive->name() << "', 'x' is " << x_tensor->ToString()
                      << ", the type is not supported.";
        return nullptr;
    }

    return result_tensor;
  }
};
#undef LESS_SCALAR_T1_CASE

REGISTER_PRIMITIVE_FUNCTION_FRONTEND_FUNC_IMPL("LessScalar", LessScalarFrontendFuncImpl);
}  // namespace ops
}  // namespace mindspore
