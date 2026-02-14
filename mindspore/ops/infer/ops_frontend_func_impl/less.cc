/**
 * Copyright 2023-2025 Huawei Technologies Co., Ltd
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
#include <complex>
#include <limits>
#include <map>
#include <memory>
#include <vector>
#include "ir/tensor_new.h"
#include "ops/ops_frontend_func_impl.h"
#include "ops_utils/op_utils.h"
#include "utils/log_adapter.h"
#include "abstract/abstract_value.h"
#include "ops_utils/op_constants.h"

namespace mindspore {
namespace ops {
namespace {
template <typename T1, typename T2, typename CommonT>
void LessImpl(const void *x1, const void *x2, void *result, size_t size) {
  MS_EXCEPTION_IF_NULL(x1);
  MS_EXCEPTION_IF_NULL(x2);
  MS_EXCEPTION_IF_NULL(result);
  const T1 *x1_data = static_cast<const T1 *>(x1);
  const T2 *x2_data = static_cast<const T2 *>(x2);
  auto result_data = static_cast<bool *>(result);
  for (size_t i = 0; i < size; ++i) {
    result_data[i] = (static_cast<CommonT>(x1_data[i]) < static_cast<CommonT>(x2_data[i]));
  }
}

#define LESS_PROMOTE_CASE(TYPE, TYPE_C)                                                                \
  case kNumberType##TYPE: {                                                                            \
    LessImpl<T1, T2, TYPE_C>(x1_tensor->data_c(), x2_tensor->data_c(), result_tensor->data_c(), size); \
    break;                                                                                             \
  }

template <typename T1, typename T2>
void LessFinalDispatch(const tensor::TensorPtr &x1_tensor, const tensor::TensorPtr &x2_tensor,
                       const tensor::TensorPtr &result_tensor, const std::string &op_name) {
  auto type_id1 = x1_tensor->data_type();
  auto type_id2 = x2_tensor->data_type();
  auto size = x1_tensor->DataSize();
  switch (PromoteType(type_id1, type_id2, op_name)) {
    LESS_PROMOTE_CASE(Float16, float16)
    LESS_PROMOTE_CASE(Float32, float)
    LESS_PROMOTE_CASE(Float64, double)
    LESS_PROMOTE_CASE(Int8, int8_t)
    LESS_PROMOTE_CASE(Int16, int16_t)
    LESS_PROMOTE_CASE(Int32, int32_t)
    LESS_PROMOTE_CASE(Int64, int64_t)
    LESS_PROMOTE_CASE(UInt8, uint8_t)
    LESS_PROMOTE_CASE(UInt16, uint16_t)
    LESS_PROMOTE_CASE(UInt32, uint32_t)
    LESS_PROMOTE_CASE(UInt64, uint64_t)
    LESS_PROMOTE_CASE(BFloat16, bfloat16)
    LESS_PROMOTE_CASE(Bool, bool)
    default:
      MS_LOG(EXCEPTION) << "For Less, the output data type is not supported.";
  }
}
#undef LESS_PROMOTE_CASE

#define LESS_T2_CASE(TYPE, TYPE_C)                                               \
  case kNumberType##TYPE: {                                                      \
    LessFinalDispatch<T1, TYPE_C>(x1_tensor, x2_tensor, result_tensor, op_name); \
    break;                                                                       \
  }

template <typename T1>
void LessDispatchT2(const tensor::TensorPtr &x1_tensor, const tensor::TensorPtr &x2_tensor,
                    const tensor::TensorPtr &result_tensor, const std::string &op_name) {
  switch (x2_tensor->data_type()) {
    LESS_T2_CASE(Float16, float16)
    LESS_T2_CASE(Float32, float)
    LESS_T2_CASE(Float64, double)
    LESS_T2_CASE(Int8, int8_t)
    LESS_T2_CASE(Int16, int16_t)
    LESS_T2_CASE(Int32, int32_t)
    LESS_T2_CASE(Int64, int64_t)
    LESS_T2_CASE(UInt8, uint8_t)
    LESS_T2_CASE(UInt16, uint16_t)
    LESS_T2_CASE(UInt32, uint32_t)
    LESS_T2_CASE(UInt64, uint64_t)
    LESS_T2_CASE(BFloat16, bfloat16)
    LESS_T2_CASE(Bool, bool)
    default:
      MS_LOG(EXCEPTION) << "For Less, the second input data type is not supported.";
  }
}
#undef LESS_T2_CASE

}  // namespace

class LessFrontendFuncImpl : public OpFrontendFuncImpl {
 public:
  ValuePtr InferValue(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    auto x1 = input_args[kIndex0]->GetValue();
    auto x2 = input_args[kIndex1]->GetValue();
    if (x1 == nullptr || x2 == nullptr || x1->isa<ValueAny>() || x2->isa<ValueAny>()) {
      return nullptr;
    }
    auto x1_tensor = x1->cast<tensor::TensorPtr>();
    auto x2_tensor = x2->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(x1_tensor);
    MS_EXCEPTION_IF_NULL(x2_tensor);

    auto x1_shape = input_args[kIndex0]->GetShape()->GetShapeVector();
    auto x2_shape = input_args[kIndex1]->GetShape()->GetShapeVector();
    if (IsDynamic(x1_shape) || IsDynamic(x2_shape) || !IsMactchedShapeInferValue(x1_shape, x2_shape)) {
      return nullptr;
    }
    auto type_id1 = x1_tensor->data_type();
    auto result_tensor = tensor::from_spec(kNumberTypeBool, x1_shape, device::DeviceType::kCPU);

#define LESS_T1_CASE(TYPE, TYPE_C)                                                  \
  case kNumberType##TYPE: {                                                         \
    LessDispatchT2<TYPE_C>(x1_tensor, x2_tensor, result_tensor, primitive->name()); \
    break;                                                                          \
  }

    switch (type_id1) {
      LESS_T1_CASE(Float16, float16)
      LESS_T1_CASE(Float32, float)
      LESS_T1_CASE(Float64, double)
      LESS_T1_CASE(Int8, int8_t)
      LESS_T1_CASE(Int16, int16_t)
      LESS_T1_CASE(Int32, int32_t)
      LESS_T1_CASE(Int64, int64_t)
      LESS_T1_CASE(UInt8, uint8_t)
      LESS_T1_CASE(UInt16, uint16_t)
      LESS_T1_CASE(UInt32, uint32_t)
      LESS_T1_CASE(UInt64, uint64_t)
      LESS_T1_CASE(BFloat16, bfloat16)
      LESS_T1_CASE(Bool, bool)
      default:
        MS_LOG(DEBUG) << "For '" << primitive->name() << "', 'x1' is " << x1_tensor->ToString()
                      << ", the type is not supported.";
        return nullptr;
    }
#undef LESS_T1_CASE
    return result_tensor;
  }
};

REGISTER_PRIMITIVE_FUNCTION_FRONTEND_FUNC_IMPL("Less", LessFrontendFuncImpl);
}  // namespace ops
}  // namespace mindspore
