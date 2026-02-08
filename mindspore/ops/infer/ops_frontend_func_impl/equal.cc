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

#include <algorithm>
#include <complex>
#include <limits>
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
template <typename T>
void EqualImpl(const void *x1, const void *x2, void *result, size_t size) {
  const T *x1_data = static_cast<const T *>(x1);
  const T *x2_data = static_cast<const T *>(x2);
  auto result_data = static_cast<bool *>(result);
  for (size_t i = 0; i < size; ++i) {
    result_data[i] = x1_data[i] == x2_data[i];
  }
}

template <typename T>
void EqualFloatImpl(const void *x1, const void *x2, void *result, size_t size) {
  const T *x1_data = static_cast<const T *>(x1);
  const T *x2_data = static_cast<const T *>(x2);
  auto result_data = static_cast<bool *>(result);
  for (size_t i = 0; i < size; ++i) {
    result_data[i] = std::abs(x1_data[i] - x2_data[i]) < std::numeric_limits<T>::epsilon();
  }
}

using Handler = std::function<void(const void *x1, const void *x2, void *result, size_t size)>;
std::map<TypeId, Handler> equal_impl_list = {{kNumberTypeBool, EqualImpl<bool>},
                                             {kNumberTypeInt8, EqualImpl<int8_t>},
                                             {kNumberTypeInt16, EqualImpl<int16_t>},
                                             {kNumberTypeInt32, EqualImpl<int32_t>},
                                             {kNumberTypeInt64, EqualImpl<int64_t>},
                                             {kNumberTypeUInt8, EqualImpl<uint8_t>},
                                             {kNumberTypeUInt16, EqualImpl<uint16_t>},
                                             {kNumberTypeUInt32, EqualImpl<uint32_t>},
                                             {kNumberTypeUInt64, EqualImpl<uint64_t>},
                                             {kNumberTypeFloat16, EqualImpl<float16>},
                                             {kNumberTypeBFloat16, EqualImpl<bfloat16>},
                                             {kNumberTypeFloat32, EqualFloatImpl<float>},
                                             {kNumberTypeFloat64, EqualFloatImpl<double>},
                                             {kNumberTypeComplex64, EqualImpl<std::complex<float>>},
                                             {kNumberTypeComplex128, EqualImpl<std::complex<double>>}};

class EqualFrontendFuncImpl : public OpFrontendFuncImpl {
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
    auto type_id = x1_tensor->data_type();
    auto data_size = x1_tensor->DataSize();
    auto result_tensor = tensor::from_spec(kNumberTypeBool, x1_shape, device::DeviceType::kCPU);
    auto iter = equal_impl_list.find(type_id);
    if (iter == equal_impl_list.end()) {
      MS_LOG(DEBUG) << "For '" << primitive->name() << "', 'x1' is " << x1_tensor->ToString()
                    << ", the type is not supported.";
      return nullptr;
    }
    auto x1_tensor_cpu = x1_tensor->cpu();
    auto x2_tensor_cpu = x2_tensor->cpu();
    iter->second(x1_tensor_cpu->data_c(), x2_tensor_cpu->data_c(), result_tensor->data_c(), data_size);
    return result_tensor;
  }
};

REGISTER_PRIMITIVE_FUNCTION_FRONTEND_FUNC_IMPL("Equal", EqualFrontendFuncImpl);
}  // namespace ops
}  // namespace mindspore
