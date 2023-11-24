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
#include "ops/ops_frontend_func_impl.h"
#include "ops/op_utils.h"
#include "utils/log_adapter.h"
#include "abstract/abstract_value.h"

namespace mindspore {
namespace ops {
template <typename T>
void NotEqualImpl(void *x1, void *x2, void *result, size_t size) {
  MS_EXCEPTION_IF_NULL(x1);
  MS_EXCEPTION_IF_NULL(x2);
  MS_EXCEPTION_IF_NULL(result);
  T *x1_data = static_cast<T *>(x1);
  T *x2_data = static_cast<T *>(x2);
  auto result_data = static_cast<bool *>(result);
  for (size_t i = 0; i < size; ++i) {
    result_data[i] = !(x1_data[i] == x2_data[i]);
  }
}

template <typename T>
void NotEqualFloatImpl(void *x1, void *x2, void *result, size_t size) {
  MS_EXCEPTION_IF_NULL(x1);
  MS_EXCEPTION_IF_NULL(x2);
  MS_EXCEPTION_IF_NULL(result);
  T *x1_data = static_cast<T *>(x1);
  T *x2_data = static_cast<T *>(x2);
  auto result_data = static_cast<bool *>(result);
  for (size_t i = 0; i < size; ++i) {
    result_data[i] = std::abs(x1_data[i] - x2_data[i]) > std::numeric_limits<T>::epsilon();
  }
}

using Handler = std::function<void(void *x1, void *x2, void *result, size_t size)>;
std::map<TypeId, Handler> not_equal_impl_list = {{kNumberTypeBool, NotEqualImpl<bool>},
                                                 {kNumberTypeInt, NotEqualImpl<int>},
                                                 {kNumberTypeInt8, NotEqualImpl<int8_t>},
                                                 {kNumberTypeInt16, NotEqualImpl<int16_t>},
                                                 {kNumberTypeInt32, NotEqualImpl<int32_t>},
                                                 {kNumberTypeInt64, NotEqualImpl<int64_t>},
                                                 {kNumberTypeUInt8, NotEqualImpl<uint8_t>},
                                                 {kNumberTypeFloat, NotEqualFloatImpl<float>},
                                                 {kNumberTypeFloat16, NotEqualImpl<float16>},
                                                 {kNumberTypeBFloat16, NotEqualImpl<bfloat16>},
                                                 {kNumberTypeFloat32, NotEqualImpl<float>},
                                                 {kNumberTypeFloat64, NotEqualImpl<double>},
                                                 {kNumberTypeComplex64, NotEqualImpl<std::complex<float>>},
                                                 {kNumberTypeComplex128, NotEqualImpl<std::complex<double>>}};

class NotEqualFrontendFuncImpl : public OpFrontendFuncImpl {
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
    auto result_tensor = std::make_shared<tensor::Tensor>(kNumberTypeBool, x1_shape);
    auto iter = not_equal_impl_list.find(type_id);
    if (iter == not_equal_impl_list.end()) {
      MS_EXCEPTION(TypeError) << "For '" << primitive->name() << "', 'x1' is " << x1_tensor->ToString()
                              << ", the type is not supported.";
    }
    iter->second(x1_tensor->data_c(), x2_tensor->data_c(), result_tensor->data_c(), data_size);
    return result_tensor;
  }
};

REGISTER_PRIMITIVE_FUNCTION_FRONTEND_FUNC_IMPL("NotEqual", NotEqualFrontendFuncImpl);
}  // namespace ops
}  // namespace mindspore
