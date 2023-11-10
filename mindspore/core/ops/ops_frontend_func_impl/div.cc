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
#include "ops/ops_func_impl/div.h"

namespace mindspore {
namespace ops {
template <typename T>
void DivImpl(void *x, void *y, void *result, size_t size) {
  MS_EXCEPTION_IF_NULL(x);
  MS_EXCEPTION_IF_NULL(y);
  MS_EXCEPTION_IF_NULL(result);
  T *x_data = static_cast<T *>(x);
  T *y_data = static_cast<T *>(y);
  auto result_data = static_cast<T *>(result);
  MS_EXCEPTION_IF_NULL(x_data);
  MS_EXCEPTION_IF_NULL(y_data);
  MS_EXCEPTION_IF_NULL(result_data);
  auto zero = static_cast<T>(0);
  for (size_t i = 0; i < size; ++i) {
    if (y_data[i] == zero) {
      if (x_data[i] == zero) {
        result_data[i] = std::numeric_limits<T>::quiet_NaN();
        continue;
      }
      if (std::numeric_limits<T>::has_infinity) {
        result_data[i] = x_data[i] > zero ? std::numeric_limits<T>::infinity() : -std::numeric_limits<T>::infinity();
      } else {
        result_data[i] = x_data[i] > zero ? std::numeric_limits<T>::max() : std::numeric_limits<T>::min();
      }
      continue;
    } else {
      result_data[i] = x_data[i] / y_data[i];
    }
  }
}

template <typename T>
void ComplexDivImpl(void *x, void *y, void *result, size_t size) {
  MS_EXCEPTION_IF_NULL(x);
  MS_EXCEPTION_IF_NULL(y);
  MS_EXCEPTION_IF_NULL(result);
  T *x_data = static_cast<T *>(x);
  T *y_data = static_cast<T *>(y);
  auto result_data = static_cast<T *>(result);
  MS_EXCEPTION_IF_NULL(x_data);
  MS_EXCEPTION_IF_NULL(y_data);
  MS_EXCEPTION_IF_NULL(result_data);
  auto zero = static_cast<T>(0);
  for (size_t i = 0; i < size; ++i) {
    if (y_data[i] == zero) {
      if (x_data[i] == zero) {
        result_data[i] = std::numeric_limits<T>::quiet_NaN();
        continue;
      }
      continue;
    }
    result_data[i] = static_cast<T>(x_data[i] / y_data[i]);
  }
}

class DivFrontendFuncImpl : public OpFrontendFuncImpl {
 public:
  ValuePtr InferValue(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    auto x = input_args[kIndex0]->GetValue();
    auto y = input_args[kIndex1]->GetValue();
    if (x == nullptr || y == nullptr || x->isa<ValueAny>() || y->isa<ValueAny>()) {
      return nullptr;
    }
    auto x_tensor = x->cast<tensor::TensorPtr>();
    auto y_tensor = y->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(x_tensor);
    MS_EXCEPTION_IF_NULL(y_tensor);
    auto x_shape = input_args[kIndex0]->GetShape()->GetShapeVector();
    auto y_shape = input_args[kIndex1]->GetShape()->GetShapeVector();
    if (IsDynamic(x_shape) || IsDynamic(y_shape) || !IsMactchedShapeInferValue(x_shape, y_shape)) {
      return nullptr;
    }
    auto data_size = x_tensor->DataSize();
    auto type_id = x_tensor->data_type();
    auto result_tensor = std::make_shared<tensor::Tensor>(type_id, x_shape);
    MS_EXCEPTION_IF_NULL(result_tensor);
    auto result_datac = result_tensor->data_c();
    auto iter = func_map.find(type_id);
    if (iter != func_map.end()) {
      iter->second(x_tensor->data_c(), y_tensor->data_c(), result_datac, data_size);
    } else {
      MS_EXCEPTION(TypeError) << "For '" << primitive->name() << "', 'x' is " << x_tensor->ToString()
                              << ", the type is not supported.";
    }
    return result_tensor;
  }

 private:
  std::map<TypeId, std::function<void(void *x, void *y, void *result, size_t size)>> func_map = {
    {kNumberTypeInt, DivImpl<int>},
    {kNumberTypeInt8, DivImpl<int8_t>},
    {kNumberTypeInt16, DivImpl<int16_t>},
    {kNumberTypeInt32, DivImpl<int32_t>},
    {kNumberTypeInt64, DivImpl<int64_t>},
    {kNumberTypeUInt8, DivImpl<uint8_t>},
    {kNumberTypeUInt16, DivImpl<uint16_t>},
    {kNumberTypeUInt32, DivImpl<uint32_t>},
    {kNumberTypeUInt64, DivImpl<uint64_t>},
    {kNumberTypeFloat16, DivImpl<float16>},
    {kNumberTypeFloat32, DivImpl<float>},
    {kNumberTypeFloat, DivImpl<float>},
    {kNumberTypeFloat64, DivImpl<double>},
    {kNumberTypeDouble, DivImpl<double>},
    {kNumberTypeComplex64, ComplexDivImpl<std::complex<float>>},
    {kNumberTypeComplex128, ComplexDivImpl<std::complex<double>>}};
};
REGISTER_PRIMITIVE_FUNCTION_FRONTEND_FUNC_IMPL("Div", DivFrontendFuncImpl);

}  // namespace ops
}  // namespace mindspore
