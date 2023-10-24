/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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
#include <map>
#include <memory>
#include <complex>
#include "ops/ops_frontend_func_impl.h"
#include "ops/ops_func_impl/reciprocal.h"

namespace mindspore::ops {
BaseShapePtr ReciprocalFuncImpl::InferShape(const PrimitivePtr &primitive,
                                            const std::vector<AbstractBasePtr> &input_args) const {
  auto x_shape = input_args[kIndex0]->GetShape();
  MS_EXCEPTION_IF_NULL(x_shape);
  return x_shape->Clone();
}

TypePtr ReciprocalFuncImpl::InferType(const PrimitivePtr &primitive,
                                      const std::vector<AbstractBasePtr> &input_args) const {
  auto x_type = input_args[kIndex0]->GetType();
  MS_EXCEPTION_IF_NULL(x_type);
  return x_type->Clone();
}

template <typename T>
void ImplReciprocal(void *origin, void *target, size_t size) {
  MS_EXCEPTION_IF_NULL(origin);
  MS_EXCEPTION_IF_NULL(target);
  auto origin_data = reinterpret_cast<T *>(origin);
  auto target_data = reinterpret_cast<T *>(target);
  T numerator = 1;
  for (size_t i = 0; i < size; ++i) {
    target_data[i] = numerator / origin_data[i];
  }
}

void ImplReciprocalFloat16(void *origin, void *target, size_t size) {
  MS_EXCEPTION_IF_NULL(origin);
  MS_EXCEPTION_IF_NULL(target);
  auto origin_data = reinterpret_cast<float16 *>(origin);
  auto target_data = reinterpret_cast<float16 *>(target);
  float16 numerator(1);
  for (size_t i = 0; i < size; ++i) {
    target_data[i] = numerator / origin_data[i];
  }
}

class ReciprocalFrontendFuncImpl : public OpFrontendFuncImpl {
 public:
  ValuePtr InferValue(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    auto x_value = input_args[kIndex0]->GetValue();
    if (x_value->ContainsValueAny()) {
      return nullptr;
    }
    auto x_tensor = x_value->cast<tensor::TensorPtr>();
    if (x_tensor == nullptr) {
      return nullptr;
    }

    auto data_size = x_tensor->DataSize();
    auto dtype = x_tensor->data_type();
    auto shape = input_args[kIndex0]->GetShape()->GetShapeVector();
    auto x_datac = x_tensor->data_c();
    auto result_tensor = std::make_shared<tensor::Tensor>(dtype, shape);
    MS_EXCEPTION_IF_NULL(result_tensor);
    auto result_datac = result_tensor->data_c();

    auto iter = func_map.find(dtype);
    if (iter != func_map.end()) {
      iter->second(x_datac, result_datac, data_size);
    } else {
      MS_EXCEPTION(TypeError) << "For '" << primitive->name() << "', 'x' is " << x_tensor->ToString()
                              << ", the type is not supported.";
    }
    return result_tensor;
  }

 private:
  std::map<TypeId, std::function<void(void *origin, void *target, size_t size)>> func_map = {
    {kNumberTypeBool, ImplReciprocal<bool>},
    {kNumberTypeInt, ImplReciprocal<int>},
    {kNumberTypeInt8, ImplReciprocal<int8_t>},
    {kNumberTypeInt16, ImplReciprocal<int16_t>},
    {kNumberTypeInt32, ImplReciprocal<int32_t>},
    {kNumberTypeInt64, ImplReciprocal<int64_t>},
    {kNumberTypeUInt, ImplReciprocal<u_int>},
    {kNumberTypeUInt8, ImplReciprocal<uint8_t>},
    {kNumberTypeUInt16, ImplReciprocal<uint16_t>},
    {kNumberTypeUInt32, ImplReciprocal<uint32_t>},
    {kNumberTypeUInt64, ImplReciprocal<uint64_t>},
    {kNumberTypeFloat16, ImplReciprocalFloat16},
    {kNumberTypeFloat32, ImplReciprocal<float>},
    {kNumberTypeFloat, ImplReciprocal<float>},
    {kNumberTypeFloat64, ImplReciprocal<double>},
    {kNumberTypeDouble, ImplReciprocal<double>},
    {kNumberTypeComplex64, ImplReciprocal<std::complex<float>>},
    {kNumberTypeComplex128, ImplReciprocal<std::complex<double>>},
    {kNumberTypeComplex, ImplReciprocal<std::complex<double>>}};
};

REGISTER_PRIMITIVE_FUNCTION_FRONTEND_FUNC_IMPL("Reciprocal", ReciprocalFrontendFuncImpl);
}  // namespace mindspore::ops
