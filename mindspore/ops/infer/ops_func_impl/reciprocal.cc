/**
 * Copyright 2021-2025 Huawei Technologies Co., Ltd
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
#include <set>
#include "ops/ops_frontend_func_impl.h"
#include "infer/ops_func_impl/reciprocal.h"
#include "utils/ms_context.h"
#include "ops_utils/op_constants.h"
#include "ir/tensor_new.h"

namespace mindspore::ops {
std::vector<TypeId> ReciprocalFuncImpl::InferType(const PrimitivePtr &primitive,
                                                  const InferInfoPtrList &input_infos) const {
  const auto &input_type_id = input_infos[kInputIndex0]->GetType();
  static const std::vector<TypeId> int_or_bool = {kNumberTypeUInt8,  kNumberTypeUInt16, kNumberTypeUInt32,
                                                  kNumberTypeUInt64, kNumberTypeInt8,   kNumberTypeInt16,
                                                  kNumberTypeInt32,  kNumberTypeInt64,  kNumberTypeBool};
  bool is_int_or_bool = std::any_of(int_or_bool.begin(), int_or_bool.end(),
                                    [&input_type_id](const TypeId &type_id) { return input_type_id == type_id; });
  if (is_int_or_bool) {
    return {kNumberTypeFloat32};
  }
  return {input_type_id};
}

template <typename T>
void ImplReciprocal(const void *origin, void *target, size_t size) {
  MS_EXCEPTION_IF_NULL(origin);
  MS_EXCEPTION_IF_NULL(target);
  auto origin_data = reinterpret_cast<const T *>(origin);
  auto target_data = reinterpret_cast<T *>(target);
  T numerator = 1;
  for (size_t i = 0; i < size; ++i) {
    target_data[i] = numerator / origin_data[i];
  }
}

template <typename T>
void ImplReciprocalInt(const void *origin, void *target, size_t size) {
  MS_EXCEPTION_IF_NULL(origin);
  MS_EXCEPTION_IF_NULL(target);
  auto origin_data = static_cast<const T *>(origin);
  auto target_data = static_cast<float *>(target);
  for (size_t i = 0; i < size; ++i) {
    if (origin_data[i] != 0) {
      target_data[i] = 1.0 / static_cast<float>(origin_data[i]);
    } else {
      if (std::numeric_limits<float>::has_infinity) {
        target_data[i] = std::numeric_limits<float>::infinity();
      } else {
        target_data[i] = static_cast<float>(origin_data[i]);
      }
    }
  }
}

void ImplReciprocalFloat16(const void *origin, void *target, size_t size) {
  MS_EXCEPTION_IF_NULL(origin);
  MS_EXCEPTION_IF_NULL(target);
  auto origin_data = reinterpret_cast<const float16 *>(origin);
  auto target_data = reinterpret_cast<float16 *>(target);
  float16 numerator(1);
  for (size_t i = 0; i < size; ++i) {
    target_data[i] = numerator / origin_data[i];
  }
}

class OPS_API ReciprocalFrontendFuncImpl : public OpFrontendFuncImpl {
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
    auto x_tensor_cpu = x_tensor->cpu();
    auto x_datac = x_tensor_cpu->data_c();

    // For integer and boolean types, the output should be float32
    static const std::set<TypeId> int_or_bool = {kNumberTypeUInt8,  kNumberTypeUInt16, kNumberTypeUInt32,
                                                 kNumberTypeUInt64, kNumberTypeInt8,   kNumberTypeInt16,
                                                 kNumberTypeInt32,  kNumberTypeInt64,  kNumberTypeBool};
    TypeId result_dtype = dtype;
    if (int_or_bool.find(dtype) != int_or_bool.end()) {
      result_dtype = kNumberTypeFloat32;
    }

    auto result_tensor = tensor::from_spec(result_dtype, shape, device::DeviceType::kCPU);
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
  std::map<TypeId, std::function<void(const void *origin, void *target, size_t size)>> func_map = {
    {kNumberTypeBool, ImplReciprocalInt<bool>},
    {kNumberTypeInt, ImplReciprocalInt<int>},
    {kNumberTypeInt8, ImplReciprocalInt<int8_t>},
    {kNumberTypeInt16, ImplReciprocalInt<int16_t>},
    {kNumberTypeInt32, ImplReciprocalInt<int32_t>},
    {kNumberTypeInt64, ImplReciprocalInt<int64_t>},
    {kNumberTypeUInt8, ImplReciprocalInt<uint8_t>},
    {kNumberTypeUInt16, ImplReciprocalInt<uint16_t>},
    {kNumberTypeUInt32, ImplReciprocalInt<uint32_t>},
    {kNumberTypeUInt64, ImplReciprocalInt<uint64_t>},
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
