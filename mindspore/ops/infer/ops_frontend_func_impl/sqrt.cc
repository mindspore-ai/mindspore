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

#include <complex>
#include <map>
#include <memory>
#include "ir/tensor_new.h"
#include "ops/ops_frontend_func_impl.h"
#include "ops_utils/op_utils.h"
#include "ops/ops_func_impl/simple_infer.h"
#include "utils/log_adapter.h"
#include "abstract/abstract_value.h"
#include "ops_utils/op_constants.h"
#include "primitive/auto_generate/gen_ops_primitive_s.h"

namespace mindspore {
namespace ops {
namespace {
template <typename T>
void ImpleSqrt(const void *origin, void *target, size_t size) {
  MS_EXCEPTION_IF_NULL(origin);
  MS_EXCEPTION_IF_NULL(target);
  auto origin_data = static_cast<const T *>(origin);
  auto target_data = static_cast<T *>(target);
  for (size_t i = 0; i < size; ++i) {
    target_data[i] = static_cast<T>(sqrt(static_cast<double>(origin_data[i])));
  }
}

template <typename T>
void ImpleComplexSqrt(const void *origin, void *target, size_t size) {
  MS_EXCEPTION_IF_NULL(origin);
  MS_EXCEPTION_IF_NULL(target);
  auto origin_data = static_cast<const T *>(origin);
  auto target_data = static_cast<T *>(target);
  for (size_t i = 0; i < size; ++i) {
    target_data[i] = static_cast<T>(sqrt(origin_data[i]));
  }
}

template <typename T>
void ImpleSqrtInteger(const void *origin, void *target, size_t size) {
  MS_EXCEPTION_IF_NULL(origin);
  MS_EXCEPTION_IF_NULL(target);
  auto origin_data = static_cast<const T *>(origin);
  auto target_data = static_cast<float *>(target);
  for (size_t i = 0; i < size; ++i) {
    target_data[i] = static_cast<float>(sqrt(static_cast<double>(origin_data[i])));
  }
}

TypeId GetOutputTypeId(const TypeId &input_type_id) {
  static std::set<TypeId> intergral_set = {kNumberTypeBool,  kNumberTypeUInt8, kNumberTypeInt8,
                                           kNumberTypeInt16, kNumberTypeInt32, kNumberTypeInt64};
  if (intergral_set.find(input_type_id) != intergral_set.end()) {
    return kNumberTypeFloat32;
  }
  return input_type_id;
}
}  // namespace

using SqrtHandler = std::function<void(const void *origin, void *target, size_t size)>;
std::map<TypeId, SqrtHandler> sqrt_impl_list = {{kNumberTypeBool, ImpleSqrtInteger<bool>},
                                                {kNumberTypeInt8, ImpleSqrtInteger<int8_t>},
                                                {kNumberTypeInt16, ImpleSqrtInteger<int16_t>},
                                                {kNumberTypeInt32, ImpleSqrtInteger<int32_t>},
                                                {kNumberTypeInt64, ImpleSqrtInteger<int64_t>},
                                                {kNumberTypeUInt8, ImpleSqrtInteger<uint8_t>},
                                                {kNumberTypeUInt16, ImpleSqrtInteger<uint16_t>},
                                                {kNumberTypeUInt32, ImpleSqrtInteger<uint32_t>},
                                                {kNumberTypeUInt64, ImpleSqrtInteger<uint64_t>},
                                                {kNumberTypeFloat16, ImpleSqrt<float16>},
                                                {kNumberTypeBFloat16, ImpleSqrt<bfloat16>},
                                                {kNumberTypeFloat32, ImpleSqrt<float>},
                                                {kNumberTypeFloat64, ImpleSqrt<double>},
                                                {kNumberTypeComplex64, ImpleComplexSqrt<std::complex<float>>},
                                                {kNumberTypeComplex128, ImpleComplexSqrt<std::complex<double>>}};

class OPS_API SqrtFrontendFuncImpl : public OpFrontendFuncImpl {
 public:
  // Do not override this interface if the op has no InferValue
  ValuePtr InferValue(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    auto x_value = input_args[kIndex0]->GetValue();
    if (x_value == nullptr || x_value->isa<ValueAny>()) {
      return nullptr;
    }
    auto x_tensor = x_value->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(x_tensor);

    auto x_shape = input_args[kIndex0]->GetShape()->GetShapeVector();
    if (IsDynamic(x_shape)) {
      return nullptr;
    }
    auto type_id = x_tensor->data_type();
    auto data_size = x_tensor->DataSize();
    auto result_tensor =
      tensor::from_spec(GetOutputTypeId(type_id), x_shape, device::DeviceType::kCPU);  // same shape and dtype
    MS_EXCEPTION_IF_NULL(result_tensor);
    auto iter = sqrt_impl_list.find(type_id);
    if (iter == sqrt_impl_list.end()) {
      MS_LOG(DEBUG) << "For '" << primitive->name() << "', 'x_value' is " << x_tensor->ToString()
                    << ", the type is not supported.";
      return nullptr;
    }
    auto x_tensor_cpu = x_tensor->cpu();
    iter->second(x_tensor_cpu->data_c(), result_tensor->data_c(), data_size);
    return result_tensor;
  }
};

REGISTER_PRIMITIVE_FUNCTION_FRONTEND_FUNC_IMPL(kNameSqrt, SqrtFrontendFuncImpl);
}  // namespace ops
}  // namespace mindspore
