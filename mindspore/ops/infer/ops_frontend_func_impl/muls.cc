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
#include <vector>
#include "ir/tensor_new.h"
#include "ops/ops_frontend_func_impl.h"
#include "ops_utils/op_utils.h"
#include "utils/log_adapter.h"
#include "abstract/abstract_value.h"
#include "ops_utils/op_constants.h"

namespace mindspore {
namespace ops {
template <typename T, typename U>
void ImplMulsHelp(const void *x1, double x2, void *result, size_t size) {
  MS_EXCEPTION_IF_NULL(x1);
  MS_EXCEPTION_IF_NULL(result);
  const U *x1_data = static_cast<const U *>(x1);
  T x2_data = static_cast<T>(x2);
  auto result_data = static_cast<T *>(result);
  for (size_t i = 0; i < size; ++i) {
    T x1_ele = static_cast<T>(x1_data[i]);
    result_data[i] = x1_ele * x2_data;
  }
}

template <typename T>
void ImplMulsBoolHelp(const void *x1, double x2, void *result, size_t size) {
  MS_EXCEPTION_IF_NULL(x1);
  MS_EXCEPTION_IF_NULL(result);
  const T *x1_data = static_cast<const T *>(x1);
  T x2_data = static_cast<T>(x2);
  auto result_data = static_cast<T *>(result);
  for (size_t i = 0; i < size; ++i) {
    result_data[i] = x1_data[i] && x2_data;
  }
}

template <typename T>
void ImplMuls(const void *x1, TypeId x1_type, double x2, void *result, size_t size) {
  switch (x1_type) {
    case kNumberTypeBool:
      ImplMulsHelp<T, bool>(x1, x2, result, size);
      break;
    case kNumberTypeInt8:
      ImplMulsHelp<T, int8_t>(x1, x2, result, size);
      break;
    case kNumberTypeInt16:
      ImplMulsHelp<T, int16_t>(x1, x2, result, size);
      break;
    case kNumberTypeInt32:
      ImplMulsHelp<T, int32_t>(x1, x2, result, size);
      break;
    case kNumberTypeInt64:
      ImplMulsHelp<T, int64_t>(x1, x2, result, size);
      break;
    case kNumberTypeUInt8:
      ImplMulsHelp<T, uint8_t>(x1, x2, result, size);
      break;
    case kNumberTypeUInt16:
      ImplMulsHelp<T, uint16_t>(x1, x2, result, size);
      break;
    case kNumberTypeUInt32:
      ImplMulsHelp<T, uint32_t>(x1, x2, result, size);
      break;
    case kNumberTypeUInt64:
      ImplMulsHelp<T, uint64_t>(x1, x2, result, size);
      break;
    case kNumberTypeFloat16:
      ImplMulsHelp<T, float16>(x1, x2, result, size);
      break;
    case kNumberTypeBFloat16:
      ImplMulsHelp<T, bfloat16>(x1, x2, result, size);
      break;
    case kNumberTypeFloat32:
      ImplMulsHelp<T, float>(x1, x2, result, size);
      break;
    case kNumberTypeFloat64:
      ImplMulsHelp<T, double>(x1, x2, result, size);
      break;
    default:
      break;
  }
}

template <typename T>
void ImplMulsComplex(const void *x1, TypeId x1_type, double x2, void *result, size_t size) {
  switch (x1_type) {
    case kNumberTypeComplex64:
      ImplMulsHelp<T, std::complex<float>>(x1, x2, result, size);
      break;
    case kNumberTypeComplex128:
      ImplMulsHelp<T, std::complex<double>>(x1, x2, result, size);
      break;
    default:
      break;
  }
}

template <typename T>
void ImplMulsBool(const void *x1, TypeId x1_type, double x2, void *result, size_t size) {
  switch (x1_type) {
    case kNumberTypeBool:
      ImplMulsBoolHelp<bool>(x1, x2, result, size);
      break;
    default:
      break;
  }
}

using HandlerMuls = std::function<void(const void *x1, TypeId x1_type, double x2, void *result, size_t size)>;
std::map<TypeId, HandlerMuls> muls_impl_list = {{kNumberTypeBool, ImplMulsBool<bool>},
                                                {kNumberTypeInt8, ImplMuls<int8_t>},
                                                {kNumberTypeInt16, ImplMuls<int16_t>},
                                                {kNumberTypeInt32, ImplMuls<int32_t>},
                                                {kNumberTypeInt64, ImplMuls<int64_t>},
                                                {kNumberTypeUInt8, ImplMuls<uint8_t>},
                                                {kNumberTypeUInt16, ImplMuls<uint16_t>},
                                                {kNumberTypeUInt32, ImplMuls<uint32_t>},
                                                {kNumberTypeUInt64, ImplMuls<uint64_t>},
                                                {kNumberTypeFloat16, ImplMuls<float16>},
                                                {kNumberTypeBFloat16, ImplMuls<bfloat16>},
                                                {kNumberTypeFloat32, ImplMuls<float>},
                                                {kNumberTypeFloat64, ImplMuls<double>},
                                                {kNumberTypeComplex64, ImplMulsComplex<std::complex<float>>},
                                                {kNumberTypeComplex128, ImplMulsComplex<std::complex<double>>}};

class MulsFrontendFuncImpl : public OpFrontendFuncImpl {
 public:
  ValuePtr InferValue(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    auto x1 = input_args[kIndex0]->GetValue();
    auto x2 = input_args[kIndex1]->GetValue();
    if (x1 == nullptr || x2 == nullptr || x1->isa<ValueAny>() || x2->isa<ValueAny>()) {
      return nullptr;
    }

    auto x1_tensor = x1->cast<tensor::TensorPtr>();
    auto x2_number = GetScalarCastValue<double>("muls", x2);
    MS_EXCEPTION_IF_NULL(x1_tensor);

    auto x1_shape = input_args[kIndex0]->GetShape()->GetShapeVector();
    if (IsDynamic(x1_shape)) {
      return nullptr;
    }

    auto data_size = x1_tensor->DataSize();
    auto x1_type = x1_tensor->data_type();
    auto x2_type = input_args[kIndex1]->GetType()->type_id();
    // Keep the type promotion rule consistent with infer stage (MulsFuncImpl::InferType).
    // NOTE: PromoteTensorScalarType only promotes by category level (bool < int < float < complex),
    // and keeps tensor dtype when both are in the same category (e.g. float32 tensor * float64 scalar -> float32).
    TypeId common_type = PromoteTensorScalarType(x1_type, x2_type);
    MS_LOG(DEBUG) << "For [" << primitive->name() << "], first input type: " << input_args[kIndex0]->GetType()
                  << ", typeid: " << input_args[kIndex0]->GetType()->type_id();
    MS_LOG(DEBUG) << "For [" << primitive->name() << "], second input type: " << input_args[kIndex1]->GetType()
                  << ", typeid: " << input_args[kIndex1]->GetType()->type_id();
    MS_LOG(DEBUG) << "For [" << primitive->name() << "], after promote type: " << common_type;

    auto result_tensor = tensor::from_spec(common_type, x1_shape, device::DeviceType::kCPU);
    MS_EXCEPTION_IF_NULL(result_tensor);
    auto iter = muls_impl_list.find(common_type);
    if (iter == muls_impl_list.end()) {
      MS_LOG(DEBUG) << "For '" << primitive->name() << "', 'x1' is " << x1_tensor->ToString() << ", and 'x2' is "
                    << x2_number << ", the type is not supported.";
      return nullptr;
    }
    auto x1_tensor_cpu = x1_tensor->cpu();
    iter->second(x1_tensor_cpu->data_c(), x1_type, x2_number, result_tensor->data_c(), data_size);
    return result_tensor;
  }
};

REGISTER_PRIMITIVE_FUNCTION_FRONTEND_FUNC_IMPL("Muls", MulsFrontendFuncImpl);
}  // namespace ops
}  // namespace mindspore
