/**
 * Copyright 2025-2026 Huawei Technologies Co., Ltd
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
#include <functional>
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
struct IsStdComplex : public std::false_type {};
template <typename T>
struct IsStdComplex<std::complex<T>> : public std::true_type {};

template <typename CommonT, typename SrcT>
void EqScalarHelp(const void *x1, double x2, void *result, size_t size) {
  MS_EXCEPTION_IF_NULL(x1);
  MS_EXCEPTION_IF_NULL(result);
  const auto *x1_data = static_cast<const SrcT *>(x1);
  auto *result_data = static_cast<bool *>(result);
  if constexpr (std::is_same_v<CommonT, bool>) {
    const bool rhs = static_cast<bool>(x2);
    for (size_t i = 0; i < size; ++i) {
      result_data[i] = static_cast<bool>(x1_data[i]) == rhs;
    }
  } else if constexpr (std::is_same_v<CommonT, std::complex<float>>) {
    const auto rhs = std::complex<float>(static_cast<float>(x2), 0.0f);
    for (size_t i = 0; i < size; ++i) {
      using RealT = typename CommonT::value_type;
      if constexpr (IsStdComplex<SrcT>::value) {
        const auto lhs = CommonT(static_cast<RealT>(x1_data[i].real()), static_cast<RealT>(x1_data[i].imag()));
        result_data[i] = lhs == rhs;
      } else {
        const auto lhs = CommonT(static_cast<RealT>(x1_data[i]), static_cast<RealT>(0));
        result_data[i] = lhs == rhs;
      }
    }
  } else if constexpr (std::is_same_v<CommonT, std::complex<double>>) {
    const auto rhs = std::complex<double>(static_cast<double>(x2), 0.0);
    for (size_t i = 0; i < size; ++i) {
      using RealT = typename CommonT::value_type;
      if constexpr (IsStdComplex<SrcT>::value) {
        const auto lhs = CommonT(static_cast<RealT>(x1_data[i].real()), static_cast<RealT>(x1_data[i].imag()));
        result_data[i] = lhs == rhs;
      } else {
        const auto lhs = CommonT(static_cast<RealT>(x1_data[i]), static_cast<RealT>(0));
        result_data[i] = lhs == rhs;
      }
    }
  } else {
    const CommonT rhs = static_cast<CommonT>(x2);
    for (size_t i = 0; i < size; ++i) {
      result_data[i] = static_cast<CommonT>(x1_data[i]) == rhs;
    }
  }
}

template <typename CommonT>
void ImplEqScalar(const void *x1, TypeId x1_type, double x2, void *result, size_t size) {
  switch (x1_type) {
    case kNumberTypeBool:
      EqScalarHelp<CommonT, bool>(x1, x2, result, size);
      break;
    case kNumberTypeInt8:
      EqScalarHelp<CommonT, int8_t>(x1, x2, result, size);
      break;
    case kNumberTypeInt16:
      EqScalarHelp<CommonT, int16_t>(x1, x2, result, size);
      break;
    case kNumberTypeInt32:
      EqScalarHelp<CommonT, int32_t>(x1, x2, result, size);
      break;
    case kNumberTypeInt64:
      EqScalarHelp<CommonT, int64_t>(x1, x2, result, size);
      break;
    case kNumberTypeUInt8:
      EqScalarHelp<CommonT, uint8_t>(x1, x2, result, size);
      break;
    case kNumberTypeUInt16:
      EqScalarHelp<CommonT, uint16_t>(x1, x2, result, size);
      break;
    case kNumberTypeUInt32:
      EqScalarHelp<CommonT, uint32_t>(x1, x2, result, size);
      break;
    case kNumberTypeUInt64:
      EqScalarHelp<CommonT, uint64_t>(x1, x2, result, size);
      break;
    case kNumberTypeFloat16:
      EqScalarHelp<CommonT, float16>(x1, x2, result, size);
      break;
    case kNumberTypeBFloat16:
      EqScalarHelp<CommonT, bfloat16>(x1, x2, result, size);
      break;
    case kNumberTypeFloat32:
      EqScalarHelp<CommonT, float>(x1, x2, result, size);
      break;
    case kNumberTypeFloat64:
      EqScalarHelp<CommonT, double>(x1, x2, result, size);
      break;
    case kNumberTypeComplex64:
      if constexpr (IsStdComplex<CommonT>::value) {
        EqScalarHelp<CommonT, std::complex<float>>(x1, x2, result, size);
      }
      break;
    case kNumberTypeComplex128:
      if constexpr (IsStdComplex<CommonT>::value) {
        EqScalarHelp<CommonT, std::complex<double>>(x1, x2, result, size);
      }
      break;
    default:
      break;
  }
}

using HandlerEqScalar = std::function<void(const void *x1, TypeId x1_type, double x2, void *result, size_t size)>;
std::map<TypeId, HandlerEqScalar> eq_scalar_impl_list = {
  {kNumberTypeBool, ImplEqScalar<bool>},
  {kNumberTypeInt8, ImplEqScalar<int8_t>},
  {kNumberTypeInt16, ImplEqScalar<int16_t>},
  {kNumberTypeInt32, ImplEqScalar<int32_t>},
  {kNumberTypeInt64, ImplEqScalar<int64_t>},
  {kNumberTypeUInt8, ImplEqScalar<uint8_t>},
  {kNumberTypeUInt16, ImplEqScalar<uint16_t>},
  {kNumberTypeUInt32, ImplEqScalar<uint32_t>},
  {kNumberTypeUInt64, ImplEqScalar<uint64_t>},
  {kNumberTypeFloat16, ImplEqScalar<float16>},
  {kNumberTypeBFloat16, ImplEqScalar<bfloat16>},
  {kNumberTypeFloat32, ImplEqScalar<float>},
  {kNumberTypeFloat64, ImplEqScalar<double>},
  {kNumberTypeComplex64, ImplEqScalar<std::complex<float>>},
  {kNumberTypeComplex128, ImplEqScalar<std::complex<double>>},
};
}  // namespace

class EqScalarFrontendFuncImpl : public OpFrontendFuncImpl {
 public:
  ValuePtr InferValue(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    auto x1 = input_args[kIndex0]->GetValue();
    auto x2 = input_args[kIndex1]->GetValue();
    if (x1 == nullptr || x2 == nullptr || x1->isa<ValueAny>() || x2->isa<ValueAny>()) {
      return nullptr;
    }

    auto x1_tensor = x1->cast<tensor::TensorPtr>();
    // EqScalar expects the second input to be a scalar number.
    if (x1_tensor == nullptr || x2->isa<tensor::Tensor>()) {
      return nullptr;
    }
    MS_EXCEPTION_IF_NULL(x1_tensor);

    auto x1_shape = input_args[kIndex0]->GetShape()->GetShapeVector();
    if (IsDynamic(x1_shape)) {
      return nullptr;
    }

    auto data_size = x1_tensor->DataSize();
    auto x1_type = x1_tensor->data_type();
    auto x2_type = input_args[kIndex1]->GetType()->type_id();
    TypeId common_type = PromoteTensorScalarType(x1_type, x2_type);

    auto iter = eq_scalar_impl_list.find(common_type);
    if (iter == eq_scalar_impl_list.end()) {
      return nullptr;
    }

    auto x2_number = GetScalarCastValue<double>(primitive->name(), x2);
    auto result_tensor = tensor::from_spec(kNumberTypeBool, x1_shape, device::DeviceType::kCPU);
    MS_EXCEPTION_IF_NULL(result_tensor);

    auto x1_tensor_cpu = x1_tensor->cpu();
    iter->second(x1_tensor_cpu->data_c(), x1_type, x2_number, result_tensor->data_c(), data_size);
    return result_tensor;
  }
};

REGISTER_PRIMITIVE_FUNCTION_FRONTEND_FUNC_IMPL("EqScalar", EqScalarFrontendFuncImpl);
}  // namespace ops
}  // namespace mindspore
