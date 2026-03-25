/**
 * Copyright 2023-2026 Huawei Technologies Co., Ltd
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
#include <type_traits>
#include <vector>
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
std::map<TypeId, Handler> equal_impl_list = {
  {kNumberTypeBool, EqualImpl<bool>},
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
  {kNumberTypeComplex128, EqualImpl<std::complex<double>>},
};

namespace {
template <typename T>
struct IsComplexType : public std::false_type {};
template <typename T>
struct IsComplexType<std::complex<T>> : public std::true_type {};

// Tag-dispatch overloads to avoid `if constexpr` on IsComplexType, which confuses cppcheck 1.90.
// complex src -> complex dst
template <typename DstT, typename SrcT>
inline bool CastCopyDispatch(const SrcT *src_data, DstT *dst_data, size_t size, std::true_type /* src_complex */,
                             std::true_type /* dst_complex */) {
  using DstReal = typename DstT::value_type;
  for (size_t i = 0; i < size; ++i) {
    dst_data[i] = DstT(static_cast<DstReal>(src_data[i].real()), static_cast<DstReal>(src_data[i].imag()));
  }
  return true;
}
// complex src -> non-complex dst: avoid dropping imaginary part implicitly for constant folding.
template <typename DstT, typename SrcT>
inline bool CastCopyDispatch(const SrcT *, DstT *, size_t, std::true_type /* src_complex */,
                             std::false_type /* dst_complex */) {
  return false;
}
// non-complex src -> complex dst
template <typename DstT, typename SrcT>
inline bool CastCopyDispatch(const SrcT *src_data, DstT *dst_data, size_t size, std::false_type /* src_complex */,
                             std::true_type /* dst_complex */) {
  using DstReal = typename DstT::value_type;
  for (size_t i = 0; i < size; ++i) {
    dst_data[i] = DstT(static_cast<DstReal>(src_data[i]), static_cast<DstReal>(0));
  }
  return true;
}
// non-complex src -> non-complex dst
template <typename DstT, typename SrcT>
inline bool CastCopyDispatch(const SrcT *src_data, DstT *dst_data, size_t size, std::false_type /* src_complex */,
                             std::false_type /* dst_complex */) {
  for (size_t i = 0; i < size; ++i) {
    dst_data[i] = static_cast<DstT>(src_data[i]);
  }
  return true;
}

template <typename DstT, typename SrcT>
inline bool CastCopy(const void *src, void *dst, size_t size) {
  MS_EXCEPTION_IF_NULL(src);
  MS_EXCEPTION_IF_NULL(dst);
  const auto *src_data = static_cast<const SrcT *>(src);
  auto *dst_data = static_cast<DstT *>(dst);
  return CastCopyDispatch(src_data, dst_data, size, IsComplexType<SrcT>{}, IsComplexType<DstT>{});
}

template <typename DstT>
inline bool CastFromTypeId(TypeId src_type, const void *src, void *dst, size_t size) {
  switch (src_type) {
    case kNumberTypeBool:
      return CastCopy<DstT, bool>(src, dst, size);
    case kNumberTypeInt8:
      return CastCopy<DstT, int8_t>(src, dst, size);
    case kNumberTypeInt16:
      return CastCopy<DstT, int16_t>(src, dst, size);
    case kNumberTypeInt32:
      return CastCopy<DstT, int32_t>(src, dst, size);
    case kNumberTypeInt64:
      return CastCopy<DstT, int64_t>(src, dst, size);
    case kNumberTypeUInt8:
      return CastCopy<DstT, uint8_t>(src, dst, size);
    case kNumberTypeUInt16:
      return CastCopy<DstT, uint16_t>(src, dst, size);
    case kNumberTypeUInt32:
      return CastCopy<DstT, uint32_t>(src, dst, size);
    case kNumberTypeUInt64:
      return CastCopy<DstT, uint64_t>(src, dst, size);
    case kNumberTypeFloat16:
      return CastCopy<DstT, float16>(src, dst, size);
    case kNumberTypeBFloat16:
      return CastCopy<DstT, bfloat16>(src, dst, size);
    case kNumberTypeFloat32:
      return CastCopy<DstT, float>(src, dst, size);
    case kNumberTypeFloat64:
      return CastCopy<DstT, double>(src, dst, size);
    case kNumberTypeComplex64:
      return CastCopy<DstT, std::complex<float>>(src, dst, size);
    case kNumberTypeComplex128:
      return CastCopy<DstT, std::complex<double>>(src, dst, size);
    default:
      return false;
  }
}

inline bool CastTensorToType(const tensor::TensorPtr &src_tensor, const tensor::TensorPtr &dst_tensor, size_t size) {
  MS_EXCEPTION_IF_NULL(src_tensor);
  MS_EXCEPTION_IF_NULL(dst_tensor);
  const auto src_type = src_tensor->data_type();
  const auto dst_type = dst_tensor->data_type();
  const void *src = src_tensor->data_c();
  void *dst = dst_tensor->data_c();
  switch (dst_type) {
    case kNumberTypeBool:
      return CastFromTypeId<bool>(src_type, src, dst, size);
    case kNumberTypeInt8:
      return CastFromTypeId<int8_t>(src_type, src, dst, size);
    case kNumberTypeInt16:
      return CastFromTypeId<int16_t>(src_type, src, dst, size);
    case kNumberTypeInt32:
      return CastFromTypeId<int32_t>(src_type, src, dst, size);
    case kNumberTypeInt64:
      return CastFromTypeId<int64_t>(src_type, src, dst, size);
    case kNumberTypeUInt8:
      return CastFromTypeId<uint8_t>(src_type, src, dst, size);
    case kNumberTypeUInt16:
      return CastFromTypeId<uint16_t>(src_type, src, dst, size);
    case kNumberTypeUInt32:
      return CastFromTypeId<uint32_t>(src_type, src, dst, size);
    case kNumberTypeUInt64:
      return CastFromTypeId<uint64_t>(src_type, src, dst, size);
    case kNumberTypeFloat16:
      return CastFromTypeId<float16>(src_type, src, dst, size);
    case kNumberTypeBFloat16:
      return CastFromTypeId<bfloat16>(src_type, src, dst, size);
    case kNumberTypeFloat32:
      return CastFromTypeId<float>(src_type, src, dst, size);
    case kNumberTypeFloat64:
      return CastFromTypeId<double>(src_type, src, dst, size);
    case kNumberTypeComplex64:
      return CastFromTypeId<std::complex<float>>(src_type, src, dst, size);
    case kNumberTypeComplex128:
      return CastFromTypeId<std::complex<double>>(src_type, src, dst, size);
    default:
      return false;
  }
}
}  // namespace

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
    auto type_id1 = x1_tensor->data_type();
    auto type_id2 = x2_tensor->data_type();
    // Equal supports mixed dtypes. For constant folding, promote inputs to a common dtype,
    // then compare values under the promoted dtype.
    // NOTE: if one operand is a scalar tensor (shape=()), treat it like a scalar and
    // use Tensor-Scalar promotion to match runtime behavior.
    auto target_type = kTypeUnknown;
    const bool x1_is_scalar_tensor = x1_shape.empty();
    const bool x2_is_scalar_tensor = x2_shape.empty();
    if (x1_is_scalar_tensor != x2_is_scalar_tensor) {
      const auto tensor_type = x1_is_scalar_tensor ? type_id2 : type_id1;
      const auto scalar_type = x1_is_scalar_tensor ? type_id1 : type_id2;
      target_type = ops::PromoteTensorScalarType(tensor_type, scalar_type);
    } else {
      target_type = PromoteType(type_id1, type_id2, primitive->name());
    }
    if (target_type == kTypeUnknown) {
      return nullptr;
    }
    auto data_size = x1_tensor->DataSize();
    auto result_tensor = tensor::from_spec(kNumberTypeBool, x1_shape, device::DeviceType::kCPU);
    auto iter = equal_impl_list.find(target_type);
    if (iter == equal_impl_list.end()) {
      MS_LOG(DEBUG) << "For '" << primitive->name() << "', 'x1' is " << x1_tensor->ToString()
                    << ", the type is not supported.";
      return nullptr;
    }
    auto x1_tensor_cpu = x1_tensor->cpu();
    auto x2_tensor_cpu = x2_tensor->cpu();

    const void *x1_data = x1_tensor_cpu->data_c();
    const void *x2_data = x2_tensor_cpu->data_c();
    tensor::TensorPtr x1_cast_tensor;
    tensor::TensorPtr x2_cast_tensor;
    if (type_id1 != target_type) {
      x1_cast_tensor = tensor::from_spec(target_type, x1_shape, device::DeviceType::kCPU);
      MS_EXCEPTION_IF_NULL(x1_cast_tensor);
      if (!CastTensorToType(x1_tensor_cpu, x1_cast_tensor, data_size)) {
        return nullptr;
      }
      x1_data = x1_cast_tensor->data_c();
    }
    if (type_id2 != target_type) {
      x2_cast_tensor = tensor::from_spec(target_type, x2_shape, device::DeviceType::kCPU);
      MS_EXCEPTION_IF_NULL(x2_cast_tensor);
      if (!CastTensorToType(x2_tensor_cpu, x2_cast_tensor, data_size)) {
        return nullptr;
      }
      x2_data = x2_cast_tensor->data_c();
    }
    iter->second(x1_data, x2_data, result_tensor->data_c(), data_size);
    return result_tensor;
  }
};

REGISTER_PRIMITIVE_FUNCTION_FRONTEND_FUNC_IMPL("Equal", EqualFrontendFuncImpl);
}  // namespace ops
}  // namespace mindspore
