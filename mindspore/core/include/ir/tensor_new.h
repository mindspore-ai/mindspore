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

#ifndef MINDSPORE_MINDSPORE_CORE_INCLUDE_IR_TENSOR_NEW_H_
#define MINDSPORE_MINDSPORE_CORE_INCLUDE_IR_TENSOR_NEW_H_
#include <memory>
#include <vector>
#include "device_address/device_type.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/base/type_id.h"
#include "mindapi/base/macros.h"
#include "ir/tensor.h"
#include "device_address/device_address_maker.h"

namespace mindspore {
namespace tensor {
template <typename T>
struct TensorTypeIdTrait {
  static TypeId type_id() {
    MS_LOG(EXCEPTION) << "Not support type " << typeid(T).name();
    return kTypeUnknown;
  }
};

#define TENSOR_TYPE_ID_TRAIT(prototype, type_id_value)          \
  template <>                                                   \
  struct TensorTypeIdTrait<prototype> {                         \
    static constexpr TypeId type_id() { return type_id_value; } \
  }

TENSOR_TYPE_ID_TRAIT(int64_t, kNumberTypeInt64);
TENSOR_TYPE_ID_TRAIT(int32_t, kNumberTypeInt32);
TENSOR_TYPE_ID_TRAIT(int16_t, kNumberTypeInt16);
TENSOR_TYPE_ID_TRAIT(int8_t, kNumberTypeInt8);
TENSOR_TYPE_ID_TRAIT(double, kNumberTypeFloat32);
TENSOR_TYPE_ID_TRAIT(float, kNumberTypeFloat32);
TENSOR_TYPE_ID_TRAIT(float16, kNumberTypeFloat16);
TENSOR_TYPE_ID_TRAIT(float8_e5m2, kNumberTypeFloat8E5M2);
TENSOR_TYPE_ID_TRAIT(float8_e4m3fn, kNumberTypeFloat8E4M3FN);
TENSOR_TYPE_ID_TRAIT(hifloat8, kNumberTypeHiFloat8);
TENSOR_TYPE_ID_TRAIT(float8_e8m0fnu, kNumberTypeFloat8E8M0FNU);
TENSOR_TYPE_ID_TRAIT(bfloat16, kNumberTypeBFloat16);
TENSOR_TYPE_ID_TRAIT(uint64_t, kNumberTypeUInt64);
TENSOR_TYPE_ID_TRAIT(uint32_t, kNumberTypeUInt32);
TENSOR_TYPE_ID_TRAIT(uint16_t, kNumberTypeUInt16);
TENSOR_TYPE_ID_TRAIT(uint8_t, kNumberTypeUInt8);
TENSOR_TYPE_ID_TRAIT(bool, kNumberTypeBool);
TENSOR_TYPE_ID_TRAIT(std::complex<float>, kNumberTypeComplex64);
TENSOR_TYPE_ID_TRAIT(std::complex<double>, kNumberTypeComplex128);
#undef TENSOR_TYPE_ID_TRAIT

template <typename T>
TypeId GetTypeId() {
  return TensorTypeIdTrait<T>::type_id();
}

MS_CORE_API TypeId TypeIdOf(const TypePtr &data_type, TypeId defaultTypeId);

/// \brief Create a tensor with data type and shape.
/// \param[in] data_type [TypeId] Data type of the tensor.
/// \param[in] shape The shape represented by ShapeVector of the tensor.
/// \param[in] device_type The device type of the Tensor.
/// \return [TensorPtr]
MS_CORE_API TensorPtr from_spec(TypeId data_type, const ShapeVector &shape, device::DeviceType device_type);

/// \brief Create a tensor with data type and shape more efficient for CPU.
///        Allocate memory without initializing it.
/// \param[in] data_type [TypeId] Data type of the tensor.
/// \param[in] shape The shape represented by ShapeVector of the tensor.
/// \param[in] device_type The device type of the Tensor.
/// \return [TensorPtr]
MS_CORE_API TensorPtr from_spec_fast(TypeId data_type, const ShapeVector &shape, device::DeviceType device_type);

/// \brief Create a tensor from a scalar.
/// \param[in] input [T] Scalar to create Tensor.
/// \param[in] data_type The data type of scalar.
/// \return [TensorPtr]
template <typename T>
TensorPtr from_scalar(T input, const TypePtr &data_type = nullptr) {
  auto type = TypeIdOf(data_type, GetTypeId<T>());
  static ShapeVector shape = {};
  return std::make_shared<Tensor>(type, shape, MakeDeviceAddress(type, input));
}

/// \brief Create a tensor from a vector.
/// \param[in] input [std::vector<T>] Vector to create Tensor.
/// \param[in] data_type The data type of the vector.
/// \return [TensorPtr]
template <typename T>
TensorPtr from_vector(const std::vector<T> &input, const TypePtr &data_type = nullptr) {
  auto type = TypeIdOf(data_type, GetTypeId<T>());
  ShapeVector shape = {static_cast<ShapeValueDType>(input.size())};
  return std::make_shared<Tensor>(type, shape, MakeDeviceAddress(type, shape, input));
}

/// \brief Create a tensor with input data buffer.
/// \param[in] data_type [TypeId] Data type of the tensor.
/// \param[in] shape The shape represented by ShapeVector of the tensor.
/// \param[in] data The input data to be copied into tensor.
/// \param[in] data_len The length of data in bytes.
/// \return [TensorPtr]
MS_CORE_API TensorPtr from_buffer(TypeId data_type, const ShapeVector &shape, void *data, size_t data_len);

/// \brief Create a tensor with input data buffer and given source data type.
/// \param[in] data_type [TypeId] Data type of the tensor.
/// \param[in] shape The shape represented by ShapeVector of the tensor.
/// \param[in] data The input data to be copied into tensor.
/// \param[in] src_data_type The source data type.
/// \return [TensorPtr]
MS_CORE_API TensorPtr from_buffer(TypeId data_type, const ShapeVector &shape, void *data, TypeId src_data_type);
}  // namespace tensor
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CORE_INCLUDE_IR_TENSOR_NEW_H_
