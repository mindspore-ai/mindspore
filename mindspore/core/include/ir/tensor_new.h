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
TypeId GetTypeId() {
  if constexpr (std::is_same_v<T, int64_t>) {
    return kNumberTypeInt64;
  } else if constexpr (std::is_same_v<T, int32_t>) {
    return kNumberTypeInt32;
  } else if constexpr (std::is_same_v<T, int16_t>) {
    return kNumberTypeInt16;
  } else if constexpr (std::is_same_v<T, int8_t>) {
    return kNumberTypeInt8;
  } else if constexpr (std::is_same_v<T, double> || std::is_same_v<T, float>) {
    return kNumberTypeFloat32;
  } else if constexpr (std::is_same_v<T, float16>) {
    return kNumberTypeFloat16;
  } else if constexpr (std::is_same_v<T, float8_e5m2>) {
    return kNumberTypeFloat8E5M2;
  } else if constexpr (std::is_same_v<T, float8_e4m3fn>) {
    return kNumberTypeFloat8E4M3FN;
  } else if constexpr (std::is_same_v<T, hifloat8>) {
    return kNumberTypeHiFloat8;
  } else if constexpr (std::is_same_v<T, bfloat16>) {
    return kNumberTypeBFloat16;
  } else if constexpr (std::is_same_v<T, uint64_t>) {
    return kNumberTypeUInt64;
  } else if constexpr (std::is_same_v<T, uint32_t>) {
    return kNumberTypeUInt32;
  } else if constexpr (std::is_same_v<T, uint16_t>) {
    return kNumberTypeUInt16;
  } else if constexpr (std::is_same_v<T, uint8_t>) {
    return kNumberTypeUInt8;
  } else if constexpr (std::is_same_v<T, bool>) {
    return kNumberTypeBool;
  } else if constexpr (std::is_same_v<T, std::complex<float>>) {
    return kNumberTypeComplex64;
  } else if constexpr (std::is_same_v<T, std::complex<double>>) {
    return kNumberTypeComplex128;
  } else {
    MS_LOG(EXCEPTION) << "Not support type " << typeid(T).name();
  }
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
