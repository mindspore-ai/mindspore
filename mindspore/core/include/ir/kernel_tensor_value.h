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

#ifndef MINDSPORE_CORE_IR_KERNEL_TENSOR_VALUE_H_
#define MINDSPORE_CORE_IR_KERNEL_TENSOR_VALUE_H_

#include <variant>
#include <utility>
#include <vector>
#include <string>
#include <memory>
#include "ir/value.h"
#include "ir/tensor.h"
#include "device_address/device_address.h"

namespace mindspore {
using DeviceAddress = device::DeviceAddress;
using DeviceAddressPtr = device::DeviceAddressPtr;
// KernelTensorValue stores values in continuous memory and supports values of the types ValueSequence, Tensor, Scalar,
// and String.
class MS_CORE_API KernelTensorValue : public Value {
 public:
  // This constructor is used to construct the KernelTensorValue to hold the Tensor.
  KernelTensorValue(const DeviceAddressPtr &device_sync, const TypePtr &t);

  // This constructor is used to construct the KernelTensorValue to hold the ValueSequence.
  KernelTensorValue(std::vector<uint8_t> &&array_data, const TypePtr &t);

  // This constructor is used to construct the KernelTensorValue to hold the String.
  KernelTensorValue(const StringImmPtr &string, const TypePtr &t);

  // This constructor is used to construct the KernelTensorValue to hold the Scalar.
  template <typename T, typename std::enable_if<std::is_scalar<std::decay_t<T>>::value>::type * = nullptr>
  KernelTensorValue(T scalar, const TypePtr &t) : Value(t) {
    auto scalar_data = std::vector<uint8_t>(sizeof(T));
    *reinterpret_cast<T *>(scalar_data.data()) = scalar;
    const_data_ = std::move(scalar_data);
    obj_type_id_ = kObjectTypeNumber;
  }

  KernelTensorValue(float16 scalar, const TypePtr &t) : Value(t) {
    auto scalar_data = std::vector<uint8_t>(sizeof(float16));
    auto err = memcpy_s(scalar_data.data(), scalar_data.size(), &scalar, sizeof(float16));
    if (err != EOK) {
      MS_LOG(EXCEPTION) << "memcpy_s failed for type float16, error code: " << err;
    }
    const_data_ = std::move(scalar_data);
    obj_type_id_ = kObjectTypeNumber;
  }

  KernelTensorValue(bfloat16 scalar, const TypePtr &t) : Value(t) {
    auto scalar_data = std::vector<uint8_t>(sizeof(bfloat16));
    auto err = memcpy_s(scalar_data.data(), scalar_data.size(), &scalar, sizeof(bfloat16));
    if (err != EOK) {
      MS_LOG(EXCEPTION) << "memcpy_s failed for type bfloat16, error code: " << err;
    }
    const_data_ = std::move(scalar_data);
    obj_type_id_ = kObjectTypeNumber;
  }

  template <typename T, typename std::enable_if<std::is_same<T, std::complex<float>>::value ||
                                                std::is_same<T, std::complex<double>>::value>::type * = nullptr>
  KernelTensorValue(T scalar, const TypePtr &t) : Value(t) {
    auto scalar_data = std::vector<uint8_t>(sizeof(T));
    auto err = memcpy_s(scalar_data.data(), scalar_data.size(), &scalar, sizeof(T));
    if (err != EOK) {
      MS_LOG(EXCEPTION) << "memcpy_s failed for type complex, error code: " << err;
    }
    const_data_ = std::move(scalar_data);
    obj_type_id_ = kObjectTypeNumber;
  }

  // This constructor is used to construct the mutable KernelTensorValue to hold any data type (such as Tensor,
  // ValueSequence, String, Scalar) use continuous memory. This constructor only malloc raw memory to prepare store
  // value data.
  KernelTensorValue(size_t size, const TypePtr &t);

  // This constructor is used to construct the mutable KernelTensorValue to hold any data type (such as Tensor,
  // ValueSequence, String, Scalar) use external const continuous memory(const void *).
  KernelTensorValue(const void *data, size_t size, const TypePtr &t);

  ~KernelTensorValue() = default;

  MS_DECLARE_PARENT(KernelTensorValue, Value)

  bool operator==(const Value &other) const override;

  bool operator==(const KernelTensorValue &other) const;

  // Get the const address of the value stored in KernelTensorValue.
  const void *GetDataPtr() const;

  // Get the mutable address of the value stored in KernelTensorValue.
  void *GetMutableDataPtr();

  // Set external data to KernelTensorValue.
  // Note: The caller needs to manage the life cycle of external values. This interface is used with the
  // KernelTensorValue(const void *data, size_t size, const TypePtr &t) constructor. There is no need to copy external
  // values to KernelTensorValue to improve performance.
  void SetDataPtr(const void *data_ptr);

  // Get the buffer size in bytes of the value stored in KernelTensorValue.
  size_t GetDataSize() const;

  // Change the continuous memory size.
  void Resize(size_t size);

 private:
  // Used to store values in continuous memory for the types ValueSequence, Tensor, Scalar and String.
  // The const_data_ is read-only after assignment.
  std::variant<std::vector<uint8_t>, DeviceAddressPtr, StringImmPtr> const_data_;

  // The object type id for the types ValueSequence, Tensor, Scalar and String.
  TypeId obj_type_id_{TypeId::kTypeUnknown};

  // Used to store values in continuous memory for the types ValueSequence, Tensor, Scalar and String.
  // The mutable_data_ that uses shared pointer one can be read, written and changed size after assignment.
  std::variant<std::shared_ptr<uint8_t[]>, const void *> mutable_data_;

  // The value size in bytes.
  size_t size_{0};

  // Whether use mutable_data_ to store any type value.
  bool use_mutable_storage_{false};
};

using KernelTensorValuePtr = std::shared_ptr<KernelTensorValue>;
}  // namespace mindspore

#endif  // MINDSPORE_CORE_IR_KERNEL_TENSOR_VALUE_H_
