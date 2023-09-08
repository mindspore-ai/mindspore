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

namespace mindspore {
// KernelTensorValue stores values in continuous memory and supports values of the types ValueSequence, Tensor, Scalar,
// and String.
class MS_CORE_API KernelTensorValue : public Value {
 public:
  // This constructor is used to construct the KernelTensorValue to hold the Tensor.
  KernelTensorValue(const tensor::TensorDataPtr &tensor_data, const TypePtr &t);

  // This constructor is used to construct the KernelTensorValue to hold the ValueSequence.
  KernelTensorValue(std::vector<uint8_t> &&array_data, const TypePtr &t);

  // This constructor is used to construct the KernelTensorValue to hold the String.
  KernelTensorValue(const StringImmPtr &string, const TypePtr &t);

  // This constructor is used to construct the KernelTensorValue to hold the Scalar.
  template <typename T, typename std::enable_if<std::is_scalar<std::decay_t<T>>::value>::type * = nullptr>
  KernelTensorValue(T scalar, const TypePtr &t) : Value(t) {
    auto scalar_data = std::vector<uint8_t>(sizeof(T));
    *reinterpret_cast<T *>(scalar_data.data()) = scalar;
    data_ = std::move(scalar_data);
    obj_type_id_ = kObjectTypeNumber;
  }

  ~KernelTensorValue() = default;

  KernelTensorValue(const KernelTensorValue &other) = delete;
  KernelTensorValue &operator=(const KernelTensorValue &other) = delete;

  MS_DECLARE_PARENT(KernelTensorValue, Value)

  bool operator==(const Value &other) const override;

  bool operator==(const KernelTensorValue &other) const;

  // Get the address of the value stored in KernelTensorValue.
  const void *GetDataPtr() const;

  // Get the buffer size in bytes of the value stored in KernelTensorValue.
  size_t GetDataSize() const;

 private:
  // Used to stores values in continuous memory for the types ValueSequence, Tensor, Scalar and String.
  std::variant<std::vector<uint8_t>, tensor::TensorDataPtr, StringImmPtr> data_;

  // The object type id for the types ValueSequence, Tensor, Scalar and String.
  TypeId obj_type_id_{TypeId::kTypeUnknown};
};

using KernelTensorValuePtr = std::shared_ptr<KernelTensorValue>;
}  // namespace mindspore

#endif  // MINDSPORE_CORE_IR_KERNEL_TENSOR_VALUE_H_
