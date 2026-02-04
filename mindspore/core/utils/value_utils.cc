/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include "utils/value_utils.h"

#include <vector>
#include <complex>
#include <set>
#include <utility>
#include <string>
#include <optional>
#include <type_traits>

#include "ir/anf.h"
#include "ir/value.h"
#include "ir/kernel_tensor_value.h"
#include "abstract/abstract_value.h"
#include "mindapi/base/macros.h"
#include "base/bfloat16.h"
#include "base/float16.h"

namespace mindspore {
template <typename T>
T GetScalarCastValue(const std::string &op_name, const ValuePtr &elem) {
  T res;
  MS_EXCEPTION_IF_NULL(elem);
  if (elem->isa<Int64Imm>()) {
    auto elem_value = GetValue<int64_t>(elem);
    res = static_cast<T>(elem_value);
  } else if (elem->isa<Int32Imm>()) {
    auto elem_value = GetValue<int32_t>(elem);
    res = static_cast<T>(elem_value);
  } else if (elem->isa<Int16Imm>()) {
    auto elem_value = GetValue<int16_t>(elem);
    res = static_cast<T>(elem_value);
  } else if (elem->isa<Int8Imm>()) {
    auto elem_value = GetValue<int8_t>(elem);
    res = static_cast<T>(elem_value);
  } else if (elem->isa<UInt64Imm>()) {
    auto elem_value = GetValue<uint64_t>(elem);
    res = static_cast<T>(elem_value);
  } else if (elem->isa<UInt32Imm>()) {
    auto elem_value = GetValue<uint32_t>(elem);
    res = static_cast<T>(elem_value);
  } else if (elem->isa<UInt16Imm>()) {
    auto elem_value = GetValue<uint16_t>(elem);
    res = static_cast<T>(elem_value);
  } else if (elem->isa<UInt8Imm>()) {
    auto elem_value = GetValue<uint8_t>(elem);
    res = static_cast<T>(elem_value);
  } else if (elem->isa<FP64Imm>()) {
    auto elem_value = GetValue<double>(elem);
    res = static_cast<T>(elem_value);
  } else if (elem->isa<FP32Imm>()) {
    auto elem_value = GetValue<float>(elem);
    res = static_cast<T>(elem_value);
  } else if (elem->isa<BoolImm>()) {
    auto elem_value = GetValue<bool>(elem);
    res = static_cast<T>(elem_value);
  } else if (elem->isa<FP16Imm>()) {
    auto elem_value = GetValue<float16>(elem);
    res = static_cast<T>(elem_value);
  } else if (elem->isa<BF16Imm>()) {
    auto elem_value = GetValue<bfloat16>(elem);
    res = static_cast<T>(elem_value);
  } else {
    MS_EXCEPTION(TypeError) << "For op '" << op_name
                            << "' input must be [int32, int64, float32, float64, bool], but got " << elem->ToString();
  }
  return res;
}

template MS_CORE_API int64_t GetScalarCastValue(const std::string &op_name, const ValuePtr &elem);
template MS_CORE_API int32_t GetScalarCastValue(const std::string &op_name, const ValuePtr &elem);
template MS_CORE_API int16_t GetScalarCastValue(const std::string &op_name, const ValuePtr &elem);
template MS_CORE_API int8_t GetScalarCastValue(const std::string &op_name, const ValuePtr &elem);
template MS_CORE_API uint64_t GetScalarCastValue(const std::string &op_name, const ValuePtr &elem);
template MS_CORE_API uint32_t GetScalarCastValue(const std::string &op_name, const ValuePtr &elem);
template MS_CORE_API uint16_t GetScalarCastValue(const std::string &op_name, const ValuePtr &elem);
template MS_CORE_API uint8_t GetScalarCastValue(const std::string &op_name, const ValuePtr &elem);
template MS_CORE_API double GetScalarCastValue(const std::string &op_name, const ValuePtr &elem);
template MS_CORE_API float GetScalarCastValue(const std::string &op_name, const ValuePtr &elem);
template MS_CORE_API bool GetScalarCastValue(const std::string &op_name, const ValuePtr &elem);

template <typename T>
std::optional<T> GetScalarValue(const ValuePtr &value) {
  MS_EXCEPTION_IF_NULL(value);
  if (value->isa<ValueAny>() || value->isa<None>()) {
    return std::nullopt;
  }

  if (value->isa<KernelTensorValue>()) {
    auto kernel_tensor_value = value->cast<KernelTensorValuePtr>();
    MS_EXCEPTION_IF_NULL(kernel_tensor_value);

    MS_EXCEPTION_IF_CHECK_FAIL((kernel_tensor_value->GetDataSize() == sizeof(T)),
                               "The data size in kernel tensor value which contains a scalar [" +
                                 std::to_string(kernel_tensor_value->GetDataSize()) +
                                 "] is not equal to the data type size [" + std::to_string(sizeof(T)) + "]");

    const T *data_ptr = reinterpret_cast<const T *>(kernel_tensor_value->GetDataPtr());
    MS_EXCEPTION_IF_NULL(data_ptr);
    return *data_ptr;
  }

  return GetValue<T>(value);
}

// Specialization for std::string type.
template <>
MS_CORE_API std::optional<std::string> GetScalarValue(const ValuePtr &value) {
  MS_EXCEPTION_IF_NULL(value);
  if (value->isa<ValueAny>()) {
    return std::nullopt;
  }

  if (value->isa<KernelTensorValue>()) {
    auto kernel_tensor_value = value->cast<KernelTensorValuePtr>();
    MS_EXCEPTION_IF_NULL(kernel_tensor_value);
    const char *data_ptr = reinterpret_cast<const char *>(kernel_tensor_value->GetDataPtr());
    MS_EXCEPTION_IF_NULL(data_ptr);
    size_t str_len = kernel_tensor_value->GetDataSize();

    return std::string(data_ptr, data_ptr + str_len);
  }

  return GetValue<std::string>(value);
}

template MS_CORE_API std::optional<int64_t> GetScalarValue(const ValuePtr &value);
template MS_CORE_API std::optional<int32_t> GetScalarValue(const ValuePtr &value);
template MS_CORE_API std::optional<int16_t> GetScalarValue(const ValuePtr &value);
template MS_CORE_API std::optional<int8_t> GetScalarValue(const ValuePtr &value);
template MS_CORE_API std::optional<uint64_t> GetScalarValue(const ValuePtr &value);
template MS_CORE_API std::optional<uint32_t> GetScalarValue(const ValuePtr &value);
template MS_CORE_API std::optional<uint16_t> GetScalarValue(const ValuePtr &value);
template MS_CORE_API std::optional<uint8_t> GetScalarValue(const ValuePtr &value);
template MS_CORE_API std::optional<double> GetScalarValue(const ValuePtr &value);
template MS_CORE_API std::optional<float> GetScalarValue(const ValuePtr &value);
template MS_CORE_API std::optional<bool> GetScalarValue(const ValuePtr &value);

// ABI-safe implementation: avoid cross-module std::optional issues
template <typename T>
bool GetScalarValuePtr(const ValuePtr &value, T *out_value) {
  MS_EXCEPTION_IF_NULL(value);
  MS_EXCEPTION_IF_NULL(out_value);
  auto opt = GetScalarValue<T>(value);
  if (opt.has_value()) {
    *out_value = opt.value();
    return true;
  }
  return false;
}

// Specialization for std::string type
template <>
MS_CORE_API bool GetScalarValuePtr<std::string>(const ValuePtr &value, std::string *out_value) {
  MS_EXCEPTION_IF_NULL(value);
  MS_EXCEPTION_IF_NULL(out_value);

  auto opt = GetScalarValue<std::string>(value);
  if (opt.has_value()) {
    *out_value = opt.value();
    return true;
  }
  return false;
}

// Explicit instantiation for ABI-safe pointer interfaces
template MS_CORE_API bool GetScalarValuePtr<int64_t>(const ValuePtr &value, int64_t *out_value);
template MS_CORE_API bool GetScalarValuePtr<int32_t>(const ValuePtr &value, int32_t *out_value);
template MS_CORE_API bool GetScalarValuePtr<int16_t>(const ValuePtr &value, int16_t *out_value);
template MS_CORE_API bool GetScalarValuePtr<int8_t>(const ValuePtr &value, int8_t *out_value);
template MS_CORE_API bool GetScalarValuePtr<uint64_t>(const ValuePtr &value, uint64_t *out_value);
template MS_CORE_API bool GetScalarValuePtr<uint32_t>(const ValuePtr &value, uint32_t *out_value);
template MS_CORE_API bool GetScalarValuePtr<uint16_t>(const ValuePtr &value, uint16_t *out_value);
template MS_CORE_API bool GetScalarValuePtr<uint8_t>(const ValuePtr &value, uint8_t *out_value);
template MS_CORE_API bool GetScalarValuePtr<double>(const ValuePtr &value, double *out_value);
template MS_CORE_API bool GetScalarValuePtr<float>(const ValuePtr &value, float *out_value);
template MS_CORE_API bool GetScalarValuePtr<bool>(const ValuePtr &value, bool *out_value);

// This interface is only used to convert values of type Sequence or Tensor to std::vector.
template <typename T>
std::optional<ArrayValue<T>> GetArrayValue(const ValuePtr &value) {
  MS_EXCEPTION_IF_NULL(value);
  if (value->isa<ValueAny>() || value->isa<None>()) {
    return std::nullopt;
  }

  std::vector<T> array_data;
  std::set<size_t> unknown_value_indexes;
  if (value->isa<KernelTensorValue>()) {
    auto kernel_tensor_value = value->cast<KernelTensorValuePtr>();
    MS_EXCEPTION_IF_NULL(kernel_tensor_value);

    if (kernel_tensor_value->GetDataSize() % sizeof(T) != 0) {
      MS_LOG(EXCEPTION) << "The size is incompatible, kernel tensor value size: " << kernel_tensor_value->GetDataSize()
                        << ", expected element size: " << sizeof(T);
    }

    size_t element_size = kernel_tensor_value->GetDataSize() / sizeof(T);
    if (element_size != 0) {
      const T *data_ptr = reinterpret_cast<const T *>(kernel_tensor_value->GetDataPtr());
      MS_EXCEPTION_IF_NULL(data_ptr);
      array_data.assign(data_ptr, data_ptr + element_size);
    }
  } else if (value->isa<ValueSequence>()) {
    // Sequence structure: Data is stored discretely.
    auto value_seq = value->cast<ValueSequencePtr>();
    MS_EXCEPTION_IF_NULL(value_seq);

    const auto &element_values = value_seq->value();
    size_t element_size = element_values.size();
    array_data.reserve(element_size);
    for (size_t i = 0; i < element_size; i++) {
      const auto &element = element_values[i];
      MS_EXCEPTION_IF_NULL(element);
      if (element->isa<ValueAny>() || element->isa<None>()) {
        array_data.push_back(static_cast<T>(0));
        (void)unknown_value_indexes.insert(i);
        continue;
      }
      if constexpr (std::is_same_v<T, float16>) {
        MS_LOG(EXCEPTION) << "For ValueSequence, float16 type is not support!";
      } else {
        array_data.push_back(GetValue<T>(element));
      }
    }
  } else if (value->isa<tensor::Tensor>()) {
    // Tensor structure: Data is stored continuously.
    auto tensor = value->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(tensor);
    size_t element_size = tensor->DataSize();
    auto cpu_tensor = tensor->cpu();
    T *data = reinterpret_cast<T *>(cpu_tensor->data_c());
    array_data.assign(data, data + element_size);
  } else {
    MS_LOG(EXCEPTION) << "Failed to get array value, expect sequence or tensor type, but got: " << value->type_name();
  }
  return std::optional<ArrayValue<T>>(std::in_place, std::move(array_data), std::move(unknown_value_indexes));
}

template <typename T>
std::optional<ArrayValue<T>> GetArrayValue(const abstract::AbstractBasePtr &abs_base) {
  MS_EXCEPTION_IF_NULL(abs_base);
  auto value = abs_base->GetValue();
  // If value is constant or is value sequence with some constant elements.
  if (!value->isa<ValueAny>()) {
    return GetArrayValue<T>(value);
  }

  // If value is ValueAny, need check whether abstract is AbstractSequence, it is in frontend.
  std::vector<T> array_data;
  std::set<size_t> unknown_value_indexes;
  if (abs_base->isa<abstract::AbstractSequence>()) {
    auto abs_sequence = abs_base->cast<abstract::AbstractSequencePtr>();
    MS_EXCEPTION_IF_NULL(abs_sequence);
    if (abs_sequence->dynamic_len()) {
      return std::nullopt;
    }
    for (size_t i = 0; i < abs_sequence->size(); ++i) {
      auto elem_value = abs_sequence->elements()[i]->GetValue();
      if (elem_value->isa<ValueAny>() || elem_value->isa<None>()) {
        array_data.push_back(static_cast<T>(0));
        (void)unknown_value_indexes.insert(i);
        continue;
      }
      if constexpr (std::is_same_v<T, float16>) {
        MS_LOG(EXCEPTION) << "For ValueSequence, float16 type is not support!";
      } else {
        array_data.push_back(GetValue<T>(elem_value));
      }
    }
    return std::optional<ArrayValue<T>>(std::in_place, std::move(array_data), std::move(unknown_value_indexes));
  }
  // Only abstract sequence with ValueAny need to handle, other situation just return nullopt.
  return std::nullopt;
}

template MS_CORE_API std::optional<ArrayValue<int64_t>> GetArrayValue(const ValuePtr &value);
template MS_CORE_API std::optional<ArrayValue<int32_t>> GetArrayValue(const ValuePtr &value);
template MS_CORE_API std::optional<ArrayValue<int16_t>> GetArrayValue(const ValuePtr &value);
template MS_CORE_API std::optional<ArrayValue<int8_t>> GetArrayValue(const ValuePtr &value);
template MS_CORE_API std::optional<ArrayValue<uint64_t>> GetArrayValue(const ValuePtr &value);
template MS_CORE_API std::optional<ArrayValue<uint32_t>> GetArrayValue(const ValuePtr &value);
template MS_CORE_API std::optional<ArrayValue<uint16_t>> GetArrayValue(const ValuePtr &value);
template MS_CORE_API std::optional<ArrayValue<uint8_t>> GetArrayValue(const ValuePtr &value);
template MS_CORE_API std::optional<ArrayValue<double>> GetArrayValue(const ValuePtr &value);
template MS_CORE_API std::optional<ArrayValue<float>> GetArrayValue(const ValuePtr &value);
template MS_CORE_API std::optional<ArrayValue<bool>> GetArrayValue(const ValuePtr &value);
template MS_CORE_API std::optional<ArrayValue<std::string>> GetArrayValue(const ValuePtr &value);
template MS_CORE_API std::optional<ArrayValue<float16>> GetArrayValue(const ValuePtr &value);
template MS_CORE_API std::optional<ArrayValue<bfloat16>> GetArrayValue(const ValuePtr &value);

template MS_CORE_API std::optional<ArrayValue<int64_t>> GetArrayValue(const abstract::AbstractBasePtr &abs_base);
template MS_CORE_API std::optional<ArrayValue<int32_t>> GetArrayValue(const abstract::AbstractBasePtr &abs_base);
template MS_CORE_API std::optional<ArrayValue<int16_t>> GetArrayValue(const abstract::AbstractBasePtr &abs_base);
template MS_CORE_API std::optional<ArrayValue<int8_t>> GetArrayValue(const abstract::AbstractBasePtr &abs_base);
template MS_CORE_API std::optional<ArrayValue<uint64_t>> GetArrayValue(const abstract::AbstractBasePtr &abs_base);
template MS_CORE_API std::optional<ArrayValue<uint32_t>> GetArrayValue(const abstract::AbstractBasePtr &abs_base);
template MS_CORE_API std::optional<ArrayValue<uint16_t>> GetArrayValue(const abstract::AbstractBasePtr &abs_base);
template MS_CORE_API std::optional<ArrayValue<uint8_t>> GetArrayValue(const abstract::AbstractBasePtr &abs_base);
template MS_CORE_API std::optional<ArrayValue<double>> GetArrayValue(const abstract::AbstractBasePtr &abs_base);
template MS_CORE_API std::optional<ArrayValue<float>> GetArrayValue(const abstract::AbstractBasePtr &abs_base);
template MS_CORE_API std::optional<ArrayValue<bool>> GetArrayValue(const abstract::AbstractBasePtr &abs_base);
template MS_CORE_API std::optional<ArrayValue<std::string>> GetArrayValue(const abstract::AbstractBasePtr &abs_base);
template MS_CORE_API std::optional<ArrayValue<float16>> GetArrayValue(const abstract::AbstractBasePtr &abs_base);
template MS_CORE_API std::optional<ArrayValue<bfloat16>> GetArrayValue(const abstract::AbstractBasePtr &abs_base);

namespace {
template <typename T>
struct TypeIdTrait {};

#define TYPEID_TRAIT(typeid, prototype)   \
  template <>                             \
  struct TypeIdTrait<prototype> {         \
    static const TypeId type_id = typeid; \
  }

TYPEID_TRAIT(kNumberTypeInt8, int8_t);
TYPEID_TRAIT(kNumberTypeUInt8, uint8_t);
TYPEID_TRAIT(kNumberTypeInt16, int16_t);
TYPEID_TRAIT(kNumberTypeUInt16, uint16_t);
TYPEID_TRAIT(kNumberTypeInt32, int32_t);
TYPEID_TRAIT(kNumberTypeUInt32, uint32_t);
TYPEID_TRAIT(kNumberTypeInt64, int64_t);
TYPEID_TRAIT(kNumberTypeUInt64, uint64_t);
TYPEID_TRAIT(kNumberTypeFloat16, float16);
TYPEID_TRAIT(kNumberTypeFloat32, float);
TYPEID_TRAIT(kNumberTypeFloat64, double);
TYPEID_TRAIT(kNumberTypeBFloat16, bfloat16);
TYPEID_TRAIT(kNumberTypeBool, bool);
TYPEID_TRAIT(kNumberTypeComplex64, std::complex<float>);
TYPEID_TRAIT(kNumberTypeComplex128, std::complex<double>);
}  // namespace

template <typename T>
T TensorItem(const tensor::TensorPtr &tensor) {
  // Wait tasks finish before get Tensor data itemsize.
  // There is implicit copy for input Tensor, which may change the Tensor device_address.
  auto cpu_tensor = tensor->cpu();
  if (cpu_tensor->DataItemSize() != sizeof(T)) {
    MS_EXCEPTION(TypeError) << "The tensor's type is " << TypeIdToString(cpu_tensor->data_type()) << ", which is not "
                            << TypeIdToString(TypeIdTrait<T>::type_id);
  }
  if (cpu_tensor->DataSize() != 1) {
    MS_EXCEPTION(ValueError) << "The tensor should have only one element, but got " << cpu_tensor->DataSize() << ","
                             << " more than one element is ambiguous.";
  }
  auto data = cpu_tensor->data_c();
  if constexpr (std::is_same_v<T, std::complex<float>>) {
    return std::complex<float>{(*static_cast<const float *>(data)), (*(static_cast<const float *>(data) + 1))};
  } else if constexpr (std::is_same_v<T, std::complex<double>>) {
    return std::complex<double>{(*static_cast<const double *>(data)), (*(static_cast<const double *>(data) + 1))};
  } else {
    return *static_cast<const T *>(data);
  }
}

#define TENSOR_ITEM_TEMPLATE(TYPE) template MS_CORE_API TYPE TensorItem<TYPE>(const tensor::TensorPtr &value)

TENSOR_ITEM_TEMPLATE(int8_t);
TENSOR_ITEM_TEMPLATE(uint8_t);
TENSOR_ITEM_TEMPLATE(int16_t);
TENSOR_ITEM_TEMPLATE(uint16_t);
TENSOR_ITEM_TEMPLATE(int32_t);
TENSOR_ITEM_TEMPLATE(uint32_t);
TENSOR_ITEM_TEMPLATE(int64_t);
TENSOR_ITEM_TEMPLATE(uint64_t);
TENSOR_ITEM_TEMPLATE(float16);
TENSOR_ITEM_TEMPLATE(float);
TENSOR_ITEM_TEMPLATE(double);
TENSOR_ITEM_TEMPLATE(bfloat16);
TENSOR_ITEM_TEMPLATE(bool);
TENSOR_ITEM_TEMPLATE(std::complex<float>);
TENSOR_ITEM_TEMPLATE(std::complex<double>);
}  //  namespace mindspore
