/**
 * Copyright 2023-2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_TRANSFORM_ACL_IR_OP_API_CONVERT_H_
#define MINDSPORE_CCSRC_TRANSFORM_ACL_IR_OP_API_CONVERT_H_

#include <dlfcn.h>
#include <numeric>
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include <algorithm>
#include <functional>
#include <regex>
#include <utility>
#include <tuple>
#include "kernel/ascend/visible.h"
#include "acl/acl_base.h"
#include "ir/tensor.h"
#include "kernel/ascend/acl_ir/acl_convert.h"
#include "device_address/device_address.h"
#include "kernel/ascend/acl_ir/acl_helper.h"

namespace mindspore::device::ascend {
// Api data struct.
typedef struct aclOpExecutor aclOpExecutor;
typedef struct aclTensor aclTensor;
typedef struct aclScalar aclScalar;
typedef struct aclIntArray aclIntArray;
typedef struct aclFloatArray aclFloatArray;
typedef struct aclBoolArray aclBoolArray;
typedef struct aclTensorList aclTensorList;

// Create operator.
using _aclCreateTensor = aclTensor *(*)(const int64_t *view_dims, uint64_t view_dims_num, aclDataType data_type,
                                        const int64_t *stride, int64_t offset, aclFormat format,
                                        const int64_t *storage_dims, uint64_t storage_dims_num, void *tensor_data);
using _aclCreateScalar = aclScalar *(*)(void *value, aclDataType data_type);
using _aclCreateIntArray = aclIntArray *(*)(const int64_t *value, uint64_t size);
using _aclCreateFloatArray = aclFloatArray *(*)(const float *value, uint64_t size);
using _aclCreateBoolArray = aclBoolArray *(*)(const bool *value, uint64_t size);
using _aclCreateTensorList = aclTensorList *(*)(const aclTensor *const *value, uint64_t size);
// Destroy operator.
using _aclDestroyTensor = int (*)(const aclTensor *tensor);
using _aclDestroyScalar = int (*)(const aclScalar *scalar);
using _aclDestroyIntArray = int (*)(const aclIntArray *array);
using _aclDestroyFloatArray = int (*)(const aclFloatArray *array);
using _aclDestroyBoolArray = int (*)(const aclBoolArray *array);
using _aclDestroyTensorList = int (*)(const aclTensorList *array);
using _aclDestroyAclOpExecutor = int (*)(aclOpExecutor *executor);
// Update operator.
using _aclSetAclOpExecutorRepeatable = int (*)(aclOpExecutor *executor);
using _aclSetTensorAddr = int (*)(aclOpExecutor *executor, const size_t index, aclTensor *tensor, void *addr);
using _aclSetDynamicTensorAddr = int (*)(aclOpExecutor *executor, const size_t index, const size_t relativeIndex,
                                         aclTensorList *tensors, void *addr);

extern void LoadOpApiLib();

// Get op api func.
inline std::string GetOpApiLibName() { return "/lib64/libopapi.so"; }

inline std::string GetCustOpApiLibName() { return "/op_api/lib/libcust_opapi.so"; }

inline void *GetOpApiFuncFromLib(void *handler, const char *lib_name, const char *api_name) {
  MS_EXCEPTION_IF_NULL(handler);
  auto func = dlsym(handler, api_name);
  if (func == nullptr) {
    MS_LOG(DEBUG) << "Dlsym " << api_name << " from " << lib_name << " failed!" << dlerror();
  }
  return func;
}

inline void *GetOpApiLibHandler(const std::string &lib_path) {
  auto handler = dlopen(lib_path.c_str(), RTLD_LAZY);
  if (handler == nullptr) {
    MS_LOG(INFO) << "Dlopen " << lib_path << " failed!" << dlerror();
  }
  return handler;
}

OPS_ASCEND_API void *GetOpApiFunc(const char *api_name);

#define GET_OP_API_FUNC(func_name) reinterpret_cast<_##func_name>(GetOpApiFunc(#func_name))

template <typename Tuple, size_t... I>
auto ConvertToOpApiFunc(const Tuple &params, void *opApiAddr, std::index_sequence<I...>) {
  using OpApiFunc = int (*)(typename std::decay<decltype(std::get<I>(params))>::type...);
  auto func = reinterpret_cast<OpApiFunc>(opApiAddr);
  return func;
}

template <typename Tuple>
auto ConvertToOpApiFunc(const Tuple &params, void *opApiAddr) {
  static constexpr auto size = std::tuple_size<Tuple>::value;
  return ConvertToOpApiFunc(params, opApiAddr, std::make_index_sequence<size>{});
}

// Convert Value
class OpApiTensorConverter : public AttrHelper<OpApiTensorConverter> {
 public:
  OpApiTensorConverter() = default;
  ~OpApiTensorConverter() = default;

  template <typename T>
  void ConvertValue(const ValuePtr &value, const AttrDeclType<T> &, aclScalar **scalar) {
    auto real_val = GetValue<T>(value);
    MS_EXCEPTION_IF_NULL(scalar);
    *scalar = CreateAclScalar(&real_val, GetDataType(value));
  }

  void ConvertValue(const ValuePtr &value, const AttrDeclType<int32_t> &, aclScalar **scalar) {
    auto real_val = static_cast<int64_t>(GetValue<int32_t>(value));
    MS_EXCEPTION_IF_NULL(scalar);
    *scalar = CreateAclScalar(&real_val, ACL_INT64);
  }

  void ConvertValue(const ValuePtr &value, const AttrDeclType<std::vector<int64_t>> &, aclIntArray *array) {
    std::vector<int64_t> array_list;
    ConvertValueSequenceToList(value, &array_list);
    array = CreateIntArray(array_list);
  }

  void ConvertValue(const ValuePtr &value, const AttrDeclType<std::vector<int32_t>> &, aclIntArray *array) {
    std::vector<int32_t> array_list;
    ConvertValueSequenceToList(value, &array_list);
    std::vector<int64_t> array_list_int64;
    (void)std::transform(array_list.begin(), array_list.end(), std::back_inserter(array_list_int64),
                         [](const int val) { return mindspore::IntToLong(val); });
    array = CreateIntArray(array_list_int64);
  }

  void ConvertValue(const ValuePtr &value, const AttrDeclType<std::vector<uint8_t>> &, aclBoolArray *array) {
    std::vector<uint8_t> array_list;
    ConvertValueSequenceToList(value, &array_list);
    array = CreateBoolArray(array_list);
  }

  void ConvertValue(const ValuePtr &value, const AttrDeclType<std::vector<float>> &, aclFloatArray *array) {
    std::vector<float> array_list;
    ConvertValueSequenceToList(value, &array_list);
    array = CreateFloatArray(array_list);
  }

  template <typename T>
  aclScalar *CreateAclScalar(T *val, aclDataType dtype) {
    static const auto aclCreateScalar = GET_OP_API_FUNC(aclCreateScalar);
    if (aclCreateScalar == nullptr) {
      MS_LOG(EXCEPTION) << "Failed to get `aclCreateScalar` func.";
    }
    return aclCreateScalar(val, dtype);
  }

  aclIntArray *CreateIntArray(const std::vector<int64_t> &val) {
    static const auto aclCreateIntArray = GET_OP_API_FUNC(aclCreateIntArray);
    if (aclCreateIntArray == nullptr) {
      MS_LOG(EXCEPTION) << "Failed to get `aclCreateIntArray` func.";
    }
    return aclCreateIntArray(val.data(), val.size());
  }

  aclBoolArray *CreateBoolArray(const std::vector<uint8_t> &val) {
    static const auto aclCreateBoolArray = GET_OP_API_FUNC(aclCreateBoolArray);
    if (aclCreateBoolArray == nullptr) {
      MS_LOG(EXCEPTION) << "Failed to get `aclCreateBoolArray` func.";
    }
    return aclCreateBoolArray(reinterpret_cast<const bool *>(val.data()), val.size());
  }

  aclFloatArray *CreateFloatArray(const std::vector<float> &val) {
    static const auto aclCreateFloatArray = GET_OP_API_FUNC(aclCreateFloatArray);
    if (aclCreateFloatArray == nullptr) {
      MS_LOG(EXCEPTION) << "Failed to get `aclCreateFloatArray` func.";
    }
    return aclCreateFloatArray(val.data(), val.size());
  }

 private:
  inline aclDataType GetDataType(const ValuePtr &value) {
    MS_EXCEPTION_IF_NULL(value);
    MS_EXCEPTION_IF_NULL(value->type());
    return AclConverter::ConvertType(value->type()->type_id());
  }
};

inline aclTensor *ConvertType(const mindspore::kernel::KernelTensor *tensor) {
  static const auto aclCreateTensor = GET_OP_API_FUNC(aclCreateTensor);
  if (aclCreateTensor == nullptr) {
    return nullptr;
  }
  if (tensor == nullptr || tensor->type_id() == kMetaTypeNone) {
    return nullptr;
  }

  auto acl_data_type = AclConverter::ConvertType(tensor->dtype_id());
  const auto &shape = tensor->GetShapeVector();
  const auto shape_size = shape.size();
  aclFormat format = ACL_FORMAT_ND;
  switch (shape_size) {
    case 3:
      format = ACL_FORMAT_NCL;
      break;
    case 4:
      format = ACL_FORMAT_NCHW;
      break;
    case 5:
      format = ACL_FORMAT_NCDHW;
      break;
    default:
      format = ACL_FORMAT_ND;
  }
  if (tensor->format() == FRACTAL_NZ) {
    format = ACL_FORMAT_FRACTAL_NZ;
  }

  aclTensor *acl_tensor = nullptr;
  const auto &storage_info = tensor->tensor_storage_info();
  if (storage_info == nullptr) {
    // Create strides.
    auto strides = shape;
    if (!strides.empty()) {
      strides.erase(strides.begin());
    }
    strides.push_back(1);
    for (int i = static_cast<int>(strides.size()) - 2; i >= 0; i--) {
      strides[i] = strides[i] * strides[i + 1];
    }
    acl_tensor = aclCreateTensor(shape.data(), shape_size, acl_data_type, strides.data(), 0, format, shape.data(),
                                 shape.size(), tensor->device_ptr());
  } else {
    const auto &strides = storage_info->strides;
    const auto &storage_shape = storage_info->ori_shape;
    acl_tensor =
      aclCreateTensor(shape.data(), shape_size, acl_data_type, strides.data(), SizeToLong(storage_info->storage_offset),
                      format, storage_shape.data(), storage_shape.size(), tensor->device_ptr());
  }

  return acl_tensor;
}

inline aclTensor *ConvertType(mindspore::kernel::KernelTensor *tensor) {
  return ConvertType(reinterpret_cast<const mindspore::kernel::KernelTensor *>(tensor));
}

inline aclTensor *ConvertType(std::pair<mindspore::kernel::KernelTensor *, bool> tensor_and_trans) {
  static const auto aclCreateTensor = GET_OP_API_FUNC(aclCreateTensor);
  if (aclCreateTensor == nullptr) {
    return nullptr;
  }
  auto tensor = tensor_and_trans.first;
  auto trans = tensor_and_trans.second;
  auto acl_data_type = AclConverter::ConvertType(tensor->dtype_id());
  auto shape = tensor->GetShapeVector();
  const auto shape_size = shape.size();
  aclFormat format = ACL_FORMAT_ND;
  switch (shape_size) {
    case 3:
      format = ACL_FORMAT_NCL;
      break;
    case 4:
      format = ACL_FORMAT_NCHW;
      break;
    case 5:
      format = ACL_FORMAT_NCDHW;
      break;
    default:
      format = ACL_FORMAT_ND;
  }
  aclTensor *acl_tensor = nullptr;
  const auto &storage_info = tensor->tensor_storage_info();
  if (storage_info == nullptr) {
    // Create strides.
    auto strides = shape;
    if (!strides.empty()) {
      strides.erase(strides.begin());
    }
    strides.push_back(1);
    for (int i = static_cast<int>(strides.size()) - 2; i >= 0; i--) {
      strides[i] = strides[i] * strides[i + 1];
    }
    // Check if shape need transpose.
    if (trans) {
      std::swap(shape[shape.size() - 1], shape[shape.size() - 2]);
      std::swap(strides[strides.size() - 1], strides[strides.size() - 2]);
    }
    acl_tensor = aclCreateTensor(shape.data(), shape_size, acl_data_type, strides.data(), 0, format, shape.data(),
                                 shape.size(), tensor->device_ptr());
  } else {
    auto strides = storage_info->strides;
    const auto &storage_shape = storage_info->ori_shape;
    // Check if shape need transpose.
    if (trans) {
      std::swap(shape[shape.size() - 1], shape[shape.size() - 2]);
      std::swap(strides[strides.size() - 1], strides[strides.size() - 2]);
    }
    acl_tensor =
      aclCreateTensor(shape.data(), shape_size, acl_data_type, strides.data(), SizeToLong(storage_info->storage_offset),
                      format, storage_shape.data(), storage_shape.size(), tensor->device_ptr());
  }
  return acl_tensor;
}

inline aclTensor *ConvertType(const tensor::TensorPtr &tensor) {
  MS_EXCEPTION_IF_NULL(tensor);
  static const auto aclCreateTensor = GET_OP_API_FUNC(aclCreateTensor);
  if (aclCreateTensor == nullptr) {
    return nullptr;
  }

  const auto &shape = tensor->shape();
  const auto shape_size = shape.size();
  aclFormat format = ACL_FORMAT_ND;
  switch (shape_size) {
    case kSizeThree:
      format = ACL_FORMAT_NCL;
      break;
    case kSizeFour:
      format = ACL_FORMAT_NCHW;
      break;
    case kSizeFive:
      format = ACL_FORMAT_NCDHW;
      break;
    default:
      format = ACL_FORMAT_ND;
  }
  auto acl_data_type = AclConverter::ConvertType(tensor->data_type());
  auto device_address = tensor->device_address();
  if (device_address->size() != 0 && device_address->GetMutablePtr() == nullptr) {
    MS_LOG(EXCEPTION) << "The device memory is null, please allocate the device memory for tensor "
                      << tensor->ToString();
  }

  static const auto GetTensorNum = [](const std::vector<int64_t> &shape) {
    auto num = std::accumulate(shape.begin(), shape.end(), int64_t(1), std::multiplies<int64_t>());
    return num;
  };
  const auto &strides = tensor->stride();
  std::vector<int64_t> storage_shape;
  const auto &storage_info = tensor->storage_info();
  if (storage_info) {
    // If format is fractal_nz, the StorageDims of aclTensor need use storage_info->ori_shape.
    if (tensor->format() == Format::FRACTAL_NZ) {
      format = ACL_FORMAT_FRACTAL_NZ;
      storage_shape = storage_info->ori_shape;
    } else {
      storage_shape = std::vector<int64_t>{GetTensorNum(storage_info->ori_shape)};
    }
  } else {
    storage_shape = std::vector<int64_t>{GetTensorNum(shape)};
  }

  auto acl_tensor =
    aclCreateTensor(shape.data(), shape.size(), acl_data_type, strides.data(), tensor->storage_offset(), format,
                    storage_shape.data(), storage_shape.size(), device_address->GetMutablePtr());

  return acl_tensor;
}

inline aclIntArray *ConvertType(const std::vector<int64_t> &int_array) {
  if (int_array.empty()) {
    MS_LOG(DEBUG) << "int array is empty!";
  }
  static OpApiTensorConverter converter;
  return converter.CreateIntArray(int_array);
}

inline aclIntArray *ConvertType(const std::optional<std::vector<int64_t>> &int_array_opt) {
  if (int_array_opt.has_value()) {
    return ConvertType(int_array_opt.value());
  }
  return nullptr;
}

inline aclIntArray *ConvertType(const std::pair<std::vector<int64_t>, bool> &int_array_pair) {
  return ConvertType(int_array_pair.first);
}

inline aclFloatArray *ConvertType(const std::vector<float> &float_array) {
  if (float_array.empty()) {
    MS_LOG(ERROR) << "float array is empty!";
  }
  static OpApiTensorConverter converter;
  return converter.CreateFloatArray(float_array);
}

inline aclBoolArray *ConvertType(const std::vector<uint8_t> &bool_array) {
  if (bool_array.empty()) {
    MS_LOG(ERROR) << "bool array is empty!";
  }
  static OpApiTensorConverter converter;
  return converter.CreateBoolArray(bool_array);
}

inline aclTensor *ConvertType(const std::optional<tensor::TensorPtr> &value) {
  if (value.has_value()) {
    return ConvertType(value.value());
  }
  return nullptr;
}

inline aclTensorList *ConvertType(const std::vector<tensor::TensorPtr> &tensor_list) {
  if (tensor_list.empty()) {
    MS_LOG(DEBUG) << "tensor list is empty!";
  }
  static const auto aclCreateTensorList = GET_OP_API_FUNC(aclCreateTensorList);
  std::vector<aclTensor *> tmp;
  std::transform(tensor_list.begin(), tensor_list.end(), std::back_inserter(tmp),
                 [](const tensor::TensorPtr &tensor) { return ConvertType(tensor); });
  return aclCreateTensorList(tmp.data(), tmp.size());
}

inline aclTensorList *ConvertType(const std::vector<mindspore::kernel::KernelTensor *> &tensor_list) {
  if (tensor_list.empty()) {
    MS_LOG(DEBUG) << "tensor list is empty!";
  }
  static const auto aclCreateTensorList = GET_OP_API_FUNC(aclCreateTensorList);
  std::vector<aclTensor *> tmp;
  std::transform(tensor_list.begin(), tensor_list.end(), std::back_inserter(tmp),
                 [](mindspore::kernel::KernelTensor *tensor) { return ConvertType(tensor); });
  return aclCreateTensorList(tmp.data(), tmp.size());
}

inline aclScalar *ConvertType(const ScalarPtr &value) {
  if (value == nullptr) {
    // for None
    return nullptr;
  }
  aclScalar *acl_scalar;
  static OpApiTensorConverter converter;
  if (value->isa<BoolImm>()) {
    converter.ConvertValue(value, AttrDeclType<bool>(), &acl_scalar);
  } else if (value->isa<Int64Imm>()) {
    converter.ConvertValue(value, AttrDeclType<int64_t>(), &acl_scalar);
  } else if (value->isa<FP64Imm>()) {
    converter.ConvertValue(value, AttrDeclType<double>(), &acl_scalar);
  } else if (value->isa<FP32Imm>()) {
    converter.ConvertValue(value, AttrDeclType<float>(), &acl_scalar);
  } else if (value->isa<Int32Imm>()) {
    converter.ConvertValue(value, AttrDeclType<int32_t>(), &acl_scalar);
  } else if (value->isa<Int8Imm>()) {
    converter.ConvertValue(value, AttrDeclType<int8_t>(), &acl_scalar);
  } else if (value->isa<Int16Imm>()) {
    converter.ConvertValue(value, AttrDeclType<int16_t>(), &acl_scalar);
  } else if (value->isa<UInt8Imm>()) {
    converter.ConvertValue(value, AttrDeclType<uint8_t>(), &acl_scalar);
  } else if (value->isa<FP64Imm>()) {
    converter.ConvertValue(value, AttrDeclType<double>(), &acl_scalar);
  } else if (value->isa<BF16Imm>()) {
    converter.ConvertValue(value, AttrDeclType<bfloat16>(), &acl_scalar);
  } else if (value->isa<FP16Imm>()) {
    converter.ConvertValue(value, AttrDeclType<float16>(), &acl_scalar);
  } else {
    MS_LOG(EXCEPTION) << "Currently not support value: " << value->ToString();
  }
  return acl_scalar;
}

inline aclScalar *ConvertType(const std::optional<ScalarPtr> &value) {
  if (value.has_value()) {
    return ConvertType(value.value());
  }
  return nullptr;
}

inline aclDataType ConvertType(TypeId type_id) { return AclConverter::ConvertType(type_id); }

inline aclDataType ConvertType(const TypePtr &type) { return AclConverter::ConvertType(type->type_id()); }

inline const char *ConvertType(const std::string &value) { return value.c_str(); }

inline void *ConvertType(const ValueTuplePtr &value) {
  MS_EXCEPTION_IF_NULL(value);
  auto elements = value->value();
  if (elements.empty()) {
    return nullptr;
  } else if (elements[0]->isa<UInt8Imm>() || elements[0]->isa<BoolImm>()) {
    auto bool_vector = GetValue<std::vector<uint8_t>>(value);
    return ConvertType(bool_vector);
  } else if (elements[0]->isa<FP32Imm>()) {
    auto float_vector = GetValue<std::vector<float>>(value);
    return ConvertType(float_vector);
  } else if (elements[0]->isa<Int64Imm>()) {
    auto int_vector = GetValue<std::vector<int64_t>>(value);
    return ConvertType(int_vector);
  } else if (elements[0]->isa<tensor::Tensor>()) {
    auto tensor_vector = GetValue<std::vector<tensor::TensorPtr>>(value);
    return ConvertType(tensor_vector);
  } else {
    MS_LOG(EXCEPTION) << "Unsupported type: " << value->ToString();
  }
}

inline void *ConvertType(const ValuePtr &value) {
  if (value == nullptr) {
    MS_LOG(DEBUG) << "Convert value is null";
    // optional input;
    return nullptr;
  } else if (value->isa<tensor::Tensor>()) {
    const auto &base_tensor = dyn_cast<tensor::Tensor>(value);
    return ConvertType(base_tensor);
  } else if (value->isa<Scalar>()) {
    const auto &scalar = dyn_cast<Scalar>(value);
    return ConvertType(scalar);
  } else if (value->isa<StringImm>()) {
    const auto &str_value = dyn_cast<StringImm>(value);
    auto str = GetValue<std::string>(str_value);
    return const_cast<char *>(ConvertType(str));
  } else if (value->isa<ValueTuple>()) {
    const auto &tuple_value = dyn_cast<ValueTuple>(value);
    return ConvertType(tuple_value);
  } else {
    MS_LOG(EXCEPTION) << "Unsupported type: " << value->ToString();
  }
}

template <typename T, typename = std::enable_if_t<std::is_scalar_v<T>>>
T ConvertType(T value) {
  return value;
}

template <typename... Ts>
constexpr auto ConvertTypes(const Ts &...args) {
  return std::make_tuple(ConvertType(args)...);
}

inline std::vector<void *> GetAddr(const std::vector<KernelTensor *> tensor_list) {
  std::vector<void *> addr_list;
  for (const auto &tensor : tensor_list) {
    MS_EXCEPTION_IF_NULL(tensor);
    (void)addr_list.emplace_back(tensor->device_ptr());
  }
  return addr_list;
}

inline std::vector<void *> GetAddr(KernelTensor *tensor) {
  MS_EXCEPTION_IF_NULL(tensor);
  return {tensor->device_ptr()};
}

inline std::vector<void *> GetAddr(const std::pair<KernelTensor *, bool> &tensor_pair) {
  MS_EXCEPTION_IF_NULL(tensor_pair.first);
  return {tensor_pair.first->device_ptr()};
}

inline std::vector<void *> GetAddr(const device::DeviceAddressPtr &device_address) {
  return {device_address->GetMutablePtr()};
}

inline std::vector<void *> GetAddr(device::DeviceAddress *device_address) { return {device_address->GetMutablePtr()}; }

inline std::vector<void *> GetAddr(const std::vector<tensor::TensorPtr> &tensor_list) {
  std::vector<void *> addr_list;
  for (const auto &tensor : tensor_list) {
    MS_EXCEPTION_IF_NULL(tensor);
    (void)addr_list.emplace_back(tensor->device_address()->GetMutablePtr());
  }
  return addr_list;
}

inline std::vector<void *> GetAddr(const tensor::TensorPtr &tensor) {
  MS_EXCEPTION_IF_NULL(tensor);
  return {tensor->device_address()->GetMutablePtr()};
}

inline std::vector<void *> GetAddr(const std::pair<tensor::TensorPtr, bool> &tensor_pair) {
  MS_EXCEPTION_IF_NULL(tensor_pair.first);
  return {tensor_pair.first->device_address()->GetMutablePtr()};
}

inline std::vector<void *> GetAddr(const std::optional<tensor::TensorPtr> &tensor) {
  if (tensor.has_value()) {
    return {tensor.value()->device_address()->GetMutablePtr()};
  }
  return {nullptr};
}

inline std::vector<void *> GetAddr(const std::pair<std::vector<int64_t>, bool> &int_array_input) {
  if (int_array_input.second) {
    return {nullptr};
  }
  return {};
}

template <typename T>
inline std::vector<void *> GetAddr(T) {
  return {};
}

inline void FillAddress(std::vector<std::vector<void *>> *) {}

template <typename T, typename... Ts>
inline void FillAddress(std::vector<std::vector<void *>> *address_list, const T &arg, const Ts &...args) {
  // Current only input/output support nullptr.
  if constexpr (std::is_same_v<T, std::nullptr_t>) {
    std::vector<void *> empty_addr = {nullptr};
    (void)address_list->emplace_back(std::move(empty_addr));
  } else {
    (void)address_list->emplace_back(GetAddr(arg));
  }
  FillAddress(address_list, args...);
}

template <typename... Ts>
inline std::vector<std::vector<void *>> GetTensorAddress(const Ts &...args) {
  std::vector<std::vector<void *>> address_list;
  FillAddress(&address_list, args...);
  return address_list;
}

template <typename T>
T ConvertKernelTensor(mindspore::kernel::KernelTensor *tensor) {
  MS_EXCEPTION_IF_NULL(tensor);
  return tensor->GetValueWithCheck<T>();
}

template <>
inline ScalarPtr ConvertKernelTensor<ScalarPtr>(mindspore::kernel::KernelTensor *tensor) {
  MS_EXCEPTION_IF_NULL(tensor);
  if (tensor->dtype_id() == kMetaTypeNone) {
    // for None
    return nullptr;
  }
  auto value_ptr = tensor->GetValueTrack();
  if (value_ptr == nullptr || !value_ptr->isa<Scalar>()) {
    if (tensor->dtype_id() == kNumberTypeBool) {
      auto value = tensor->GetValueWithCheck<bool>();
      value_ptr = std::make_shared<BoolImm>(value);
    } else if (tensor->dtype_id() == kNumberTypeInt64) {
      auto value = tensor->GetValueWithCheck<int64_t>();
      value_ptr = std::make_shared<Int64Imm>(value);
    } else if (tensor->dtype_id() == kNumberTypeDouble || tensor->dtype_id() == kNumberTypeFloat64) {
      auto value = tensor->GetValueWithCheck<double>();
      value_ptr = std::make_shared<FP64Imm>(value);
    } else if (tensor->dtype_id() == kNumberTypeFloat32) {
      auto value = tensor->GetValueWithCheck<float>();
      value_ptr = std::make_shared<FP32Imm>(value);
    } else if (tensor->dtype_id() == kNumberTypeInt32) {
      auto value = tensor->GetValueWithCheck<int32_t>();
      value_ptr = std::make_shared<Int32Imm>(value);
    } else if (tensor->dtype_id() == kNumberTypeInt8) {
      auto value = tensor->GetValueWithCheck<int8_t>();
      value_ptr = std::make_shared<Int8Imm>(value);
    } else if (tensor->dtype_id() == kNumberTypeInt16) {
      auto value = tensor->GetValueWithCheck<int16_t>();
      value_ptr = std::make_shared<Int16Imm>(value);
    } else if (tensor->dtype_id() == kNumberTypeUInt8) {
      auto value = tensor->GetValueWithCheck<uint8_t>();
      value_ptr = std::make_shared<UInt8Imm>(value);
    } else if (tensor->dtype_id() == kNumberTypeBFloat16) {
      auto value = tensor->GetValueWithCheck<bfloat16>();
      value_ptr = std::make_shared<BF16Imm>(value);
    } else if (tensor->dtype_id() == kNumberTypeFloat16) {
      auto value = tensor->GetValueWithCheck<float16>();
      value_ptr = std::make_shared<FP16Imm>(value);
    } else {
      MS_LOG(EXCEPTION) << "Currently not support value type: " << tensor->dtype_id();
    }
  }

  MS_EXCEPTION_IF_NULL(value_ptr);
  auto scalar_ptr = value_ptr->cast<ScalarPtr>();
  MS_EXCEPTION_IF_NULL(scalar_ptr);
  return scalar_ptr;
}

template <>
inline std::vector<int64_t> ConvertKernelTensor<std::vector<int64_t>>(mindspore::kernel::KernelTensor *tensor) {
  MS_EXCEPTION_IF_NULL(tensor);
  auto res = tensor->GetOptionalValueWithCheck<std::vector<int64_t>>();
  if (res.has_value()) {
    return res.value();
  }
  return {};
}

template <>
inline std::vector<float> ConvertKernelTensor<std::vector<float>>(mindspore::kernel::KernelTensor *tensor) {
  MS_EXCEPTION_IF_NULL(tensor);
  return tensor->GetValueWithCheck<std::vector<float>>();
}

template <>
inline std::vector<uint8_t> ConvertKernelTensor<std::vector<uint8_t>>(mindspore::kernel::KernelTensor *tensor) {
  MS_EXCEPTION_IF_NULL(tensor);
  return tensor->GetValueWithCheck<std::vector<uint8_t>>();
}

template <>
inline TypeId ConvertKernelTensor<TypeId>(mindspore::kernel::KernelTensor *tensor) {
  MS_EXCEPTION_IF_NULL(tensor);
  return tensor->dtype_id();
}

template <>
inline std::vector<mindspore::kernel::KernelTensorPtr>
ConvertKernelTensor<std::vector<mindspore::kernel::KernelTensorPtr>>(mindspore::kernel::KernelTensor *tensor) {
  MS_EXCEPTION_IF_NULL(tensor);
  if (tensor->type_id() != kObjectTypeTuple && tensor->type_id() != kObjectTypeList) {
    return {std::make_shared<mindspore::kernel::KernelTensor>(*tensor)};
  }
  auto shape = tensor->GetShapeVector();
  if (shape.empty()) {
    MS_LOG(EXCEPTION) << "Current tensor is a tuple of tensor, but get a empty shape!";
  }
  if (shape[kIndex0] <= 0) {
    MS_LOG(EXCEPTION) << shape << " is an invalid shape, please check op infer!";
  }

  std::vector<mindspore::kernel::KernelTensorPtr> res;

  auto split_num = shape[kIndex0];
  auto offset = tensor->size() / split_num;
  auto new_shape = shape;
  new_shape.erase(new_shape.begin());

  for (int i = 0; i < split_num; ++i) {
    auto new_tensor = std::make_shared<mindspore::kernel::KernelTensor>(*tensor);
    new_tensor->SetType(std::make_shared<TensorType>(TypeIdToType(tensor->dtype_id())));
    auto tensor_shape = std::make_shared<abstract::TensorShape>();
    tensor_shape->SetShapeVector(new_shape);
    new_tensor->SetShape(tensor_shape);
    new_tensor->set_device_ptr(
      reinterpret_cast<void *>(reinterpret_cast<uint8_t *>(tensor->device_ptr()) + offset * i));
    new_tensor->set_size(offset);
    (void)res.emplace_back(new_tensor);
  }
  return res;
}

inline void Release(aclTensor *p) {
  static const auto aclDestroyTensor = GET_OP_API_FUNC(aclDestroyTensor);
  if (aclDestroyTensor == nullptr) {
    return;
  }
  aclDestroyTensor(p);
}

inline void Release(aclScalar *p) {
  static const auto aclDestroyScalar = GET_OP_API_FUNC(aclDestroyScalar);
  if (aclDestroyScalar == nullptr) {
    return;
  }
  aclDestroyScalar(p);
}

inline void Release(aclIntArray *p) {
  static const auto aclDestroyIntArray = GET_OP_API_FUNC(aclDestroyIntArray);
  if (aclDestroyIntArray == nullptr) {
    return;
  }

  aclDestroyIntArray(p);
}

inline void Release(aclBoolArray *p) {
  static const auto aclDestroyBoolArray = GET_OP_API_FUNC(aclDestroyBoolArray);
  if (aclDestroyBoolArray == nullptr) {
    return;
  }

  aclDestroyBoolArray(p);
}

inline void Release(aclFloatArray *p) {
  static const auto aclDestroyFloatArray = GET_OP_API_FUNC(aclDestroyFloatArray);
  if (aclDestroyFloatArray == nullptr) {
    return;
  }

  aclDestroyFloatArray(p);
}

inline void Release(aclTensorList *p) {
  static const auto aclDestroyTensorList = GET_OP_API_FUNC(aclDestroyTensorList);
  if (aclDestroyTensorList == nullptr) {
    return;
  }

  aclDestroyTensorList(p);
}

template <typename T>
void Release(T value) {
  (void)value;
}

template <typename Tuple, size_t... I>
void CallRelease(Tuple t, std::index_sequence<I...>) {
  (void)std::initializer_list<int>{(Release(std::get<I>(t)), 0)...};
}

template <typename Tuple>
void ReleaseConvertTypes(const Tuple &t) {
  static constexpr auto size = std::tuple_size<Tuple>::value;
  CallRelease(t, std::make_index_sequence<size>{});
}

inline ShapeVector UpdateOutputShape(const aclTensor *tensor) {
  MS_EXCEPTION_IF_NULL(tensor);
  static const auto op_api_func = GetOpApiFunc("aclGetViewShape");
  if (op_api_func == nullptr) {
    MS_LOG(EXCEPTION) << "aclGetViewShape not in " << GetOpApiLibName() << ", please check!";
  }
  using aclGetViewShapeFunc = int (*)(const aclTensor *tensor, int64_t **view_dims, uint64_t *view_dims_num);
  auto aclGetViewShape = reinterpret_cast<aclGetViewShapeFunc>(op_api_func);
  int64_t *view_dims = nullptr;
  uint64_t view_dim_num = 0;
  auto ret = aclGetViewShape(tensor, &view_dims, &view_dim_num);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "aclGetViewShape failed!";
  }
  ShapeVector output_shape(view_dims, view_dims + view_dim_num);
  delete[] view_dims;
  view_dims = nullptr;
  return output_shape;
}

inline void GetShape(aclTensor *tensor, std::vector<ShapeVector> *shape_list) {
  shape_list->emplace_back(UpdateOutputShape(tensor));
}

template <typename T>
void GetShape(T param, std::vector<ShapeVector> *shape_list) {
  shape_list->emplace_back(ShapeVector());
}

template <typename Tuple, size_t... I>
void CallFillShape(std::vector<ShapeVector> *shape_list, Tuple t, std::index_sequence<I...>) {
  (void)std::initializer_list<int>{(GetShape(std::get<I>(t), shape_list), 0)...};
}

template <typename Tuple>
std::vector<ShapeVector> FillShapeListFromTuple(const Tuple &t) {
  static constexpr auto size = std::tuple_size<Tuple>::value;
  std::vector<ShapeVector> shape_list;
  CallFillShape(&shape_list, t, std::make_index_sequence<size>{});
  return shape_list;
}

inline void UpdateAddress(aclOpExecutor *executor, aclTensor *tensor, const std::vector<void *> &address, size_t *idx) {
  MS_EXCEPTION_IF_NULL(executor);
  static const auto aclSetTensorAddr = GET_OP_API_FUNC(aclSetTensorAddr);
  if (aclSetTensorAddr == nullptr) {
    MS_LOG(EXCEPTION) << "aclSetTensorAddr is nullptr";
    return;
  }

  if (address.size() < 1) {
    MS_LOG(DEBUG) << "UpdateAddress when address list size is: " << address.size();
    return;
  }
  if (address[0] == nullptr) {
    MS_LOG(DEBUG) << "Optional input's address of index " << idx;
    (*idx)++;
    return;
  }
  aclSetTensorAddr(executor, *idx, tensor, address[0]);
  MS_LOG(DEBUG) << "aclSetTensorAddr, idx " << *idx << " address " << address[0];
  (*idx)++;
}

inline void UpdateAddress(aclOpExecutor *executor, aclTensorList *tensors, const std::vector<void *> &address,
                          size_t *idx) {
  MS_EXCEPTION_IF_NULL(executor);
  MS_EXCEPTION_IF_NULL(tensors);
  static const auto aclSetDynamicTensorAddr = GET_OP_API_FUNC(aclSetDynamicTensorAddr);
  if (aclSetDynamicTensorAddr == nullptr) {
    MS_LOG(EXCEPTION) << "aclSetDynamicTensorAddr is nullptr";
    return;
  }
  for (size_t i = 0; i < address.size(); ++i) {
    aclSetDynamicTensorAddr(executor, *idx, i, tensors, address[i]);
    MS_LOG(DEBUG) << "aclSetDynamicTensorAddr, idx " << *idx << ", i " << i << " address " << address[i];
  }
  (*idx)++;
}

template <typename T>
inline void UpdateAddress(aclOpExecutor *, T, const std::vector<void *> &c, size_t *idx) {
  if (!c.empty()) {
    MS_LOG(DEBUG) << "UpdateAddress, optional tensor idx: " << *idx;
    (*idx)++;
  }
}

template <typename Tuple, size_t... I>
void CallUpdate(aclOpExecutor *executor, const std::vector<std::vector<void *>> &address_list, Tuple t,
                std::index_sequence<I...>) {
  size_t valid_index = 0;
  (void)std::initializer_list<int>{(UpdateAddress(executor, std::get<I>(t), address_list[I], &valid_index), 0)...};
}

template <typename Tuple>
void UpdateAddressForTensor(aclOpExecutor *executor, const std::vector<std::vector<void *>> &address_list,
                            const Tuple &t) {
  static constexpr auto size = std::tuple_size<Tuple>::value - kIndex2;
  CallUpdate(executor, address_list, t, std::make_index_sequence<size>{});
}

// return a Scalar with the input type
#define MAKE_SCALAR(num, typeid, out)                                       \
  switch (typeid) {                                                         \
    case kNumberTypeFloat32: {                                              \
      out = std::make_shared<FP32Imm>(static_cast<float>(num));             \
      break;                                                                \
    }                                                                       \
    case kNumberTypeFloat16: {                                              \
      out = std::make_shared<FP32Imm>(static_cast<float>(num));             \
      break;                                                                \
    }                                                                       \
    case kNumberTypeFloat64: {                                              \
      out = std::make_shared<FP64Imm>(static_cast<double>(num));            \
      break;                                                                \
    }                                                                       \
    case kNumberTypeInt8: {                                                 \
      out = std::make_shared<Int8Imm>(static_cast<int8_t>(num));            \
      break;                                                                \
    }                                                                       \
    case kNumberTypeInt16: {                                                \
      out = std::make_shared<Int16Imm>(static_cast<int16_t>(num));          \
      break;                                                                \
    }                                                                       \
    case kNumberTypeInt32: {                                                \
      out = std::make_shared<Int32Imm>(static_cast<int>(num));              \
      break;                                                                \
    }                                                                       \
    case kNumberTypeInt64: {                                                \
      out = std::make_shared<Int64Imm>(static_cast<int64_t>(num));          \
      break;                                                                \
    }                                                                       \
    case kNumberTypeBool: {                                                 \
      out = std::make_shared<BoolImm>(static_cast<bool>(num));              \
      break;                                                                \
    }                                                                       \
    case kNumberTypeUInt8: {                                                \
      out = std::make_shared<UInt8Imm>(static_cast<uint8_t>(num));          \
      break;                                                                \
    }                                                                       \
    case kNumberTypeBFloat16: {                                             \
      out = std::make_shared<BF16Imm>(static_cast<bfloat16>(num));          \
      break;                                                                \
    }                                                                       \
    default: {                                                              \
      MS_LOG(EXCEPTION) << "Not support typeid " << TypeIdToString(typeid); \
    }                                                                       \
  }

}  // namespace  mindspore::device::ascend
#endif  // MINDSPORE_CCSRC_TRANSFORM_ACL_IR_OP_API_CONVERT_H_
