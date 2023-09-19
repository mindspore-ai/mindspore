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

#ifndef MINDSPORE_CCSRC_TRANSFORM_ACL_IR_OP_API_CONVERT_H_
#define MINDSPORE_CCSRC_TRANSFORM_ACL_IR_OP_API_CONVERT_H_

#include <dlfcn.h>
#include <vector>
#include <string>
#include <algorithm>
#include <functional>
#include "acl/acl_base.h"
#include "ir/tensor.h"
#include "transform/acl_ir/acl_convert.h"
#include "plugin/device/ascend/hal/common/ascend_utils.h"
#include "plugin/device/ascend/hal/device/ascend_device_address.h"

namespace mindspore::transform {

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

// Get op api func.
inline std::string GetOpApiLibName() { return "/lib64/libopapi.so"; }

inline std::string GetCustOpApiLibName() { return "/op_api/lib/libcust_opapi.so"; }

inline void *GetOpApiFuncFromLib(void *handler, const char *lib_name, const char *api_name) {
  MS_EXCEPTION_IF_NULL(handler);
  auto func = dlsym(handler, api_name);
  if (func == nullptr) {
    MS_LOG(WARNING) << "Dlsym " << api_name << " from " << lib_name << " failed!" << dlerror();
  }
  return func;
}

inline void *GetOpApiLibHandler(const std::string &lib_path) {
  auto handler = dlopen(lib_path.c_str(), RTLD_LAZY);
  if (handler == nullptr) {
    MS_LOG(WARNING) << "Dlopen " << lib_path << " failed!" << dlerror();
  }
  return handler;
}

inline void *GetOpApiFunc(const char *api_name) {
  std::string cust_path = common::GetEnv("ASCEND_CUSTOM_OPP_PATH");
  if (!cust_path.empty()) {
    auto cust_lib_path = cust_path + GetCustOpApiLibName();
    static auto cust_handler = GetOpApiLibHandler(cust_lib_path);
    if (cust_handler != nullptr) {
      auto cust_func = GetOpApiFuncFromLib(cust_handler, cust_lib_path.c_str(), api_name);
      if (cust_func != nullptr) {
        return cust_func;
      }
    }
  }

  auto ascend_path = device::ascend::GetAscendPath();
  std::vector<std::string> depend_libs = {"libdummy_tls.so", "libnnopbase.so"};
  for (const auto &dep_lib : depend_libs) {
    (void)GetOpApiLibHandler(ascend_path + "lib64/" + dep_lib);
  }
  auto lib_path = ascend_path + GetOpApiLibName();
  static auto handle = GetOpApiLibHandler(lib_path);
  if (handle == nullptr) {
    return nullptr;
  }
  return GetOpApiFuncFromLib(handle, lib_path.c_str(), api_name);
}

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
class OpApiAttrConverter : public AttrHelper<OpApiAttrConverter> {
 public:
  template <typename T>
  void ConvertValue(const ValuePtr &value, const AttrDeclType<T> &, aclScalar *scalar) {
    auto real_val = GetValue<T>(value);
    MS_EXCEPTION_IF_NULL(scalar);
    static const auto aclCreateScalar = GET_OP_API_FUNC(aclCreateScalar);
    if (aclCreateScalar == nullptr) {
      scalar = nullptr;
    }
    scalar = aclCreateScalar(&real_val, GetDataType(value));
  }

  void ConvertValue(const ValuePtr &value, const AttrDeclType<int32_t> &, aclScalar *scalar) {
    auto real_val = static_cast<int64_t>(GetValue<int32_t>(value));
    MS_EXCEPTION_IF_NULL(scalar);
    static const auto aclCreateScalar = GET_OP_API_FUNC(aclCreateScalar);
    if (aclCreateScalar == nullptr) {
      scalar = nullptr;
    }
    scalar = aclCreateScalar(&real_val, ACL_INT64);
  }

  void ConvertValue(const ValuePtr &value, const AttrDeclType<std::vector<int64_t>> &, aclIntArray *array) {
    std::vector<int64_t> array_list;
    ConvertValueSequenceToList(value, &array_list);
    static const auto aclCreateIntArray = GET_OP_API_FUNC(aclCreateIntArray);
    if (aclCreateIntArray == nullptr) {
      array = nullptr;
    }
    array = aclCreateIntArray(array_list.data(), array_list.size());
  }

  void ConvertValue(const ValuePtr &value, const AttrDeclType<std::vector<int32_t>> &, aclIntArray *array) {
    std::vector<int32_t> array_list;
    ConvertValueSequenceToList(value, &array_list);
    std::vector<int64_t> array_list_int64;
    (void)std::transform(array_list.begin(), array_list.end(), std::back_inserter(array_list_int64),
                         [](const int val) { return IntToLong(val); });
    static const auto aclCreateIntArray = GET_OP_API_FUNC(aclCreateIntArray);
    if (aclCreateIntArray == nullptr) {
      array = nullptr;
    }
    array = aclCreateIntArray(array_list_int64.data(), array_list_int64.size());
  }

  void ConvertValue(const ValuePtr &value, const AttrDeclType<std::vector<uint8_t>> &, aclBoolArray *array) {
    std::vector<uint8_t> array_list;
    ConvertValueSequenceToList(value, &array_list);
    static const auto aclCreateBoolArray = GET_OP_API_FUNC(aclCreateBoolArray);
    if (aclCreateBoolArray == nullptr) {
      array = nullptr;
    }
    array = aclCreateBoolArray(reinterpret_cast<const bool *>(array_list.data()), array_list.size());
  }

  void ConvertValue(const ValuePtr &value, const AttrDeclType<std::vector<float>> &, aclFloatArray *array) {
    std::vector<float> array_list;
    ConvertValueSequenceToList(value, &array_list);
    static const auto aclCreateFloatArray = GET_OP_API_FUNC(aclCreateFloatArray);
    if (aclCreateFloatArray == nullptr) {
      array = nullptr;
    }
    array = aclCreateFloatArray(array_list.data(), array_list.size());
  }

 private:
  OpApiAttrConverter() = default;
  ~OpApiAttrConverter() = default;

  inline aclDataType GetDataType(const ValuePtr &value) { return AclConverter::ConvertType(value->type()->type_id()); }
};

inline aclTensor *ConvertType(const device::ascend::AscendDeviceAddressPtr &tensor) {
  static const auto aclCreateTensor = GET_OP_API_FUNC(aclCreateTensor);
  if (aclCreateTensor == nullptr) {
    return nullptr;
  }

  auto acl_data_type = AclConverter::ConvertType(tensor->type_id());
  auto shape = tensor->host_shape();
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

  // Create strides.
  auto strides = shape;
  if (!strides.empty()) {
    strides.erase(strides.begin());
  }
  strides.push_back(1);
  for (int i = static_cast<int>(strides.size()) - 2; i >= 0; i--) {
    strides[i] = strides[i] * strides[i + 1];
  }
  auto acl_tensor = aclCreateTensor(shape.data(), shape_size, acl_data_type, strides.data(), 0, format, shape.data(),
                                    shape_size, tensor->GetMutablePtr());
  return acl_tensor;
}

template <typename T>
T ConvertType(const ValuePtr &value) {
  MS_EXCEPTION_IF_NULL(value);
  OpApiAttrConverter op_api_attr_converter;
  T res;
  op_api_attr_converter.ConvertValueToRealType(value, "", res);
  return res;
}

template <typename T>
T ConvertType(T value) {
  return value;
}

template <typename... Ts>
constexpr auto ConvertTypes(Ts &... args) {
  return std::make_tuple(ConvertType(args)...);
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
}  // namespace mindspore::transform
#endif  // MINDSPORE_CCSRC_TRANSFORM_ACL_IR_OP_API_CONVERT_H_
