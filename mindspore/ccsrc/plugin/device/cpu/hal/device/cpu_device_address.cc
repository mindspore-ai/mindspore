/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include <vector>
#include <memory>
#include "runtime/device/convert_tensor_utils.h"
#include "runtime/hardware/device_context_manager.h"
#include "plugin/device/cpu/hal/hardware/cpu_memory_pool.h"
#include "plugin/device/cpu/hal/device/cpu_hash_table_util.h"
#ifndef ENABLE_SECURITY
#include "include/backend/debug/data_dump/dump_json_parser.h"
#endif

namespace mindspore {
namespace device {
namespace cpu {
namespace {
bool CopySameTypeMem(void *dst_ptr, size_t dst_size, const void *src_ptr, size_t src_size, TypeId type) {
  if (src_size != dst_size) {
    MS_LOG(ERROR) << "The src device size is not equal of the dst device size, src device size: " << src_size
                  << ", dst device size: " << dst_size;
    return false;
  }

  auto ret = memcpy_s(dst_ptr, dst_size, src_ptr, src_size);
  // Return ERANGE when the copy size is larger than SECUREC_MEM_MAX_LEN.
  if (ret == ERANGE) {
    ConvertSameType(dst_ptr, src_ptr, dst_size, type);
    return true;
  } else if (ret != EOK) {
    MS_LOG(ERROR) << "Failed to copy tensor!";
    return false;
  } else {
    return true;
  }
}

// Synchronize user data from host to device.
bool SyncUserDataToDevice(const UserDataPtr &user_data, const void *host_ptr, size_t size) {
  MS_EXCEPTION_IF_NULL(user_data);
  MS_EXCEPTION_IF_NULL(host_ptr);
  const auto &user_data_type = user_data->get<UserDataType>(kUserDataType);
  MS_EXCEPTION_IF_NULL(user_data_type);

  if (*user_data_type == UserDataType::kUserTypeHashTable) {
    auto key_type = user_data->get<TypeId>(kHashTableKeyType);
    auto value_type = user_data->get<TypeId>(kHashTableValueType);
    MS_EXCEPTION_IF_NULL(key_type);
    MS_EXCEPTION_IF_NULL(value_type);
    const auto &iter = cpu_hash_table_funcs.find({*key_type, *value_type});
    if (iter != cpu_hash_table_funcs.end()) {
      // Import key, value, status tensors to CPU hash table.
      return std::get<kImportFuncIndex>(iter->second)(user_data, host_ptr, size);
    } else {
      MS_LOG(EXCEPTION) << "Unsupported hash table type, key type:" << TypeIdLabel(*key_type)
                        << ", value type:" << TypeIdLabel(*value_type);
    }
  }
  return true;
}
}  // namespace

void CPUDeviceAddress::SetDevicePtrDeleter() {
  const auto &kernel_tensor = this->kernel_tensor();
  if (!kernel_tensor) {
    return;
  }
  kernel_tensor->set_deleter([](void *ptr, bool from_mem_pool) {
    if (ptr != nullptr && from_mem_pool) {
      CPUMemoryPool::GetInstance().FreeTensorMem(ptr);
    }
  });
}

void CPUDeviceAddress::ClearDeviceMemory() {
  if (GetDevicePtr() == nullptr) {
    return;
  }
  if (from_mem_pool()) {
    CPUMemoryPool::GetInstance().FreeTensorMem(GetDevicePtr());
    SetDevicePtr(nullptr);
  }
}

void CPUDeviceAddress::ClearUserData() {
  if (user_data() == nullptr) {
    return;
  }
  auto user_data_type = user_data()->get<UserDataType>(kUserDataType);
  if (user_data_type == nullptr) {
    return;
  }
  if (*user_data_type == UserDataType::kUserTypeHashTable) {
    auto key_type = user_data()->get<TypeId>(kHashTableKeyType);
    auto value_type = user_data()->get<TypeId>(kHashTableValueType);
    MS_EXCEPTION_IF_NULL(key_type);
    MS_EXCEPTION_IF_NULL(value_type);
    const auto &iter = cpu_hash_table_funcs.find({*key_type, *value_type});
    if (iter != cpu_hash_table_funcs.end()) {
      // Clear CPU hash table.
      return std::get<kClearFuncIndex>(iter->second)(user_data());
    } else {
      MS_LOG(EXCEPTION) << "Unsupported hash table type:" << *key_type << " and:" << *value_type;
    }
  }
}

bool CPUDeviceAddress::DumpMemToFile(const std::string &filepath, const std::string &, const ShapeVector &host_shape,
                                     TypeId host_type, bool) const {
  bool ret = false;
#ifndef ENABLE_SECURITY
  if (filepath.empty()) {
    MS_LOG(ERROR) << "Dump file path is null!";
    return ret;
  }
  std::string path = filepath + '.' + format();
  MS_LOG(DEBUG) << "E2E Dump path is " << path;
  if (GetSize() == 0) {
    MS_LOG(INFO) << "Data size is 0 for file: " << path << ", no need to dump.";
    return true;
  }
  ret = DumpJsonParser::DumpToFile(path, GetDevicePtr(), GetSize(), host_shape, host_type);
#endif
  return ret;
}

bool CPUDeviceAddress::SyncDeviceToHost(const ShapeVector &, size_t size, TypeId type, void *host_ptr) const {
  // The input or output may be empty.
  if ((size == 0) || (GetSize() == 0)) {
    MS_LOG(INFO) << "No need sync, host size: " << size << ", device size: " << GetSize();
    return true;
  }
  MS_EXCEPTION_IF_NULL(GetDevicePtr());
  if (host_ptr == GetDevicePtr()) {
    MS_LOG(DEBUG) << "host_ptr is equal to device ptr, request ignored.";
    return true;
  }

  if (type == type_id()) {
    if (size > GetSize()) {
      MS_LOG(WARNING) << "Please check whether need sync data, host size: " << size << ", device size: " << GetSize();
      return true;
    }
    auto ret_code = memcpy_s(host_ptr, size, GetDevicePtr(), size);
    // Return ERANGE when the copy size is larger than SECUREC_MEM_MAX_LEN.
    if (ret_code == ERANGE) {
      ConvertSameType(host_ptr, GetDevicePtr(), size, type);
    } else if (ret_code != EOK) {
      MS_LOG(ERROR) << "Failed to copy tensor!";
      return false;
    } else {
      return true;
    }
  } else if (type == kNumberTypeFloat16 && type_id() == kNumberTypeFloat32) {
    FloatToHalf(host_ptr, GetDevicePtr(), size >> 1);
  } else if (type == kNumberTypeFloat64 && type_id() == kNumberTypeFloat32) {
    FloatToDouble(host_ptr, GetDevicePtr(), size / sizeof(double));
  } else if (type == kNumberTypeFloat32 && type_id() == kNumberTypeFloat64) {
    DoubleToFloat(host_ptr, GetDevicePtr(), size >> 1);
  } else if (type == kNumberTypeInt16 && type_id() == kNumberTypeInt32) {
    IntToShort(host_ptr, GetDevicePtr(), size >> 1);
  } else if (type == kNumberTypeInt64 && type_id() == kNumberTypeInt32) {
    IntToLong(host_ptr, GetDevicePtr(), size / sizeof(int64_t));
  } else {
    MS_LOG(ERROR) << "Types not match. Device type: " << TypeIdLabel(type_id()) << ", host type: " << TypeIdLabel(type)
                  << " device_address:" << this << " !";
    return false;
  }
  return true;
}

bool CPUDeviceAddress::SyncHostToDevice(const ShapeVector &, size_t size, TypeId type, const void *host_ptr,
                                        const std::string &) const {
  if (user_data() != nullptr && user_data()->has(kUserDataType)) {
    return SyncUserDataToDevice(user_data(), host_ptr, size);
  }

  // The input or output may be empty.
  if ((size == 0) || (GetSize() == 0)) {
    MS_LOG(INFO) << "No need sync, host size: " << size << ", device size: " << GetSize();
    return true;
  }
  if (GetDevicePtr() == nullptr) {
    MS_LOG(ERROR) << "The pointer device ptr is null!";
    return false;
  }
  if (host_ptr == GetDevicePtr()) {
    MS_LOG(DEBUG) << "host_ptr is equal to device ptr, request ignored.";
    return true;
  }

  if (type == type_id()) {
    if (size > GetSize()) {
      MS_LOG(WARNING) << "Please check whether need sync data, host size: " << size << ", device size: " << GetSize();
      return true;
    }

    // If the value of host is a scalar type, then the host addr is a temporary address, which will be released after
    // the sync ends. Therefore, if the value is a string type or whose length is less than 16, it needs to be copied.
    const size_t kCopySize = 16;
    if (size <= kCopySize || type == kObjectTypeString) {
      return ((memcpy_s(GetDevicePtr(), size, host_ptr, size) != EOK) ? false : true);
    }
    if (is_view_) {
      return CopySameTypeMem(GetDevicePtr(), size, host_ptr, size, type);
    }

    // Use the tensor host ptr to set the device ptr.
    if (from_mem_pool()) {
      CPUMemoryPool::GetInstance().FreeTensorMem(GetDevicePtr());
      set_from_mem_pool(false);
    }
    SetDevicePtr(const_cast<void *>(host_ptr));
    set_original_ref_count(SIZE_MAX);
    set_ref_count(SIZE_MAX);
  } else if (type_id() == kNumberTypeFloat32 && type == kNumberTypeFloat16) {
    HalfToFloat(GetDevicePtr(), host_ptr, size >> 1);
  } else if (type_id() == kNumberTypeFloat32 && type == kNumberTypeFloat64) {
    DoubleToFloat(GetDevicePtr(), host_ptr, size / sizeof(double));
  } else if (type_id() == kNumberTypeInt32 && type == kNumberTypeInt16) {
    ShortToInt(GetDevicePtr(), host_ptr, size >> 1);
  } else if (type_id() == kNumberTypeInt32 && type == kNumberTypeInt64) {
    LongToInt(GetDevicePtr(), host_ptr, size / sizeof(int64_t));
  } else {
    MS_LOG(ERROR) << "Types not match. Device type: " << TypeIdLabel(type_id()) << ", host type: " << TypeIdLabel(type)
                  << "!";
    return false;
  }
  return true;
}

bool CPUDeviceAddress::SyncDeviceToDevice(const DeviceSync *src_device_addr) const {
  MS_EXCEPTION_IF_NULL(src_device_addr);
  auto src_cpu_device = dynamic_cast<const CPUDeviceAddress *>(src_device_addr);
  MS_EXCEPTION_IF_NULL(src_cpu_device);
  return SyncDeviceToDevice(src_cpu_device->host_shape(), src_cpu_device->GetSize(), src_cpu_device->type_id(),
                            src_cpu_device->GetPtr(), src_cpu_device->format());
}

bool CPUDeviceAddress::SyncDeviceToDevice(const ShapeVector &, size_t size, TypeId type, const void *src_ptr,
                                          const std::string &format) const {
  MS_LOG(DEBUG) << "SyncDeviceToDevice, dst(format:" << DeviceAddress::format()
                << ", type_id:" << TypeIdLabel(type_id()) << ", size:" << GetSize() << "), src(format:" << format
                << ", type_id:" << TypeIdLabel(type) << ", size:" << size << ")";
  if (GetDevicePtr() == src_ptr) {
    MS_LOG(INFO) << "Dst addr is same with src addr, no need memcpy data.";
    return true;
  }
  // The input or output may be empty.
  if ((size == 0) || (GetSize() == 0)) {
    MS_LOG(INFO) << "No need sync, src device size: " << size << ", dst device size: " << GetSize();
    return true;
  }
  if (DeviceAddress::format() != format) {
    MS_LOG(ERROR) << "Format is different, src(format:" << format << "), dst(format:" << DeviceAddress::format()
                  << ").";
    return false;
  }

  MS_EXCEPTION_IF_NULL(src_ptr);
  MS_EXCEPTION_IF_NULL(GetDevicePtr());
  if (type == type_id()) {
    return CopySameTypeMem(GetDevicePtr(), size, src_ptr, size, type);
  } else if (type_id() == kNumberTypeFloat32 && type == kNumberTypeFloat16) {
    HalfToFloat(GetDevicePtr(), src_ptr, size >> 1);
  } else if (type_id() == kNumberTypeFloat16 && type == kNumberTypeFloat32) {
    FloatToHalf(GetDevicePtr(), src_ptr, size / sizeof(float));
  } else if (type_id() == kNumberTypeFloat32 && type == kNumberTypeFloat64) {
    DoubleToFloat(GetDevicePtr(), src_ptr, size / sizeof(double));
  } else if (type_id() == kNumberTypeFloat64 && type == kNumberTypeFloat32) {
    FloatToDouble(GetDevicePtr(), src_ptr, size / sizeof(float));
  } else if (type_id() == kNumberTypeInt64 && type == kNumberTypeInt32) {
    IntToLong(GetDevicePtr(), src_ptr, size / sizeof(int32_t));
  } else if (type_id() == kNumberTypeInt32 && type == kNumberTypeInt64) {
    LongToInt(GetDevicePtr(), src_ptr, size / sizeof(int64_t));
  } else {
    MS_LOG(ERROR) << "Types not match. Device type: " << TypeIdLabel(type_id()) << ", host type: " << TypeIdLabel(type)
                  << "!";
    return false;
  }
  return true;
}
}  // namespace cpu
}  // namespace device
}  // namespace mindspore
