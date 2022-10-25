/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#include "plugin/device/cpu/hal/hardware/cpu_memory_pool.h"
#ifndef ENABLE_SECURITY
#include "debug/data_dump/dump_json_parser.h"
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
}  // namespace
CPUDeviceAddress::~CPUDeviceAddress() { DoClearDeviceMemory(); }

void CPUDeviceAddress::DoClearDeviceMemory() {
  if (ptr_ == nullptr) {
    return;
  }
  if (from_mem_pool_) {
    CPUMemoryPool::GetInstance().FreeTensorMem(ptr_);
    ptr_ = nullptr;
  }
}

void CPUDeviceAddress::ClearDeviceMemory() { DoClearDeviceMemory(); }

bool CPUDeviceAddress::DumpMemToFile(const std::string &filepath, const std::string &, const ShapeVector &host_shape,
                                     TypeId host_type, bool) const {
  bool ret = false;
#ifndef ENABLE_SECURITY
  if (filepath.empty()) {
    MS_LOG(ERROR) << "Dump file path is null!";
    return ret;
  }
  std::string path = filepath + '.' + format_;
  MS_LOG(DEBUG) << "E2E Dump path is " << path;
  ret = DumpJsonParser::DumpToFile(path, ptr_, size_, host_shape, host_type);
#endif
  return ret;
}

bool CPUDeviceAddress::SyncDeviceToHost(const ShapeVector &, size_t size, TypeId type, void *host_ptr) const {
  // The input or output may be empty.
  if ((size == 0) || (size_ == 0)) {
    MS_LOG(INFO) << "No need sync, host size: " << size << ", device size: " << size_;
    return true;
  }
  if (ptr_ == nullptr) {
    MS_LOG(ERROR) << "The pointer ptr_ is null!";
    return false;
  }
  if (host_ptr == ptr_) {
    MS_LOG(DEBUG) << "host_ptr is equal to ptr_, request ignored.";
    return true;
  }

  if (type == type_id_) {
    if (size > size_) {
      MS_LOG(WARNING) << "Please check whether need sync data, host size: " << size << ", device size: " << size_;
      return true;
    }
    auto ret_code = memcpy_s(host_ptr, size, ptr_, size);
    // Return ERANGE when the copy size is larger than SECUREC_MEM_MAX_LEN.
    if (ret_code == ERANGE) {
      ConvertSameType(host_ptr, ptr_, size, type);
    } else if (ret_code != EOK) {
      MS_LOG(ERROR) << "Failed to copy tensor!";
      return false;
    } else {
      return true;
    }
  } else if (type == kNumberTypeFloat16 && type_id_ == kNumberTypeFloat32) {
    FloatToHalf(host_ptr, ptr_, size >> 1);
  } else if (type == kNumberTypeFloat64 && type_id_ == kNumberTypeFloat32) {
    FloatToDouble(host_ptr, ptr_, size / sizeof(double));
  } else if (type == kNumberTypeFloat32 && type_id_ == kNumberTypeFloat64) {
    DoubleToFloat(host_ptr, ptr_, size >> 1);
  } else if (type == kNumberTypeInt16 && type_id_ == kNumberTypeInt32) {
    IntToShort(host_ptr, ptr_, size >> 1);
  } else if (type == kNumberTypeInt64 && type_id_ == kNumberTypeInt32) {
    IntToLong(host_ptr, ptr_, size / sizeof(int64_t));
  } else {
    MS_LOG(ERROR) << "Types not match. Device type: " << TypeIdLabel(type_id_) << ", host type: " << TypeIdLabel(type)
                  << "!";
    return false;
  }
  return true;
}

bool CPUDeviceAddress::SyncHostToDevice(const ShapeVector &, size_t size, TypeId type, const void *host_ptr,
                                        const std::string &) const {
  // The input or output may be empty.
  if ((size == 0) || (size_ == 0)) {
    MS_LOG(INFO) << "No need sync, host size: " << size << ", device size: " << size_;
    return true;
  }
  if (ptr_ == nullptr) {
    MS_LOG(ERROR) << "The pointer ptr_ is null!";
    return false;
  }
  if (host_ptr == ptr_) {
    MS_LOG(DEBUG) << "host_ptr is equal to ptr_, request ignored.";
    return true;
  }

  if (type == type_id_) {
    if (size > size_) {
      MS_LOG(WARNING) << "Please check whether need sync data, host size: " << size << ", device size: " << size_;
      return true;
    }

    // If the value of host is a scalar type, then the host addr is a temporary address, which will be released after
    // the sync ends. Therefore, if the value is a string type or whose length is less than 16, it needs to be copied.
#ifndef __APPLE__
    const size_t kCopySize = 16;
    if (size <= kCopySize || type == kObjectTypeString) {
      return ((memcpy_s(ptr_, size, host_ptr, size) != EOK) ? false : true);
    }
#endif

    // Use the tensor host ptr to set the device ptr.
    if (from_mem_pool_) {
      CPUMemoryPool::GetInstance().FreeTensorMem(ptr_);
      from_mem_pool_ = false;
    }
    ptr_ = const_cast<void *>(host_ptr);
    original_ref_count_ = SIZE_MAX;
    ref_count_ = SIZE_MAX;
  } else if (type_id_ == kNumberTypeFloat32 && type == kNumberTypeFloat16) {
    HalfToFloat(ptr_, host_ptr, size >> 1);
  } else if (type_id_ == kNumberTypeFloat32 && type == kNumberTypeFloat64) {
    DoubleToFloat(ptr_, host_ptr, size / sizeof(double));
  } else if (type_id_ == kNumberTypeInt32 && type == kNumberTypeInt16) {
    ShortToInt(ptr_, host_ptr, size >> 1);
  } else if (type_id_ == kNumberTypeInt32 && type == kNumberTypeInt64) {
    LongToInt(ptr_, host_ptr, size / sizeof(int64_t));
  } else {
    MS_LOG(ERROR) << "Types not match. Device type: " << TypeIdLabel(type_id_) << ", host type: " << TypeIdLabel(type)
                  << "!";
    return false;
  }
  return true;
}

bool CPUDeviceAddress::SyncDeviceToDevice(const DeviceSync *src_device_addr) const {
  MS_EXCEPTION_IF_NULL(src_device_addr);
  auto src_cpu_device = dynamic_cast<const CPUDeviceAddress *>(src_device_addr);
  MS_EXCEPTION_IF_NULL(src_cpu_device);
  auto src_size = src_cpu_device->GetSize();
  auto src_ptr = src_cpu_device->GetMutablePtr();
  auto src_type = src_cpu_device->type_id();

  // The input or output may be empty.
  if ((src_size == 0) || (size_ == 0)) {
    MS_LOG(INFO) << "No need sync, src device size: " << src_size << ", dst device size: " << size_;
    return true;
  }
  MS_EXCEPTION_IF_NULL(src_ptr);
  MS_EXCEPTION_IF_NULL(ptr_);
  if (src_type == type_id_) {
    return CopySameTypeMem(ptr_, size_, src_ptr, src_size, src_type);
  } else if (type_id_ == kNumberTypeFloat32 && src_type == kNumberTypeFloat16) {
    HalfToFloat(ptr_, src_ptr, src_size >> 1);
  } else if (type_id_ == kNumberTypeFloat16 && src_type == kNumberTypeFloat32) {
    FloatToHalf(ptr_, src_ptr, src_size / sizeof(float));
  } else if (type_id_ == kNumberTypeFloat32 && src_type == kNumberTypeFloat64) {
    DoubleToFloat(ptr_, src_ptr, src_size / sizeof(double));
  } else if (type_id_ == kNumberTypeFloat64 && src_type == kNumberTypeFloat32) {
    FloatToDouble(ptr_, src_ptr, src_size / sizeof(float));
  } else if (type_id_ == kNumberTypeInt64 && src_type == kNumberTypeInt32) {
    IntToLong(ptr_, src_ptr, src_size / sizeof(int32_t));
  } else if (type_id_ == kNumberTypeInt32 && src_type == kNumberTypeInt64) {
    LongToInt(ptr_, src_ptr, src_size / sizeof(int64_t));
  } else {
    MS_LOG(ERROR) << "Types not match. Device type: " << TypeIdLabel(type_id_)
                  << ", host type: " << TypeIdLabel(src_type) << "!";
    return false;
  }
  return true;
}
}  // namespace cpu
}  // namespace device
}  // namespace mindspore
