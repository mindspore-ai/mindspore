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
#include "runtime/device/cpu/cpu_device_address.h"
#include <vector>
#include <memory>
#include "runtime/device/convert_tensor_utils.h"
#include "runtime/hardware/cpu/cpu_memory_pool.h"
#ifndef ENABLE_SECURITY
#include "debug/data_dump/dump_json_parser.h"
#endif

namespace mindspore {
namespace device {
namespace cpu {
CPUDeviceAddress::~CPUDeviceAddress() { ClearDeviceMemory(); }

void CPUDeviceAddress::ClearDeviceMemory() {
  if (ptr_ == nullptr) {
    return;
  }
  if (from_mem_pool_) {
    CPUMemoryPool::GetInstance().FreeTensorMem(ptr_);
    ptr_ = nullptr;
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
      MS_LOG(INFO) << "No need sync, host size: " << size << ", device size: " << size_;
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
      MS_LOG(WARNING) << "No need sync, host size: " << size << ", device size: " << size_;
      return true;
    }
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
}  // namespace cpu
}  // namespace device
}  // namespace mindspore
