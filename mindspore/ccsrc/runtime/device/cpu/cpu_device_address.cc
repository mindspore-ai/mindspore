/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include "debug/data_dump/dump_json_parser.h"

namespace mindspore {
namespace device {
namespace cpu {
bool CPUDeviceAddress::DumpMemToFile(const std::string &filepath, const std::string &host_fmt,
                                     const ShapeVector &host_shape, TypeId host_type, bool trans_flag) const {
  bool ret = false;
  if (filepath.empty()) {
    MS_LOG(ERROR) << "Dump file path is null!";
    return ret;
  }
  std::string shape = "shape";
  if (host_shape.empty()) {
    shape += "_0";
  } else {
    for (auto &value : host_shape) {
      shape += '_' + std::to_string(value);
    }
  }
  std::string file_extension = ".bin";
  std::string path = filepath + '_' + shape + '_' + TypeIdToType(type_id_)->ToString() + '_' + format_ + file_extension;
  MS_LOG(DEBUG) << "E2E Dump path is " << path;
  auto host_tmp = std::vector<uint8_t>(size_);
  auto ret_code = memcpy_s(host_tmp.data(), size_, ptr_, size_);
  if (ret_code != EOK) {
    MS_LOG(ERROR) << "Failed to copy tensor!";
    return ret;
  }
  ret = DumpJsonParser::DumpToFile(path, host_tmp.data(), size_);
  return ret;
}

bool CPUDeviceAddress::SyncDeviceToHost(const ShapeVector & /*shape*/, size_t size, TypeId type, void *host_ptr) const {
  if (ptr_ == nullptr) {
    MS_LOG(ERROR) << "The pointer ptr_ is null!";
    return false;
  }
  if (host_ptr == ptr_) {
    MS_LOG(DEBUG) << "host_ptr is equal to ptr_, request ignored.";
    return true;
  }
  if (type == type_id_) {
    auto ret_code = memcpy_s(host_ptr, size, ptr_, size_);
    if (ret_code != EOK) {
      MS_LOG(ERROR) << "Failed to copy tensor!";
      return false;
    }
  } else if (type == kNumberTypeFloat16 && type_id_ == kNumberTypeFloat32) {
    FloatToHalf(host_ptr, ptr_, size / 2);
  } else if (type == kNumberTypeFloat64 && type_id_ == kNumberTypeFloat32) {
    FloatToDouble(host_ptr, ptr_, size / sizeof(double));
  } else if (type == kNumberTypeInt16 && type_id_ == kNumberTypeInt32) {
    IntToShort(host_ptr, ptr_, size / 2);
  } else if (type == kNumberTypeInt64 && type_id_ == kNumberTypeInt32) {
    IntToLong(host_ptr, ptr_, size / sizeof(int64_t));
  } else {
    MS_LOG(ERROR) << "Types not match. Device type: " << TypeIdLabel(type_id_) << ", host type: " << TypeIdLabel(type)
                  << "!";
    return false;
  }
  return true;
}

bool CPUDeviceAddress::SyncHostToDevice(const ShapeVector & /*shape*/, size_t size, TypeId type,
                                        const void *host_ptr) const {
  if (ptr_ == nullptr) {
    MS_LOG(ERROR) << "The pointer ptr_ is null!";
    return false;
  }
  if (host_ptr == ptr_) {
    MS_LOG(DEBUG) << "host_ptr is equal to ptr_, request ignored.";
    return true;
  }
  if (type_id_ == kNumberTypeFloat32 && type == kNumberTypeFloat16) {
    HalfToFloat(ptr_, host_ptr, size / 2);
  } else if (type_id_ == kNumberTypeFloat32 && type == kNumberTypeFloat64) {
    DoubleToFloat(ptr_, host_ptr, size / sizeof(double));
  } else if (type_id_ == kNumberTypeInt32 && type == kNumberTypeInt16) {
    ShortToInt(ptr_, host_ptr, size / 2);
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
