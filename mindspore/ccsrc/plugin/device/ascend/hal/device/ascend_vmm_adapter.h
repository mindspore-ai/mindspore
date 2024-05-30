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

#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_ASCEND_VMM_ADAPTER_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_ASCEND_VMM_ADAPTER_H_

#include <atomic>
#include <memory>
#include <map>
#include <vector>
#include <string>
#include <queue>

#include "acl/acl.h"
#include "utils/dlopen_macro.h"
#include "utils/hash_map.h"
#include "utils/log_adapter.h"

#include "include/backend/mem_reuse/mem_dynamic_allocator.h"
#include "plugin/device/ascend/hal/common/ascend_utils.h"

namespace mindspore {
namespace device {
namespace ascend {
class AscendVmmAdapter {
 public:
  static AscendVmmAdapter &GetInstance() {
    static AscendVmmAdapter instance{};
    return instance;
  }

  AscendVmmAdapter() {
    alloc_conf_map_ = parseAllocConf();
    for (auto &kv : alloc_conf_map_) {
      MS_LOG(INFO) << "VMM alloc conf key: " << kv.first << ", value: " << kv.second;
    }
    auto align_size = alloc_conf_map_["vmm_align_size"];
    if (align_size.empty()) {
      kVmmAlignSize = kDefaultAlignSize;
    } else {
      kVmmAlignSize = StringToMB(align_size) * kMB;
      if (kVmmAlignSize % kDefaultAlignSize != 0) {
        MS_LOG(EXCEPTION) << "VMM align size must be multiple of 2MB, but got " << kVmmAlignSize;
      }
    }
  }
  ~AscendVmmAdapter();

 public:
  size_t GetRoundUpAlignSize(size_t input_size) const;
  size_t GetRoundDownAlignSize(size_t input_size) const;

  void ClearAllMemory();
  size_t AllocDeviceMem(size_t size, DeviceMemPtr *addr);
  size_t MmapDeviceMem(const size_t size, const DeviceMemPtr addr);
  size_t EagerFreeDeviceMem(const DeviceMemPtr addr, const size_t size);

  const bool IsEnabled() const {
    static bool is_vmm_enabled =
      alloc_conf_map_.find("enable_vmm") != alloc_conf_map_.end() &&
      (alloc_conf_map_.at("enable_vmm") == "True" || alloc_conf_map_.at("enable_vmm") == "true");
    return is_vmm_enabled;
  }

 private:
  uint64_t kVmmAlignSize;
  DeviceMemPtr FindVmmSegment(const DeviceMemPtr addr);
  size_t GetHandleSize(size_t input_size);
  std::map<DeviceMemPtr, aclrtDrvMemHandle> vmm_map_;
  std::vector<DeviceMemPtr> all_reserve_mems_;
  std::queue<aclrtDrvMemHandle> handle_queue_;
  std::map<std::string, std::string> alloc_conf_map_;
  static constexpr uint64_t kMB = 1024 * 1024;
  static constexpr uint64_t kDefaultAlignSize = 2 * kMB;
  static int StringToMB(const std::string &str) {
    std::stringstream ss(str);
    int num;
    std::string unit;
    if (!(ss >> num)) {
      MS_LOG(EXCEPTION) << "No valid number could be extracted from the string, " << str;
    }
    if (!(ss >> unit) || unit != "MB") {
      MS_LOG(EXCEPTION) << "The unit of the string is not MB, " << str;
    }
    if (ss.rdbuf()->in_avail() > 0) {
      MS_LOG(EXCEPTION) << "The string has extra characters, " << str;
    }
    return num;
  }
  static std::map<std::string, std::string> parseAllocConf() {
    std::map<std::string, std::string> result;
    auto env_value = common::GetEnv("MS_ALLOC_CONF");
    if (env_value.empty()) {
      return result;
    }
    std::stringstream ss(env_value);
    std::string item;

    while (std::getline(ss, item, ',')) {
      std::size_t delimiterPos = item.find(':');
      if (delimiterPos != std::string::npos) {
        std::string key = item.substr(0, delimiterPos);
        std::string value = item.substr(delimiterPos + 1);
        result[key] = value;
      }
    }
    return result;
  }
};
}  // namespace ascend
}  // namespace device
}  // namespace mindspore

#endif
