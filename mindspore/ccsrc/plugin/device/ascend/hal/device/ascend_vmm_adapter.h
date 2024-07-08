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
#include "utils/log_adapter.h"

#include "include/backend/mem_reuse/mem_dynamic_allocator.h"
#include "plugin/device/ascend/hal/common/ascend_utils.h"
#include "utils/ms_context.h"

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
    auto align_size = common::GetAllocConfigValue(common::kAllocVmmAlignSize);
    if (align_size.empty()) {
      kVmmAlignSize = kDefaultAlignSize;
    } else {
      kVmmAlignSize = StringToMB(align_size) * kMB;
      if (kVmmAlignSize % kDefaultAlignSize != 0) {
        MS_LOG(EXCEPTION) << "VMM align size must be multiple of 2MB, but got " << kVmmAlignSize;
      }
    }
    MS_LOG(INFO) << "VMM align size is " << kVmmAlignSize;
  }
  ~AscendVmmAdapter();

 public:
  size_t GetRoundUpAlignSize(size_t input_size) const;
  size_t GetRoundDownAlignSize(size_t input_size) const;

  void ClearAllMemory();
  size_t AllocDeviceMem(size_t size, DeviceMemPtr *addr);
  size_t MmapDeviceMem(const size_t size, const DeviceMemPtr addr, const size_t max_size);
  size_t EagerFreeDeviceMem(const DeviceMemPtr addr, const size_t size);

  static const bool IsEnabled() {
    static bool is_enable_vmm = IsVmmEnabled();
    return is_enable_vmm;
  }

 private:
  static const bool IsVmmEnabled() {
    auto ctx = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(ctx);
    if (ctx->GetJitLevel() == kAttrJitLevelO2) {
      MS_LOG(INFO) << "Jit level is O2, vmm is disabled.";
      return false;
    }

    if (common::IsEnableAlllocConfig(common::kAllocEnableVmm)) {
      MS_LOG(INFO) << "VMM is explicitly enabled.";
      return true;
    }

    if (common::IsDisableAlllocConfig(common::kAllocEnableVmm)) {
      MS_LOG(INFO) << "VMM is explicitly disabled.";
      return false;
    }

    const auto &soc_version = ctx->ascend_soc_version();
    if (!(soc_version == "ascend910b" || soc_version == "ascend910c")) {
      MS_LOG(INFO) << "Soc is neither ascend910b nor ascend910c, vmm is disabled by default.";
      return false;
    }

    // Not open vmm by default in PyNative mode
    bool is_enable_vmm = ctx->get_param<int>(MS_CTX_EXECUTION_MODE) != kPynativeMode;
    MS_LOG(INFO) << "VMM is " << (is_enable_vmm ? "enabled" : "disabled") << " by default.";
    return is_enable_vmm;
  }

 private:
  uint64_t kVmmAlignSize;
  DeviceMemPtr FindVmmSegment(const DeviceMemPtr addr);
  size_t GetHandleSize(size_t input_size);
  std::atomic<size_t> physical_handle_size_{0};
  std::map<DeviceMemPtr, aclrtDrvMemHandle> vmm_map_;
  std::vector<DeviceMemPtr> all_reserve_mems_;
  std::queue<aclrtDrvMemHandle> handle_queue_;
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
};
}  // namespace ascend
}  // namespace device
}  // namespace mindspore

#endif
