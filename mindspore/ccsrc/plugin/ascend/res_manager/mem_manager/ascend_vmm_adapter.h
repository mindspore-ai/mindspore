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
#include <set>
#include <string>
#include <fstream>
#include <iostream>

#include "acl/acl.h"
#include "ops_utils/op_constants.h"
#include "utils/log_adapter.h"

#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "include/runtime/memory/mem_pool/mem_dynamic_allocator.h"
#include "include/utils/utils.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace device {
namespace ascend {
/// @brief Check if enable_mem_huge_1g environment variable is enabled.
/// @return True if enable_mem_huge_1g is enabled, allocate 1G physical memory each time, false otherwise.
inline bool IsEnableMemHuge1G() {
  static const bool is_enable_mem_huge_1g =
    memory::mem_pool::IsEnableAllocConfig(memory::mem_pool::kAllocEnableMemHuge1G);
  return is_enable_mem_huge_1g;
}

class AscendVmmAdapter {
 public:
  static AscendVmmAdapter &GetInstance() {
    static AscendVmmAdapter instance{};
    return instance;
  }

  AscendVmmAdapter() {
    auto align_size = memory::mem_pool::GetAllocConfigValue(memory::mem_pool::kAllocVmmAlignSize);
    enable_mem_huge_1g_ = IsEnableMemHuge1G();
    if (enable_mem_huge_1g_) {
      vmm_align_size_ = kGB;
    } else if (align_size.empty()) {
      vmm_align_size_ = kDefaultAlignSize;
    } else {
      vmm_align_size_ = StringToMB(align_size) * kMB;
      if (vmm_align_size_ % kDefaultAlignSize != 0) {
        MS_LOG(EXCEPTION) << "VMM align size must be multiple of 2MB, but got " << vmm_align_size_;
      }
    }
    MS_LOG(INFO) << "VMM align size is " << vmm_align_size_;
  }
  ~AscendVmmAdapter() = default;

  size_t GetRoundUpAlignSize(size_t input_size) const;
  size_t GetRoundDownAlignSize(size_t input_size) const;

  void ClearAllMemory();
  size_t AllocDeviceMem(size_t size, DeviceMemPtr *addr);
  size_t MmapDeviceMem(const size_t size, const DeviceMemPtr addr, const size_t max_size);
  size_t EagerFreeDeviceMem(const DeviceMemPtr addr, const size_t size);
  size_t GetAllocatedSize() { return physical_handle_size_ * vmm_align_size_; }

  size_t EmptyCache();

  static const bool IsEnabled() {
    static bool is_enable_vmm = IsVmmEnabled();
    return is_enable_vmm;
  }

 private:
  /// Allocates physical memory using ACL interface.
  ///
  /// When enable_mem_huge_1g_ is true, use ACL_HBM_MEM_HUGE1G to allocate 1G physical
  /// memory each time. In case that there is no continuous 1G memory, the memory will be
  /// allocated in 2M huge page size if the allocation fails.
  ///
  /// @param handle Memory handle that will be assigned the allocated memory handle.
  /// @param size Size of memory to allocate in bytes.
  /// @param prop Properties for allocation.
  /// @param flags Allocation flags.
  /// @param mapped_vmm_handle Mapping between virtual memory address and physical memory handle.
  /// @return bool, true if allocation succeeds, false otherwise.
  bool AllocPhysicalMem(aclrtDrvMemHandle *handle, size_t size, aclrtPhysicalMemProp *prop, uint64_t flags,
                        std::map<DeviceMemPtr, aclrtDrvMemHandle> &mapped_vmm_handle);

  static const bool IsVmmEnabled() {
    auto ctx = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(ctx);
    if (AnfAlgo::IsBackendGe() && IsDisableGeKernel()) {
      MS_LOG(INFO) << "Jit level is O2, vmm is disabled.";
      return false;
    }

    if (common::IsCompileSimulation()) {
      MS_LOG(INFO) << "Dry run, vmm is disabled.";
      return false;
    }

    if (memory::mem_pool::IsEnableAllocConfig(memory::mem_pool::kAllocEnableVmm)) {
      MS_LOG(INFO) << "VMM is explicitly enabled.";
      return true;
    }

    if (memory::mem_pool::IsDisableAllocConfig(memory::mem_pool::kAllocEnableVmm)) {
      MS_LOG(INFO) << "VMM is explicitly disabled.";
      return false;
    }

    const auto &soc_version = ctx->ascend_soc_version();
    if (!(soc_version == "ascend910b" || soc_version == "ascend910_93")) {
      MS_LOG(INFO) << "Soc is neither ascend910b nor ascend910_93, vmm is disabled by default.";
      return false;
    }

    if (!CheckVmmDriverVersion()) {
      return false;
    }

    MS_LOG(INFO) << "VMM is enabled.";
    return true;
  }

  bool enable_mem_huge_1g_;
  uint64_t vmm_align_size_;
  DeviceMemPtr FindVmmSegment(const DeviceMemPtr addr);
  size_t GetHandleSize(size_t input_size);
  std::atomic<size_t> physical_handle_size_{0};
  std::map<DeviceMemPtr, aclrtDrvMemHandle> vmm_map_;
  std::vector<DeviceMemPtr> all_reserve_mems_;
  std::set<aclrtDrvMemHandle> cached_handle_sets_;
  static constexpr uint64_t kMB = 1024 * 1024;
  static constexpr uint64_t kGB = 1024 * kMB;
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
  static bool CheckVmmDriverVersion() {
    // Get driver version
    constexpr auto ascend_install_info = "/etc/ascend_install.info";
    const std::string DRIVER_INSTALL_PATH_PARAM = "Driver_Install_Path_Param=";
    std::string driver_path = "/usr/local/Ascend";

    std::ifstream ascend_install_file(ascend_install_info);
    if (!ascend_install_file.is_open()) {
      MS_LOG(WARNING) << "Open file " << ascend_install_info << " failed.";
    } else {
      std::string line;
      while (std::getline(ascend_install_file, line)) {
        size_t pos = line.find(DRIVER_INSTALL_PATH_PARAM);
        if (pos != std::string::npos) {
          // Extract the path after "Driver_Install_Path_Param="
          driver_path = line.substr(pos + DRIVER_INSTALL_PATH_PARAM.length());
          MS_LOG(INFO) << "Driver path is " << driver_path;
          break;
        }
      }
    }

    auto splitString = [](const std::string &str, char delimiter) -> std::vector<std::string> {
      std::vector<std::string> tokens;
      std::string token;
      std::istringstream tokenStream(str);
      while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
      }
      return tokens;
    };

    auto driver_version_info = driver_path + "/driver/version.info";
    const std::string DRIVER_VERSION_PARAM = "Version=";
    std::ifstream driver_version_file(driver_version_info);
    if (!driver_version_file.is_open()) {
      MS_LOG(WARNING) << "Open file " << driver_version_info << " failed.";
    } else {
      std::string line;
      while (std::getline(driver_version_file, line)) {
        size_t pos = line.find(DRIVER_VERSION_PARAM);
        if (pos != std::string::npos) {
          // Extract the version after "Version="
          std::string driver_version = line.substr(pos + DRIVER_VERSION_PARAM.length());
          auto split_version = splitString(driver_version, '.');
          MS_LOG(INFO) << "Driver version is " << driver_version << ", major version is " << split_version[0];
          if (split_version[0] < "24") {
            MS_LOG(WARNING) << "Driver version is less than 24.0.0, vmm is disabled by default, drvier_version: "
                            << driver_version;
            return false;
          }
          break;
        }
      }
    }
    return true;
  }
};
}  // namespace ascend
}  // namespace device
}  // namespace mindspore

#endif
