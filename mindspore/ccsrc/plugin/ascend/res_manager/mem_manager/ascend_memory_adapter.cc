/**
 * Copyright 2021-2024 Huawei Technologies Co., Ltd
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

#include "plugin/ascend/res_manager/mem_manager/ascend_memory_adapter.h"
#include "plugin/ascend/res_manager/device_context_conf/op_tuning_conf.h"
#include "plugin/ascend/res_manager/mem_manager/ascend_two_pointer_mem_adapter.h"
#include "plugin/ascend/res_manager/mem_manager/ascend_dynamic_mem_adapter.h"
#include "plugin/ascend/res_manager/mem_manager/ascend_gmem_adapter.h"
#include "plugin/ascend/res_manager/mem_manager/ascend_vmm_adapter.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_rt_symbol.h"
#include "plugin/ascend/res_manager/symbol_interface/symbol_utils.h"
#include "utils/ms_utils.h"
#include "include/runtime/utils/runtime_conf/runtime_conf.h"
#include "include/runtime/utils/runtime_conf/runtime_env.h"

namespace mindspore {
namespace device {
namespace ascend {
namespace {
constexpr uint64_t kAscendMemAlignSize = 512;
constexpr double kHalfRatio = 0.5;
constexpr double kMSMemoryRatio = 0.9375;           // 15/16
constexpr double kReservedMemoryRatio = 0.0625;     // 1/16
constexpr size_t kPerHugePageMemorySize = 2097152;  // 2mb
constexpr size_t kExtraReservedMemory = 10485760;   // 10mb
constexpr size_t kSimuHBMTotalMemSizeGB = 64;
}  // namespace
AscendMemAdapterPtr AscendMemAdapter::instance_ = nullptr;

AscendMemAdapterPtr AscendMemAdapter::GetInstance() {
  if (instance_ == nullptr) {
    auto op_tuning_conf = OpTuningConf::GetInstance();
    MS_EXCEPTION_IF_NULL(op_tuning_conf);
    if (IsDisableGeKernel() || common::IsCompileSimulation() || op_tuning_conf->EnableAoeOnline()) {
      // disable ge kernel or dry run.
      instance_ = std::make_shared<AscendTwoPointerMemAdapter>();
    } else {
      instance_ = std::make_shared<AscendDynamicMemAdapter>();
    }
  }
  return instance_;
}

size_t AscendMemAdapter::GetRoundDownAlignSize(size_t input_size) {
  return (input_size / kAscendMemAlignSize) * kAscendMemAlignSize;
}

size_t AscendMemAdapter::GetRoundUpAlignSize(size_t input_size) {
  return ((input_size + kAscendMemAlignSize - 1) / kAscendMemAlignSize) * kAscendMemAlignSize;
}

size_t AscendMemAdapter::GetDeviceMemSizeFromContext() const {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  size_t size_from_context;
  auto max_device_memory = runtime::RuntimeConf::GetInstance()->mem_max_size();
  float total_device_memory = 32.0f;
  if (context->ascend_soc_version() == kAscendVersion910b || context->ascend_soc_version() == kAscendVersion910_93) {
    total_device_memory = 64.0f;
  }
  if (context->ascend_soc_version() == kAscendVersion310p) {
    total_device_memory = 43.0f;
  }
  if (max_device_memory <= total_device_memory) {
    MS_LOG(INFO) << "context max_device_memory:" << max_device_memory;
    size_from_context = FloatToSize(max_device_memory * kGBToByte);
  } else {
    auto variable_memory_max_size = context->get_param<std::string>(MS_CTX_VARIABLE_MEMORY_MAX_SIZE);
    if (variable_memory_max_size == "0") {
      return 0;
    }
    MS_LOG(INFO) << "context variable_memory_max_size:" << variable_memory_max_size;
    auto pos = variable_memory_max_size.find('*');
    if (pos == std::string::npos) {
      MS_LOG(EXCEPTION) << "Invalid variable_memory_max_size";
    }
    auto gb_str = variable_memory_max_size.substr(0, pos);
    auto gb_var = std::stoull(gb_str);
    MS_LOG(INFO) << "variable_memory_max_size(GB):" << gb_var;
    size_from_context = gb_var * kGBToByte;
  }

  return size_from_context;
}

bool AscendMemAdapter::Initialize() {
  if (initialized_) {
    return true;
  }

  if (UseSimulationApi()) {
    SimulationInitialize();
    return true;
  }

  float huge_page_reserve_size = runtime::RuntimeConf::GetInstance()->mem_huge_page_reserve_size();
  device_hbm_huge_page_reserved_size_ = static_cast<size_t>(huge_page_reserve_size * kGBToByte);
  if (AscendVmmAdapter::IsEnabled() && device_hbm_huge_page_reserved_size_ > 0) {
    MS_LOG(WARNING) << "Reserve huge page feature is not available when VMM is enabled.";
  }
  MS_LOG(INFO) << "Config huge_page_reserve_size : " << huge_page_reserve_size
               << ", device_hbm_huge_page_reserved_size_ : " << device_hbm_huge_page_reserved_size_;

  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  auto ret = CALL_ASCEND_API(aclrtGetMemInfo, ACL_HBM_MEM, &device_hbm_free_size_, &device_hbm_total_size_);
  if (ret != ACL_SUCCESS || device_hbm_total_size_ == 0) {
    MS_LOG(EXCEPTION) << "Internal Error: Get Device MOC memory size failed, ret = " << ret
                      << ", total MOC size :" << device_hbm_total_size_;
  }

  if (device_hbm_free_size_ < LongToSize(DoubleToLong(device_hbm_total_size_ * kHalfRatio))) {
    unsigned int device_id = context_ptr->get_param<uint32_t>(MS_CTX_DEVICE_ID);
    MS_LOG(WARNING) << "Free memory size is less "
                       "than half of total memory size."
                    << "Device " << device_id << " Device MOC total size:" << device_hbm_total_size_
                    << " Device MOC free size:" << device_hbm_free_size_
                    << " may be other processes occupying this card, check as: ps -ef|grep python";
  }

  // get user define max backend memory
  auto user_define_ms_size = GetDeviceMemSizeFromContext();
  auto recommend_mem_size_for_others = LongToSize(DoubleToLong(device_hbm_free_size_ * kReservedMemoryRatio));
  size_t reserved_mem_size_for_others;
  if (user_define_ms_size == 0) {
    ms_used_hbm_size_ = DoubleToLong(device_hbm_free_size_ * kMSMemoryRatio);
    // sub the extra reserved 10mb after rounding down the 2mb
    ms_used_hbm_size_ = (ms_used_hbm_size_ / kPerHugePageMemorySize) * kPerHugePageMemorySize - kExtraReservedMemory;
    reserved_mem_size_for_others = device_hbm_free_size_ - SizeToLong(ms_used_hbm_size_);
  } else {
    if (user_define_ms_size >= device_hbm_free_size_) {
      static auto tft_env_value = common::GetEnv("MS_ENABLE_TFT");
      constexpr auto optARF = "ARF:1";
      constexpr auto optRSC = "RSC:1";
      if (tft_env_value.find(optARF) != std::string::npos || tft_env_value.find(optRSC) != std::string::npos) {
        MS_LOG(WARNING) << "#umsg#Framework Error Message:#umsg#The Free Device Memory Size is "
                        << (SizeToFloat(device_hbm_free_size_) / kGBToByte)
                        << " GB, max_device_memory should be in range (0-"
                        << (SizeToFloat(device_hbm_free_size_) / kMBToByte) << "]MB, but got "
                        << (SizeToFloat(user_define_ms_size) / kMBToByte)
                        << "MB, please set the context key max_device_memory in valid range.";
      } else {
        MS_LOG(EXCEPTION) << "#umsg#Framework Error Message:#umsg#The Free Device Memory Size is "
                          << (SizeToFloat(device_hbm_free_size_) / kGBToByte)
                          << " GB, max_device_memory should be in range (0-"
                          << (SizeToFloat(device_hbm_free_size_) / kMBToByte) << "]MB, but got "
                          << (SizeToFloat(user_define_ms_size) / kMBToByte)
                          << "MB, please set the context key max_device_memory in valid range.";
      }
    }
    ms_used_hbm_size_ = SizeToLong(user_define_ms_size);

    reserved_mem_size_for_others = device_hbm_total_size_ - LongToSize(ms_used_hbm_size_);
    if (reserved_mem_size_for_others < recommend_mem_size_for_others) {
      MS_LOG(WARNING) << "Reserved memory size for other components(" << reserved_mem_size_for_others
                      << ") is less than recommend size(" << recommend_mem_size_for_others
                      << "), It may lead to Out Of Memory in HCCL or other components, Please double check context key "
                         "'variable_memory_max_size'/'max_device_memory'";
    }
  }

  if (AscendVmmAdapter::IsEnabled()) {
    ms_used_hbm_size_ = SizeToLong(AscendVmmAdapter::GetInstance().GetRoundDownAlignSize(ms_used_hbm_size_));
  } else if (AscendGmemAdapter::GetInstance().is_eager_free_enabled()) {
    ms_used_hbm_size_ = SizeToLong(AscendGmemAdapter::GetInstance().GetRoundDownAlignSize(ms_used_hbm_size_));
  } else {
    ms_used_hbm_size_ = SizeToLong(GetRoundDownAlignSize(ms_used_hbm_size_));
  }
  max_available_ms_hbm_size_ = ms_used_hbm_size_;

  auto get_init_info = [this, &reserved_mem_size_for_others, &recommend_mem_size_for_others,
                        &user_define_ms_size]() -> std::string {
    std::ostringstream oss;
    oss << "Device MOC Size:" << device_hbm_total_size_ / kMBToByte
        << "M, Device free MOC Size:" << device_hbm_free_size_ / kMBToByte
        << "M, Reserved MOC size for Other Components(HCCL/rts/etc.):" << reserved_mem_size_for_others / kMBToByte
        << "M, Recommend Reserved MOC size for Other Components:" << recommend_mem_size_for_others / kMBToByte
        << "M, User define MindSpore MOC Size:" << user_define_ms_size / kGBToByte
        << "G, MindSpore Used MOC Size:" << ms_used_hbm_size_ / kMBToByte << "M.";
    return oss.str();
  };

  MS_LOG(INFO) << get_init_info();
  if (runtime::IsEnableRuntimeConfig(runtime::kRuntimeMemoryStat)) {
    std::cout << "[MS_RUNTIME_PROF]" << get_init_info() << std::endl;
  }
  initialized_ = true;
  return true;
}

void AscendMemAdapter::SimulationInitialize() {
  device_hbm_total_size_ = kSimuHBMTotalMemSizeGB * kGBToByte;
  device_hbm_free_size_ = device_hbm_total_size_;
  size_t reserved_mem_size_for_others;
  auto user_define_ms_size = GetDeviceMemSizeFromContext();
  if (user_define_ms_size == 0) {
    ms_used_hbm_size_ = DoubleToLong(device_hbm_free_size_ * kMSMemoryRatio);
    ms_used_hbm_size_ = (ms_used_hbm_size_ / kPerHugePageMemorySize) * kPerHugePageMemorySize - kExtraReservedMemory;
    reserved_mem_size_for_others = device_hbm_free_size_ - SizeToLong(ms_used_hbm_size_);
  } else {
    ms_used_hbm_size_ = SizeToLong(user_define_ms_size);
    if (user_define_ms_size > device_hbm_total_size_) {
      device_hbm_total_size_ = user_define_ms_size;
    }
    reserved_mem_size_for_others = device_hbm_total_size_ - user_define_ms_size;
  }

  MS_LOG(INFO) << "Simulation Device MOC Size:" << device_hbm_total_size_ / kMBToByte
               << "M, Device free MOC Size:" << device_hbm_free_size_ / kMBToByte
               << "M, Reserved MOC size for Other Components(HCCL/rts/etc.):"
               << reserved_mem_size_for_others / kMBToByte
               << "M, User define MindSpore MOC Size:" << user_define_ms_size / kGBToByte
               << "G, MindSpore Used MOC Size:" << ms_used_hbm_size_ / kMBToByte << "M.";
  max_available_ms_hbm_size_ = ms_used_hbm_size_;
  initialized_ = true;
}

bool AscendMemAdapter::DeInitialize() {
  if (!initialized_) {
    MS_LOG(INFO) << "DeInitialize Ascend Memory Adapter when it is not initialize";
    return false;
  }
  std::ostringstream oss_buf;
  oss_buf << "Ascend Memory Adapter deinitialize success, statistics:" << DevMemStatistics();
  MS_LOG(INFO) << oss_buf.str();
  if (common::IsCompileSimulation() || common::IsNeedMemoryStatistic()) {
    MS_LOG(WARNING) << oss_buf.str();
  }
  if (runtime::IsEnableRuntimeConfig(runtime::kRuntimeMemoryStat)) {
    std::cout << "[MS_RUNTIME_PROF]" << oss_buf.str() << std::endl;
  }
  device_hbm_total_size_ = 0;
  device_hbm_free_size_ = 0;
  ms_used_hbm_size_ = 0;
  max_available_ms_hbm_size_ = 0;
  initialized_ = false;
  return true;
}

namespace {
struct HugeMemReserver {
  HugeMemReserver(size_t size, size_t reserver_size) {
    MS_LOG(INFO) << "Allocate size : " << size << ", reserve_size : " << reserver_size << ".";
    if (reserver_size < kMBToByte) {
      return;
    }
    size_t free_size = 0;
    size_t total_size = 0;
    auto ret = CALL_ASCEND_API(aclrtGetMemInfo, ACL_HBM_MEM_HUGE, &free_size, &total_size);
    MS_LOG(INFO) << "Huge mem reserve free_size : " << free_size << ", total_size : " << total_size << ".";
    if (ret == ACL_SUCCESS) {
      if (free_size < reserver_size + size) {
        MS_LOG(WARNING) << "Free size of huge page mem[" << free_size
                        << "] is less than the sum of reserver_size and allocate size. Reserve size " << reserver_size
                        << ", allocate size : " << size << ", total ACL_HBM_MEM_HUGE size : " << total_size << ".";
        if (free_size < reserver_size) {
          MS_LOG(ERROR) << "Free size of huge page mem[" << free_size
                        << "] is less than reserver_size : " << reserver_size
                        << ", change reserve operation with free size.";
          reserver_size = free_size;
        }
        ret = CALL_ASCEND_API(aclrtMalloc, reinterpret_cast<void **>(&addr_), reserver_size, ACL_MEM_MALLOC_HUGE_ONLY);
        if (ret != ACL_RT_SUCCESS) {
          addr_ = nullptr;
          MS_LOG(ERROR) << "aclrtMalloc mem size[" << reserver_size << "] fail, ret[" << ret << "]";
        } else {
          MS_LOG(INFO) << "Huge mem reserve success, addr : " << addr_ << ", size : " << reserver_size << ".";
        }
      }
    } else {
      MS_LOG(WARNING) << "aclrtGetMemInfo mem size[" << size << "] fail, ret[" << ret << "]";
    }
  }

  ~HugeMemReserver() {
    if (addr_ != nullptr) {
      auto ret = CALL_ASCEND_API(aclrtFree, addr_);
      if (ret != ACL_SUCCESS) {
        MS_LOG(ERROR) << "aclrtFree mem [" << addr_ << "] fail, ret[" << ret << "]";
      } else {
        MS_LOG(INFO) << "Huge mem reserve success, free : " << addr_ << ".";
      }
    }
  }

  void *addr_{nullptr};
};
}  // namespace

uint8_t *AscendMemAdapter::MallocFromRts(size_t size) const {
  uint8_t *ptr = nullptr;
  if (AscendVmmAdapter::IsEnabled()) {
    return nullptr;
  }
  if (AscendGmemAdapter::GetInstance().is_eager_free_enabled()) {
    return AscendGmemAdapter::GetInstance().MmapMemory(size, reinterpret_cast<void *>(ptr));
  }

  HugeMemReserver huge_mem_reserver(size, device_hbm_huge_page_reserved_size_);
  auto ret = CALL_ASCEND_API(aclrtMalloc, reinterpret_cast<void **>(&ptr), size, ACL_MEM_TYPE_HIGH_BAND_WIDTH);
  if (ret != ACL_RT_SUCCESS) {
    if (ret == ACL_ERROR_RT_MEMORY_ALLOCATION) {
      auto context_ptr = MsContext::GetInstance();
      MS_EXCEPTION_IF_NULL(context_ptr);
      unsigned int device_id = context_ptr->get_param<uint32_t>(MS_CTX_DEVICE_ID);
      size_t free_size = 0;
      size_t total = 0;
      (void)CALL_ASCEND_API(aclrtGetMemInfo, ACL_HBM_MEM, &free_size, &total);
      MS_LOG(EXCEPTION) << "#umsg#Framework Error Message:#umsg#Malloc device memory failed, size[" << size << "], ret["
                        << ret << "], "
                        << "Device " << device_id << " Available MOC size:" << total << " free size:" << free_size
                        << " may be other processes occupying this card, check as: ps -ef|grep python";
    } else {
      MS_EXCEPTION(DeviceProcessError) << "rtMalloc mem size[" << size << "] fail, ret[" << ret << "]";
    }
  } else {
    MS_LOG(INFO) << "Call rtMalloc to allocate device memory Success, size: " << size
                 << " bytes, address start: " << reinterpret_cast<void *>(ptr)
                 << " end: " << reinterpret_cast<void *>(ptr + size);
  }
  return ptr;
}

uint8_t *AscendMemAdapter::MallocAlign32FromRts(size_t size) const {
  uint8_t *ptr = nullptr;
  auto ret = CALL_ASCEND_API(aclrtMallocAlign32, reinterpret_cast<void **>(&ptr), size, ACL_MEM_TYPE_HIGH_BAND_WIDTH);
  if (ret != ACL_RT_SUCCESS) {
    MS_LOG(WARNING) << "Call rtMallocAlign32 to allocate device memory failed.";
    if (ret == ACL_ERROR_RT_MEMORY_ALLOCATION) {
      auto context_ptr = MsContext::GetInstance();
      MS_EXCEPTION_IF_NULL(context_ptr);
      unsigned int device_id = context_ptr->get_param<uint32_t>(MS_CTX_DEVICE_ID);
      size_t free_size = 0;
      size_t total = 0;
      (void)CALL_ASCEND_API(aclrtGetMemInfo, ACL_HBM_MEM, &free_size, &total);
      MS_LOG(WARNING) << "MallocAlign32 device memory failed, size[" << size << "], ret[" << ret << "], "
                      << "device " << device_id << ", available MOC size:" << total << ", free size:" << free_size;
    } else {
      MS_LOG(WARNING) << "MallocAlign32 device memory failed, size[" << size << "], ret[" << ret << "]";
    }
  } else {
    MS_LOG(INFO) << "Call rtMallocAlign32 to allocate device memory Success, size: " << size
                 << " bytes, address start: " << reinterpret_cast<void *>(ptr)
                 << ", address end: " << reinterpret_cast<void *>(ptr + size);
  }
  return ptr;
}

bool AscendMemAdapter::FreeAlign32ToRts(void *devPtr) const {
  auto ret = CALL_ASCEND_API(aclrtFree, devPtr);
  if (ret != ACL_SUCCESS) {
    MS_LOG(ERROR) << "aclrtFree mem [" << devPtr << "] fail, ret[" << ret << "]";
    return false;
  }
  return true;
}

bool AscendMemAdapter::FreeToRts(void *devPtr, const size_t size) const {
  if (devPtr != nullptr) {
    if (AscendGmemAdapter::GetInstance().is_eager_free_enabled()) {
      return AscendGmemAdapter::GetInstance().MunmapMemory(devPtr, size);
    }
    auto ret = CALL_ASCEND_API(aclrtFree, devPtr);
    if (ret != ACL_SUCCESS) {
      MS_LOG(ERROR) << "aclrtFree mem [" << devPtr << "] fail, ret[" << ret << "]";
      return false;
    }
  }
  return true;
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
