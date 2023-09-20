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

#include "plugin/device/ascend/hal/device/ascend_dma_handle.h"
#include "runtime/rt.h"
#include "runtime/mem.h"
#include "acl/acl_rt.h"
#include "acl/acl.h"
#if defined(RT_MEMORY_P2PDMA)
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <cstdlib>
#include <string>
#include "toolchain/slog.h"
#include "external/runtime/rt_error_codes.h"
#endif
#include "utils/log_adapter.h"
#include "include/common/utils/utils.h"
#include "runtime/device/kernel_runtime_manager.h"

namespace mindspore {
namespace device {
namespace ascend {
AscendDmaHandle &AscendDmaHandle::GetInstance() {
  static AscendDmaHandle instance{};
  return instance;
}

void *AscendDmaHandle::GetBuf() const { return buf_; }
void *AscendDmaHandle::GetDargs() const { return dargs_; }
size_t AscendDmaHandle::GetSize() const { return hbm_alloc_size_; }

AscendDmaHandle::~AscendDmaHandle() {
#if defined(RT_MEMORY_P2PDMA)
  munmap(buf_, hbm_alloc_size_);
  close(p2p_fd_);
  aclrtFree(dargs_);
#endif
}

AscendDmaHandle::AscendDmaHandle() {
  InitRuntimeInstance();
  InitDmaMem();
}

void AscendDmaHandle::InitRuntimeInstance() {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  device_id_ = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  runtime_instance_ = device::KernelRuntimeManager::Instance().GetKernelRuntime(kAscendDevice, device_id_);
  MS_EXCEPTION_IF_NULL(runtime_instance_);
}

void AscendDmaHandle::InitDmaMem() {
#if defined(RT_MEMORY_P2PDMA)
  auto ret = aclrtSetDevice(device_id_);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "aclrtSetDevice failed:" << ret;
  }
  ret = aclrtGetMemInfo(ACL_HBM_MEM, &device_hbm_free_size_, &device_hbm_total_size_);
  MS_LOG(INFO) << "InitDmaMem device_hbm_free_size_:" << device_hbm_free_size_
               << ", device_hbm_total_size_:" << device_hbm_total_size_;
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "rtMemGetInfo failed:" << ret;
  }
  ret = rtMalloc(&dargs_, hbm_alloc_size_, RT_MEMORY_P2PDMA, 0);
  if (ret != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "rtMalloc failed:" << ret;
  }
  ret = aclrtMemset(dargs_, hbm_alloc_size_, 0x44, hbm_alloc_size_);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "aclrtMemset failed:" << ret;
  }
  std::string p2p_device_name = "/dev/p2pdma" + std::to_string(device_id_);
  p2p_fd_ = open(p2p_device_name.c_str(), O_RDWR);
  if (p2p_fd_ < 0) {
    MS_LOG(EXCEPTION) << "Open device failed";
  }
  buf_ = mmap(nullptr, hbm_alloc_size_, PROT_READ | PROT_WRITE, MAP_SHARED, p2p_fd_, 0);
  if (!buf_) {
    MS_LOG(EXCEPTION) << "Fail to mmap";
  }
  MS_LOG(INFO) << "AscendDmaHandle mmap success device prt:" << dargs_ << " buffer ptr:" << buf_;
#endif
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
