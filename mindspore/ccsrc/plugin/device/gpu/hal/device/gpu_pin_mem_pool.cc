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

#include "plugin/device/gpu/hal/device/gpu_pin_mem_pool.h"
#include "plugin/device/gpu/hal/device/cuda_driver.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace device {
namespace gpu {
namespace {
constexpr size_t kMemPoolAlignSize = 512;
}
GPUPinMemPool &GPUPinMemPool::GetInstance() {
  static GPUPinMemPool instance{};
  return instance;
}

void GPUPinMemPool::PinnedMemAlloc(DeviceMemPtr *addr, size_t alloc_size) {
#if defined(_WIN32) || defined(_WIN64)
  MS_LOG(WARNING) << "The WIN platform is not implemented.";
#else
  auto status = posix_memalign(addr, kMemPoolAlignSize, alloc_size);
  if (status != 0) {
    MS_LOG(ERROR) << "The PinMemPool posix_memalign failed, error code is " << status << ".";
    return;
  }
  CudaDriver::CudaHostRegister(*addr, alloc_size);
  MS_LOG(INFO) << "Enable pinned memory success addr:" << *addr << " size:" << alloc_size;
#endif
}

bool GPUPinMemPool::FreeDeviceMem(const DeviceMemPtr &addr) {
  MS_EXCEPTION_IF_NULL(addr);
  if (pinned_mem_) {
    CudaDriver::CudaHostUnregister(addr);
  }
  free(addr);
  return true;
}
}  // namespace gpu
}  // namespace device
}  // namespace mindspore
