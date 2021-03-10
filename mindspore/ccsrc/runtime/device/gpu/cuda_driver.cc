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

#include "runtime/device/gpu/cuda_driver.h"
#include <iostream>
#include "utils/log_adapter.h"
#include "utils/convert_utils.h"

namespace mindspore {
namespace device {
namespace gpu {
size_t CudaDriver::AllocDeviceMem(size_t size, DeviceMemPtr *addr) {
  size_t retreat_count = 0;
  auto ret = cudaMalloc(reinterpret_cast<void **>(addr), size);
  // If free memory is not enough, then retry with mem_malloc_retry_rate_.
  while (ret == cudaErrorMemoryAllocation) {
    size = FloatToSize(size * mem_malloc_retry_rate_);
    size = (size / mem_malloc_align_size_) * mem_malloc_align_size_;
    ret = cudaMalloc(reinterpret_cast<void **>(addr), size);
    retreat_count++;
    if (retreat_count > mem_malloc_retry_conut_max_) {
      break;
    }
  }

  if (ret != cudaSuccess) {
    MS_LOG(ERROR) << "cudaMalloc failed, ret[" << static_cast<int>(ret) << "], " << cudaGetErrorString(ret);
    return 0;
  }
  return size;
}

bool CudaDriver::FreeDeviceMem(const DeviceMemPtr &addr) {
  auto ret = cudaFree(addr);
  if (ret != cudaSuccess) {
    MS_LOG(ERROR) << "cudaFree failed, ret[" << static_cast<int>(ret) << "], " << cudaGetErrorString(ret);
    return false;
  }
  return true;
}

size_t CudaDriver::AllocHostPinnedMem(size_t size, void **addr) {
  if (size == 0) {
    MS_LOG(EXCEPTION) << "The memory allocate size is 0";
  }
  auto ret = cudaHostAlloc(addr, size, cudaHostAllocDefault);
  if (ret != cudaSuccess) {
    MS_LOG(ERROR) << "cudaHostAlloc failed, ret[" << static_cast<int>(ret) << "], " << cudaGetErrorString(ret);
    return 0;
  }
  return size;
}

void CudaDriver::FreeHostPinnedMem(void *addr) {
  if (addr) {
    auto ret = cudaFreeHost(addr);
    if (ret != cudaSuccess) {
      MS_LOG(EXCEPTION) << "cudaFreeHost failed, ret[" << static_cast<int>(ret) << "], " << cudaGetErrorString(ret);
    }
  }
}

bool CudaDriver::CopyHostMemToDevice(const DeviceMemPtr &dst, const void *src, size_t size) {
  auto ret = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
  if (ret != cudaSuccess) {
    MS_LOG(ERROR) << "cudaMemcpy failed, ret[" << static_cast<int>(ret) << "], " << cudaGetErrorString(ret);
    return false;
  }
  return true;
}

bool CudaDriver::CopyDeviceMemToHost(const HostMemPtr &dst, const DeviceMemPtr &src, size_t size) {
  auto ret = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
  if (ret != cudaSuccess) {
    MS_LOG(ERROR) << "cudaMemcpy failed, ret[" << static_cast<int>(ret) << "], " << cudaGetErrorString(ret);
    return false;
  }
  return true;
}

bool CudaDriver::CopyHostMemToDeviceAsync(const DeviceMemPtr &dst, const void *src, size_t size, DeviceStream stream) {
  auto ret = cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, (cudaStream_t)stream);
  if (ret != cudaSuccess) {
    MS_LOG(ERROR) << "cudaMemcpyAsync failed, ret[" << static_cast<int>(ret) << "], " << cudaGetErrorString(ret);
    return false;
  }
  return true;
}

bool CudaDriver::CopyDeviceMemToHostAsync(const HostMemPtr &dst, const DeviceMemPtr &src, size_t size,
                                          DeviceStream stream) {
  auto ret = cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, (cudaStream_t)stream);
  if (ret != cudaSuccess) {
    MS_LOG(ERROR) << "cudaMemcpyAsync failed, ret[" << static_cast<int>(ret) << "], " << cudaGetErrorString(ret);
    return false;
  }
  return true;
}

bool CudaDriver::CopyDeviceMemToDeviceAsync(const DeviceMemPtr &dst, const DeviceMemPtr &src, size_t size,
                                            DeviceStream stream) {
  auto ret = cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, (cudaStream_t)stream);
  if (ret != cudaSuccess) {
    MS_LOG(ERROR) << "cudaMemcpyAsync failed, ret[" << static_cast<int>(ret) << "], " << cudaGetErrorString(ret);
    return false;
  }
  return true;
}

size_t CudaDriver::total_mem_size() {
  size_t free;
  size_t total;
  auto ret = cudaMemGetInfo(&free, &total);
  if (ret != cudaSuccess) {
    MS_LOG(ERROR) << "cudaMemGetInfo failed, ret[" << static_cast<int>(ret) << "], " << cudaGetErrorString(ret);
    return 0;
  }
  return total;
}

size_t CudaDriver::free_mem_size() {
  size_t free;
  size_t total;
  auto ret = cudaMemGetInfo(&free, &total);
  if (ret != cudaSuccess) {
    MS_LOG(ERROR) << "cudaMemGetInfo failed, ret[" << static_cast<int>(ret) << "], " << cudaGetErrorString(ret);
    return 0;
  }

  return free;
}

bool CudaDriver::CreateStream(DeviceStream *stream) {
  auto ret = cudaStreamCreateWithFlags(reinterpret_cast<CUstream_st **>(stream), cudaStreamNonBlocking);
  if (ret != cudaSuccess) {
    MS_LOG(ERROR) << "cudaStreamCreate failed, ret[" << static_cast<int>(ret) << "], " << cudaGetErrorString(ret);
    return false;
  }
  return true;
}

bool CudaDriver::DestroyStream(const DeviceStream &stream) {
  auto ret = cudaStreamDestroy((cudaStream_t)stream);
  if (ret != cudaSuccess) {
    MS_LOG(ERROR) << "cudaStreamDestroy failed, ret[" << static_cast<int>(ret) << "], " << cudaGetErrorString(ret);
    return false;
  }
  return true;
}

bool CudaDriver::SyncStream(const DeviceStream &stream) {
  auto ret = cudaStreamSynchronize((cudaStream_t)stream);
  if (ret != cudaSuccess) {
    MS_LOG(ERROR) << "cudaStreamSynchronize failed, ret[" << static_cast<int>(ret) << "], " << cudaGetErrorString(ret);
    return false;
  }
  return true;
}

bool CudaDriver::CreateEvent(DeviceEvent *event, unsigned int flag) {
  auto ret = cudaEventCreateWithFlags(reinterpret_cast<cudaEvent_t *>(event), flag);
  if (ret != cudaSuccess) {
    MS_LOG(ERROR) << "cudaEventCreateWithFlags failed, ret[" << static_cast<int>(ret) << "], "
                  << cudaGetErrorString(ret);
    return false;
  }
  return true;
}

bool CudaDriver::DestroyEvent(const DeviceEvent &event) {
  auto ret = cudaEventDestroy((cudaEvent_t)event);
  if (ret != cudaSuccess) {
    MS_LOG(ERROR) << "cudaEventDestroy failed, ret[" << static_cast<int>(ret) << "], " << cudaGetErrorString(ret);
    return false;
  }
  return true;
}

bool CudaDriver::RecordEvent(DeviceEvent event, DeviceStream stream) {
  auto ret = cudaEventRecord((cudaEvent_t)event, (cudaStream_t)stream);
  if (ret != cudaSuccess) {
    MS_LOG(ERROR) << "cudaEventRecord failed, ret[" << static_cast<int>(ret) << "], " << cudaGetErrorString(ret);
    return false;
  }
  return true;
}

bool CudaDriver::SyncEvent(const DeviceEvent &event) {
  auto ret = cudaEventSynchronize((cudaEvent_t)event);
  if (ret != cudaSuccess) {
    MS_LOG(ERROR) << "cudaEventSynchronize failed, ret[" << static_cast<int>(ret) << "], " << cudaGetErrorString(ret);
    return false;
  }
  return true;
}

bool CudaDriver::QueryEvent(const DeviceEvent &event) {
  auto ret = cudaEventQuery((cudaEvent_t)event);
  if (ret == cudaSuccess) {
    return true;
  } else if (ret == cudaErrorNotReady) {
    return false;
  } else {
    MS_LOG(ERROR) << "cudaEventQuery failed, ret[" << static_cast<int>(ret) << "], " << cudaGetErrorString(ret);
    return false;
  }
}

bool CudaDriver::ElapsedTime(float *cost_time, const DeviceEvent &start, const DeviceEvent &end) {
  auto ret = cudaEventElapsedTime(cost_time, (cudaEvent_t)start, (cudaEvent_t)end);
  if (ret == cudaSuccess) {
    return true;
  } else {
    MS_LOG(ERROR) << "cudaEventElapsedTime failed, ret[" << static_cast<int>(ret) << "], " << cudaGetErrorString(ret);
    return false;
  }
}

int CudaDriver::device_count() {
  int dev_count;
  auto ret = cudaGetDeviceCount(&dev_count);
  if (ret != cudaSuccess) {
    MS_LOG(ERROR) << "cudaGetDeviceCount failed, ret[" << static_cast<int>(ret) << "], " << cudaGetErrorString(ret);
  }
  return dev_count;
}

bool CudaDriver::set_current_device(int index) {
  auto ret = cudaSetDevice(index);
  if (ret != cudaSuccess) {
    MS_LOG(ERROR) << "cudaSetDevice " << index << " failed, ret[" << static_cast<int>(ret) << "], "
                  << cudaGetErrorString(ret);
    return false;
  }
  return true;
}
}  // namespace gpu
}  // namespace device
}  // namespace mindspore
