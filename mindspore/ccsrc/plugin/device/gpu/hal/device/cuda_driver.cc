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

#include "plugin/device/gpu/hal/device/cuda_driver.h"
#include <nvrtc.h>
#include "utils/log_adapter.h"
#include "include/common/utils/convert_utils.h"

namespace mindspore {
namespace device {
namespace gpu {
size_t CudaDriver::AllocDeviceMem(size_t size, DeviceMemPtr *addr) {
  if (size <= 0) {
    MS_LOG(EXCEPTION) << "#umsg#Cuda error:#umsg#The cudaMalloc alloc size is under 0.";
  }
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
  if (addr == nullptr) {
    return true;
  }
  auto ret = cudaFree(addr);
  if (ret != cudaSuccess) {
    MS_LOG(ERROR) << "cudaFree failed, ret[" << static_cast<int>(ret) << "], " << cudaGetErrorString(ret);
    return false;
  }
  return true;
}

size_t CudaDriver::AllocHostPinnedMem(size_t size, void **addr) {
  if (size == 0) {
    MS_LOG(EXCEPTION) << "#umsg#Cuda error:#umsg#The cudaHostAlloc allocate size is 0";
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
      MS_LOG(EXCEPTION) << "#umsg#Cuda error:#umsg#The cudaFreeHost failed, ret[" << static_cast<int>(ret) << "], "
                        << cudaGetErrorString(ret);
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

bool CudaDriver::CopyHostMemToHost(const HostMemPtr &dst, const void *src, size_t size) {
  auto ret = cudaMemcpy(dst, src, size, cudaMemcpyHostToHost);
  if (ret != cudaSuccess) {
    MS_LOG(ERROR) << "cudaMemcpy failed, ret[" << static_cast<int>(ret) << "], " << cudaGetErrorString(ret);
    return false;
  }
  return true;
}

bool CudaDriver::CopyHostMemToDeviceAsync(const DeviceMemPtr &dst, const void *src, size_t size,
                                          CudaDeviceStream stream) {
  auto ret = cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, (cudaStream_t)stream);
  if (ret != cudaSuccess) {
    MS_LOG(ERROR) << "cudaMemcpyAsync failed, ret[" << static_cast<int>(ret) << "], " << cudaGetErrorString(ret);
    return false;
  }
  return true;
}

bool CudaDriver::CopyDeviceMemToHostAsync(const HostMemPtr &dst, const void *src, size_t size,
                                          CudaDeviceStream stream) {
  auto ret = cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, (cudaStream_t)stream);
  if (ret != cudaSuccess) {
    MS_LOG(ERROR) << "cudaMemcpyAsync failed, ret[" << static_cast<int>(ret) << "], " << cudaGetErrorString(ret);
    return false;
  }
  return true;
}

bool CudaDriver::CopyDeviceMemToDeviceAsync(const DeviceMemPtr &dst, const void *src, size_t size,
                                            CudaDeviceStream stream) {
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

bool CudaDriver::CreateStream(CudaDeviceStream *stream) {
  auto ret = cudaStreamCreateWithFlags(reinterpret_cast<CUstream_st **>(stream), cudaStreamNonBlocking);
  if (ret != cudaSuccess) {
    MS_LOG(EXCEPTION) << "#umsg#Cuda error:#umsg#The cudaStreamCreateWithFlags failed, ret[" << static_cast<int>(ret)
                      << "], " << cudaGetErrorString(ret);
  }
  return true;
}

bool CudaDriver::DestroyStream(const CudaDeviceStream &stream) {
  auto ret = cudaStreamDestroy((cudaStream_t)stream);
  if (ret != cudaSuccess) {
    MS_LOG(ERROR) << "cudaStreamDestroy failed, ret[" << static_cast<int>(ret) << "], " << cudaGetErrorString(ret);
    return false;
  }
  return true;
}

bool CudaDriver::SyncStream(const CudaDeviceStream &stream) {
  auto ret = cudaStreamSynchronize((cudaStream_t)stream);
  if (ret != cudaSuccess) {
    MS_LOG(ERROR) << "cudaStreamSynchronize failed, ret[" << static_cast<int>(ret) << "], " << cudaGetErrorString(ret);
    return false;
  }
  return true;
}

bool CudaDriver::ConstructEvent(CudaDeviceEvent *event, unsigned int flag) {
  auto ret = cudaEventCreateWithFlags(reinterpret_cast<cudaEvent_t *>(event), flag);
  if (ret != cudaSuccess) {
    MS_LOG(ERROR) << "cudaEventCreateWithFlags failed, ret[" << static_cast<int>(ret) << "], "
                  << cudaGetErrorString(ret);
    return false;
  }
  return true;
}

bool CudaDriver::DestroyEvent(const CudaDeviceEvent &event) {
  auto ret = cudaEventDestroy((cudaEvent_t)event);
  if (ret != cudaSuccess) {
    MS_LOG(ERROR) << "cudaEventDestroy failed, ret[" << static_cast<int>(ret) << "], " << cudaGetErrorString(ret);
    return false;
  }
  return true;
}

bool CudaDriver::RecordEvent(CudaDeviceEvent event, CudaDeviceStream stream) {
  auto ret = cudaEventRecord((cudaEvent_t)event, (cudaStream_t)stream);
  if (ret != cudaSuccess) {
    MS_LOG(ERROR) << "cudaEventRecord failed, ret[" << static_cast<int>(ret) << "], " << cudaGetErrorString(ret);
    return false;
  }
  return true;
}

bool CudaDriver::SyncEvent(const CudaDeviceEvent &event) {
  auto ret = cudaEventSynchronize((cudaEvent_t)event);
  if (ret != cudaSuccess) {
    MS_LOG(ERROR) << "cudaEventSynchronize failed, ret[" << static_cast<int>(ret) << "], " << cudaGetErrorString(ret);
    return false;
  }
  return true;
}

bool CudaDriver::QueryEvent(const CudaDeviceEvent &event) {
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

bool CudaDriver::ElapsedTime(float *cost_time, const CudaDeviceEvent &start, const CudaDeviceEvent &end) {
  auto ret = cudaEventElapsedTime(cost_time, (cudaEvent_t)start, (cudaEvent_t)end);
  if (ret == cudaSuccess) {
    return true;
  } else {
    MS_LOG(ERROR) << "cudaEventElapsedTime failed, ret[" << static_cast<int>(ret) << "], " << cudaGetErrorString(ret);
    return false;
  }
}

int CudaDriver::device_count() {
  auto last_error = cudaGetLastError();
  if (last_error != cudaSuccess) {
    MS_LOG(EXCEPTION) << "#umsg#Cuda error:#umsg#The cudaGetLastError[" << static_cast<int>(last_error) << "], "
                      << cudaGetErrorString(last_error);
  }

  int dev_count = 0;
  auto ret = cudaGetDeviceCount(&dev_count);
  if (ret != cudaSuccess) {
    MS_LOG(EXCEPTION) << "#umsg#Cuda error:#umsg#The cudaGetDeviceCount failed, ret[" << static_cast<int>(ret) << "], "
                      << cudaGetErrorString(ret);
  }
  return dev_count;
}

bool CudaDriver::SetDevice(int index) {
  auto ret = cudaSetDevice(index);
  if (ret != cudaSuccess) {
    MS_LOG(EXCEPTION)
      << "#umsg#Cuda error:#umsg#SetDevice for id:" << index << " failed, ret[" << static_cast<int>(ret) << "], "
      << cudaGetErrorString(ret)
      << ". Please make sure that the 'device_id' set in context is in the range:[0, total number of GPU). "
         "If the environment variable 'CUDA_VISIBLE_DEVICES' is set, the total number of GPU will be the number set "
         "in the environment variable 'CUDA_VISIBLE_DEVICES'. For example, if export CUDA_VISIBLE_DEVICES=4,5,6, the "
         "'device_id' can be 0,1,2 at the moment, 'device_id' starts from 0, and 'device_id'=0 means using GPU of "
         "number 4.";
  }
  int major = 0;
  int minor = 0;
  auto curtc_ret = nvrtcVersion(&major, &minor);
  if (curtc_ret == nvrtcResult::NVRTC_SUCCESS) {
    MS_LOG(DEBUG) << "NVRTC version is " << major << "." << minor;
  }
  return true;
}
}  // namespace gpu
}  // namespace device
}  // namespace mindspore
