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

#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_GPU_CUDA_DRIVER_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_GPU_CUDA_DRIVER_H_

#include <cuda_runtime_api.h>

namespace mindspore {
namespace device {
namespace gpu {
typedef void *CudaDeviceStream;
typedef void *CudaDeviceEvent;
typedef void *HostMemPtr;
typedef void *DeviceMemPtr;

class CudaDriver {
 public:
  // Encapsulate the cuda APIs associated with memory operations
  // such as malloc/free and memory copy from host to device and reverse.
  static size_t AllocDeviceMem(size_t size, DeviceMemPtr *addr);
  static bool FreeDeviceMem(const DeviceMemPtr &addr);
  static size_t AllocHostPinnedMem(size_t size, void **addr);
  static void FreeHostPinnedMem(void *addr);

  static void CudaHostRegister(void *addr, size_t alloc_size);

  static void CudaHostUnregister(void *addr);

  static bool CopyHostMemToDevice(const DeviceMemPtr &dst, const void *src, size_t size);
  static bool CopyDeviceMemToHost(const HostMemPtr &dst, const DeviceMemPtr &src, size_t size);
  static bool CopyHostMemToHost(const DeviceMemPtr &dst, const void *src, size_t size);

  static bool CopyHostMemToDeviceAsync(const DeviceMemPtr &dst, const void *src, size_t size,
                                       CudaDeviceStream stream = 0);
  static bool CopyDeviceMemToHostAsync(const HostMemPtr &dst, const void *src, size_t size,
                                       CudaDeviceStream stream = 0);
  static bool CopyDeviceMemToDeviceAsync(const DeviceMemPtr &dst, const void *src, size_t size,
                                         CudaDeviceStream stream = 0);

  static size_t total_mem_size();
  static size_t free_mem_size();

  // Encapsulate the cuda APIs associated with device resource
  // such as Stream and Event.
  static bool CreateStream(CudaDeviceStream *stream);
  static bool DestroyStream(const CudaDeviceStream &stream);
  static bool SyncStream(const CudaDeviceStream &stream);

  static bool ConstructEvent(CudaDeviceEvent *event, unsigned int flag = cudaEventDefault);
  static bool DestroyEvent(const CudaDeviceEvent &event);
  static bool RecordEvent(CudaDeviceEvent event, CudaDeviceStream stream = 0);
  static bool SyncEvent(const CudaDeviceEvent &event);
  static bool QueryEvent(const CudaDeviceEvent &event);
  static bool ElapsedTime(float *cost_time, const CudaDeviceEvent &start, const CudaDeviceEvent &end);

  // Encapsulate the cuda APIs associated with device management.
  static int device_count();
  static bool SetDevice(int index);

 private:
  CudaDriver() = delete;
  ~CudaDriver() = delete;
  CudaDriver(const CudaDriver &) = delete;
  CudaDriver &operator=(const CudaDriver &) = delete;

  static constexpr float mem_malloc_retry_rate_{0.99};
  static constexpr size_t mem_malloc_retry_conut_max_{10};
  static constexpr size_t mem_malloc_align_size_{4};
};
}  // namespace gpu
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_GPU_CUDA_DRIVER_H_
