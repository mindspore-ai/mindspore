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

#ifndef MINDSPORE_CCSRC_DEVICE_GPU_GPU_DEVICE_MANAGER_H_
#define MINDSPORE_CCSRC_DEVICE_GPU_GPU_DEVICE_MANAGER_H_

#include <cudnn.h>
#include <cublas_v2.h>
#include <vector>
#include <memory>
#include "device/gpu/cuda_driver.h"
#include "device/gpu/gpu_memory_allocator.h"

namespace mindspore {
namespace device {
namespace gpu {
class GPUDeviceManager {
 public:
  void InitDevice();
  void ReleaseDevice();

  int device_count() const;
  bool set_cur_device_id(uint32_t device_id);
  uint32_t cur_device_id() const;
  bool is_device_id_init() const;

  bool CreateStream(DeviceStream* stream);
  bool SyncStream(const DeviceStream& stream) const;
  const DeviceStream& default_stream() const;

  const cudnnHandle_t& GetCudnnHandle() const;
  const cublasHandle_t& GetCublasHandle() const;

  bool CopyDeviceMemToHost(const HostMemPtr& dst, const DeviceMemPtr& src, size_t size) const;
  bool CopyHostMemToDevice(const DeviceMemPtr& dst, const void* src, size_t size) const;

  static GPUDeviceManager& GetInstance() {
    static GPUDeviceManager instance;
    return instance;
  }

 private:
  GPUDeviceManager() : dev_id_init_(false), cur_dev_id_(0) {}
  ~GPUDeviceManager() = default;
  GPUDeviceManager(const GPUDeviceManager&) = delete;
  GPUDeviceManager& operator=(const GPUDeviceManager&) = delete;

  // default CUDA stream used for all the kernels.
  DeviceStream default_stream_{nullptr};

  // all gpu CUDA streams including default_stream_.
  std::vector<DeviceStream> gpu_streams_;

  // handle used for cuDNN kernels.
  cudnnHandle_t cudnn_handle_{nullptr};

  // handle used for cuBLAS kernels.
  cublasHandle_t cublas_handle_{nullptr};

  bool dev_id_init_;
  uint32_t cur_dev_id_;
};
}  // namespace gpu
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_DEVICE_GPU_GPU_DEVICE_MANAGER_H_
