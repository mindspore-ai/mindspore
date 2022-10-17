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

#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_GPU_GPU_DEVICE_MANAGER_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_GPU_GPU_DEVICE_MANAGER_H_

#include <cudnn.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cusparse.h>
#include <vector>
#include <memory>
#include "plugin/device/gpu/hal/device/cuda_driver.h"
#include "plugin/device/gpu/hal/device/gpu_memory_allocator.h"

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

  bool CreateStream(CudaDeviceStream *stream);
  bool CreateStream(size_t *stream_id);
  bool DestroyStream(size_t stream_id);
  CudaDeviceStream GetStream(size_t stream_id) const;
  bool SyncStream(size_t stream_id) const;
  bool SyncStream(const CudaDeviceStream &stream) const;
  bool SyncAllStreams() const;
  const CudaDeviceStream &default_stream() const;
  size_t default_stream_id() const;

  const cudnnHandle_t &GetCudnnHandle() const;
  const cublasHandle_t &GetCublasHandle() const;
  const cusolverDnHandle_t &GetCusolverDnHandle() const;
  const cusparseHandle_t &GetCuSparseHandle() const;

  bool CopyDeviceMemToHost(const HostMemPtr &dst, const DeviceMemPtr &src, size_t size) const;
  bool CopyHostMemToDevice(const DeviceMemPtr &dst, const void *src, size_t size) const;

  bool CopyDeviceMemToHostAsync(const HostMemPtr &dst, const DeviceMemPtr &src, size_t size,
                                CudaDeviceStream stream) const;
  bool CopyHostMemToDeviceAsync(const DeviceMemPtr &dst, const void *src, size_t size, CudaDeviceStream stream) const;
  bool CopyDeviceMemToDeviceAsync(const DeviceMemPtr &dst, const DeviceMemPtr &src, size_t size,
                                  CudaDeviceStream stream) const;
  bool CopyHostMemToHost(const HostMemPtr &dst, const void *src, size_t size) const;

  static GPUDeviceManager &GetInstance();

 private:
  GPUDeviceManager() : dev_id_init_(false), cur_dev_id_(0), dev_alive_(false) {}
  ~GPUDeviceManager() = default;
  GPUDeviceManager(const GPUDeviceManager &) = delete;
  GPUDeviceManager &operator=(const GPUDeviceManager &) = delete;

  // Ensure the thread safety for creating and destroying stream.
  std::mutex stream_mutex_;

  // default CUDA stream used for all the kernels.
  CudaDeviceStream default_stream_{nullptr};

  size_t default_stream_id_{0};

  // all gpu CUDA streams including default_stream_.
  std::vector<CudaDeviceStream> gpu_streams_;

  // handle used for cuDNN kernels.
  cudnnHandle_t cudnn_handle_{nullptr};

  // handle used for cuBLAS kernels.
  cublasHandle_t cublas_handle_{nullptr};

  // handle used for cusolver dn kernels;
  cusolverDnHandle_t cusolver_dn_handle_{nullptr};

  // handle used for cusparse kernels;
  cusparseHandle_t cusparse_handle_{nullptr};
  bool dev_id_init_;
  uint32_t cur_dev_id_;
  bool dev_alive_;
};
}  // namespace gpu
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_GPU_GPU_DEVICE_MANAGER_H_
