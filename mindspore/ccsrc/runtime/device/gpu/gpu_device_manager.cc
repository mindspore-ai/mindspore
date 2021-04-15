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

#include "runtime/device/gpu/gpu_device_manager.h"
#include "runtime/device/gpu/gpu_common.h"
#include "utils/log_adapter.h"
#include "utils/convert_utils.h"
#include "runtime/device/gpu/gpu_buffer_mgr.h"

namespace mindspore {
namespace device {
namespace gpu {
void GPUDeviceManager::InitDevice() {
  CHECK_OP_RET_WITH_EXCEPT(CudaDriver::set_current_device(SizeToInt(cur_dev_id_)), "Failed to set current device id");
  CHECK_OP_RET_WITH_EXCEPT(CreateStream(&default_stream_), "Failed to create CUDA stream.");
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreate(&cudnn_handle_), "Failed to create cuDNN handle");
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnSetStream(cudnn_handle_, reinterpret_cast<cudaStream_t>(default_stream())),
                                      "Failed to set stream for cuDNN handle.");
  CHECK_CUBLAS_RET_WITH_EXCEPT_NOTRACE(cublasCreate(&cublas_handle_), "Failed to create cuBLAS handle.");
  CHECK_CUBLAS_RET_WITH_EXCEPT_NOTRACE(
    cublasSetStream(cublas_handle_, reinterpret_cast<cudaStream_t>(default_stream())),
    "Failed to set stream for cuBLAS handle.");
  CHECK_CUSOLVER_RET_WITH_EXCEPT_NOTRACE(cusolverDnCreate(&cusolver_dn_handle_),
                                         "Failed to create cusolver dn handle.");
  CHECK_CUSOLVER_RET_WITH_EXCEPT_NOTRACE(
    cusolverDnSetStream(cusolver_dn_handle_, reinterpret_cast<cudaStream_t>(default_stream())),
    "Failed to set stream for cusolver dn handle");
  CHECK_OP_RET_WITH_EXCEPT(GPUMemoryAllocator::GetInstance().Init(), "Failed to Init gpu memory allocator")
  dev_alive_ = true;
}

void GPUDeviceManager::ReleaseDevice() {
  for (DeviceStream stream : gpu_streams_) {
    if (stream != nullptr) {
      CHECK_OP_RET_WITH_ERROR(CudaDriver::DestroyStream(stream), "Failed to destroy CUDA stream.");
    }
  }
  if (cudnn_handle_ != nullptr) {
    CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnDestroy(cudnn_handle_), "Failed to destroy cuDNN handle");
  }
  if (cublas_handle_ != nullptr) {
    CHECK_CUBLAS_RET_WITH_ERROR(cublasDestroy(cublas_handle_), "Failed to destroy cuBLAS handle.");
  }
  if (cusolver_dn_handle_ != nullptr) {
    CHECK_CUSOLVER_RET_WITH_ERROR(cusolverDnDestroy(cusolver_dn_handle_), "Failed to destroy cusolver dn handle.");
  }
  CHECK_OP_RET_WITH_ERROR(GPUMemoryAllocator::GetInstance().Finalize(), "Failed to destroy gpu memory allocator");
  dev_alive_ = false;
}

bool GPUDeviceManager::CreateStream(DeviceStream *stream) {
  CHECK_OP_RET_WITH_EXCEPT(CudaDriver::CreateStream(stream), "Failed to create CUDA stream");
  gpu_streams_.emplace_back(*stream);
  return true;
}

const DeviceStream &GPUDeviceManager::default_stream() const { return default_stream_; }

int GPUDeviceManager::device_count() const { return CudaDriver::device_count(); }

bool GPUDeviceManager::set_cur_device_id(uint32_t device_id) {
  if (!dev_id_init_) {
    dev_id_init_ = true;
    cur_dev_id_ = device_id;
    mindspore::device::GpuBufferMgr::GetInstance().set_device_id(UintToInt(device_id));
    return true;
  } else {
    MS_LOG(ERROR) << "Device already been set.";
    return false;
  }
}

uint32_t GPUDeviceManager::cur_device_id() const { return cur_dev_id_; }

bool GPUDeviceManager::is_device_id_init() const { return dev_id_init_; }

const cudnnHandle_t &GPUDeviceManager::GetCudnnHandle() const { return cudnn_handle_; }

const cublasHandle_t &GPUDeviceManager::GetCublasHandle() const { return cublas_handle_; }
const cusolverDnHandle_t &GPUDeviceManager::GetCusolverDnHandle() const { return cusolver_dn_handle_; }
bool GPUDeviceManager::SyncStream(const DeviceStream &stream) const {
  return dev_alive_ ? CudaDriver::SyncStream(stream) : false;
}

bool GPUDeviceManager::CopyDeviceMemToHost(const HostMemPtr &dst, const DeviceMemPtr &src, size_t size) const {
  return CudaDriver::CopyDeviceMemToHost(dst, src, size);
}

bool GPUDeviceManager::CopyHostMemToDevice(const DeviceMemPtr &dst, const void *src, size_t size) const {
  return CudaDriver::CopyHostMemToDevice(dst, src, size);
}

bool GPUDeviceManager::CopyDeviceMemToHostAsync(const HostMemPtr &dst, const DeviceMemPtr &src, size_t size,
                                                DeviceStream stream) const {
  return CudaDriver::CopyDeviceMemToHostAsync(dst, src, size, stream);
}

bool GPUDeviceManager::CopyHostMemToDeviceAsync(const DeviceMemPtr &dst, const void *src, size_t size,
                                                DeviceStream stream) const {
  return CudaDriver::CopyHostMemToDeviceAsync(dst, src, size, stream);
}

bool GPUDeviceManager::CopyDeviceMemToDeviceAsync(const DeviceMemPtr &dst, const DeviceMemPtr &src, size_t size,
                                                  DeviceStream stream) const {
  return CudaDriver::CopyDeviceMemToDeviceAsync(dst, src, size, stream);
}
}  // namespace gpu
}  // namespace device
}  // namespace mindspore
