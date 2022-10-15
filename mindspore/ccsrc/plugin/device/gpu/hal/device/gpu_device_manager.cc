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

#include "plugin/device/gpu/hal/device/gpu_device_manager.h"
#include "plugin/device/gpu/hal/device/gpu_common.h"
#include "utils/log_adapter.h"
#include "include/common/utils/convert_utils.h"

namespace mindspore {
namespace device {
namespace gpu {
GPUDeviceManager &GPUDeviceManager::GetInstance() {
  static GPUDeviceManager instance;
  return instance;
}

void GPUDeviceManager::InitDevice() {
  CHECK_OP_RET_WITH_EXCEPT(CudaDriver::SetDevice(SizeToInt(cur_dev_id_)), "Failed to set current device id");
  if (dev_alive_) {
    return;
  }
  CHECK_OP_RET_WITH_EXCEPT(CreateStream(&default_stream_), "Failed to create CUDA stream.");
  default_stream_id_ = gpu_streams_.size() - 1;
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
  // Create cusparse handle.
  CHECK_CUSPARSE_RET_WITH_EXCEPT(cusparseCreate(&cusparse_handle_), "Failed to create sparse handle.");
  CHECK_CUSPARSE_RET_WITH_EXCEPT(cusparseSetStream(cusparse_handle_, reinterpret_cast<cudaStream_t>(default_stream())),
                                 "Failed to set stream for cusparse handle");

  CHECK_OP_RET_WITH_EXCEPT(GPUMemoryAllocator::GetInstance().Init(), "Failed to Init gpu memory allocator")
  dev_alive_ = true;
}

void GPUDeviceManager::ReleaseDevice() {
  // Avoid repeated release device resource.
  if (!dev_alive_) {
    return;
  }
  {
    std::lock_guard<std::mutex> lock_gpu_streams(stream_mutex_);
    for (CudaDeviceStream stream : gpu_streams_) {
      if (stream != nullptr) {
        CHECK_OP_RET_WITH_ERROR(CudaDriver::DestroyStream(stream), "Failed to destroy CUDA stream.");
      }
    }
    gpu_streams_.clear();
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
  if (cusparse_handle_ != nullptr) {
    CHECK_CUSPARSE_RET_WITH_ERROR(cusparseDestroy(cusparse_handle_), "Failed to destroy cusparse handle.");
  }

  dev_alive_ = false;
}

bool GPUDeviceManager::CreateStream(CudaDeviceStream *stream) {
  std::lock_guard<std::mutex> lock_gpu_streams(stream_mutex_);
  CHECK_OP_RET_WITH_EXCEPT(CudaDriver::CreateStream(stream), "Failed to create CUDA stream");
  (void)gpu_streams_.emplace_back(*stream);
  return true;
}

bool GPUDeviceManager::CreateStream(size_t *stream_id) {
  std::lock_guard<std::mutex> lock_gpu_streams(stream_mutex_);
  CudaDeviceStream stream;
  CHECK_OP_RET_WITH_EXCEPT(CudaDriver::CreateStream(&stream), "Failed to create CUDA stream");
  *stream_id = gpu_streams_.size();
  (void)gpu_streams_.emplace_back(stream);
  return true;
}

bool GPUDeviceManager::DestroyStream(size_t stream_id) {
  std::lock_guard<std::mutex> lock_gpu_streams(stream_mutex_);
  if (stream_id >= gpu_streams_.size()) {
    MS_LOG(ERROR) << "CUDA stream not found for stream id " << stream_id;
    return false;
  }
  if (gpu_streams_.at(stream_id) == nullptr) {
    MS_LOG(WARNING) << "CUDA stream hsa been destroyed for stream id " << stream_id;
    return true;
  }
  CHECK_OP_RET_WITH_EXCEPT(CudaDriver::DestroyStream(gpu_streams_.at(stream_id)), "Failed to create CUDA stream");
  gpu_streams_[stream_id] = nullptr;
  return true;
}

CudaDeviceStream GPUDeviceManager::GetStream(size_t stream_id) const {
  if (stream_id >= gpu_streams_.size()) {
    MS_LOG(DEBUG) << "Stream for stream id[" << stream_id << "] not found, return nullptr.";
    return nullptr;
  }
  return gpu_streams_[stream_id];
}

const CudaDeviceStream &GPUDeviceManager::default_stream() const { return default_stream_; }

size_t GPUDeviceManager::default_stream_id() const { return default_stream_id_; }

int GPUDeviceManager::device_count() const { return CudaDriver::device_count(); }

bool GPUDeviceManager::set_cur_device_id(uint32_t device_id) {
  if (!dev_id_init_) {
    dev_id_init_ = true;
    cur_dev_id_ = device_id;
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

const cusparseHandle_t &GPUDeviceManager::GetCuSparseHandle() const { return cusparse_handle_; }

bool GPUDeviceManager::SyncStream(size_t stream_id) const {
  if (!dev_alive_) {
    return false;
  }
  auto stream = GetStream(stream_id);
  if (stream == nullptr) {
    MS_LOG(EXCEPTION) << "Get CUDA stream for stream id failed.";
  }
  return SyncStream(stream);
}

bool GPUDeviceManager::SyncStream(const CudaDeviceStream &stream) const {
  return dev_alive_ && CudaDriver::SyncStream(stream);
}

bool GPUDeviceManager::SyncAllStreams() const {
  if (!dev_alive_) {
    return false;
  }
  for (const auto &stream : gpu_streams_) {
    if (stream != nullptr && !SyncStream(stream)) {
      return false;
    }
  }
  return true;
}

bool GPUDeviceManager::CopyDeviceMemToHost(const HostMemPtr &dst, const DeviceMemPtr &src, size_t size) const {
  return CudaDriver::CopyDeviceMemToHost(dst, src, size);
}

bool GPUDeviceManager::CopyHostMemToDevice(const DeviceMemPtr &dst, const void *src, size_t size) const {
  return CudaDriver::CopyHostMemToDevice(dst, src, size);
}

bool GPUDeviceManager::CopyHostMemToHost(const HostMemPtr &dst, const void *src, size_t size) const {
  return CudaDriver::CopyHostMemToHost(dst, src, size);
}

bool GPUDeviceManager::CopyDeviceMemToHostAsync(const HostMemPtr &dst, const DeviceMemPtr &src, size_t size,
                                                CudaDeviceStream stream) const {
  return CudaDriver::CopyDeviceMemToHostAsync(dst, src, size, stream);
}

bool GPUDeviceManager::CopyHostMemToDeviceAsync(const DeviceMemPtr &dst, const void *src, size_t size,
                                                CudaDeviceStream stream) const {
  return CudaDriver::CopyHostMemToDeviceAsync(dst, src, size, stream);
}

bool GPUDeviceManager::CopyDeviceMemToDeviceAsync(const DeviceMemPtr &dst, const DeviceMemPtr &src, size_t size,
                                                  CudaDeviceStream stream) const {
  return CudaDriver::CopyDeviceMemToDeviceAsync(dst, src, size, stream);
}
}  // namespace gpu
}  // namespace device
}  // namespace mindspore
