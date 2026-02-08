/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_CUDA_DEVICE_INFO_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_CUDA_DEVICE_INFO_H_

#include <cuda_fp16.h>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <memory>
#include <mutex>
#include <string>
#ifdef _MSC_VER
#include <cassert>
#else
#include <assert.h>
#endif

#ifdef _MSC_VER
#ifdef BUILDING_CUDA_OPS_DLL
#define CUDA_LIB_EXPORT __declspec(dllexport)
#else
#define CUDA_LIB_EXPORT __declspec(dllimport)
#endif
#else
#define CUDA_LIB_EXPORT __attribute__((visibility("default")))
#endif  // _MSC_VER

#ifdef _WIN32
#ifndef uint
#define uint unsigned int
#endif
#endif

#ifndef _MSC_VER
#define CUDA_KERNEL_ASSERT(cond)                                                       \
  if (!(cond)) {                                                                       \
    __assert_fail(#cond, __FILE__, static_cast<unsigned int>(__LINE__), __FUNCTION__); \
  }
#else
#define CUDA_KERNEL_ASSERT(cond) \
  if (!(cond)) {                 \
    assert(0);                   \
  }
#endif
namespace mindspore {
namespace device {
namespace gpu {
class GPUdeviceInfo {
 public:
  explicit GPUdeviceInfo(const uint32_t device_id);
  ~GPUdeviceInfo();
  inline const cudaDeviceProp &properties() const { return prop_; }
  inline const std::string &name() const { return name_; }
  inline size_t threads_num() const { return threads_per_block_; }
  inline size_t threads_num(size_t size) const { return std::min(size, threads_per_block_); }
  inline size_t major_sm() const { return major_sm_; }
  inline size_t minor_sm() const { return minor_sm_; }
  inline float cuda_cap() const { return static_cast<float>(major_sm_ * 10 + minor_sm_) / 10.0; }
  inline size_t blocks_num(const size_t total_threads) const {
    return std::min(static_cast<size_t>(((total_threads - 1) / threads_per_block_) + 1), max_blocks_);
  }
  inline size_t blocks_num(const size_t total_threads, const size_t block_size) const {
    size_t valid_block_size = std::min(block_size, threads_per_block_);
    return std::min(static_cast<size_t>(((total_threads - 1) / valid_block_size) + 1), max_blocks_);
  }
  inline size_t blocks_max_num(size_t size) const { return std::min(size, max_blocks_); }
  inline dim3 grids_max_size() const { return max_grid_size_; }
  size_t share_memory_size() const { return max_share_memory_; }
  void set_check_sm(const bool &flag) { check_sm_ = flag; }
  bool check_sm() const { return check_sm_; }

  static std::shared_ptr<GPUdeviceInfo> GetInstance(uint32_t device_id);

 private:
  GPUdeviceInfo(const GPUdeviceInfo &) = delete;
  GPUdeviceInfo &operator=(const GPUdeviceInfo &) = delete;

  cudaDeviceProp prop_;
  std::string name_;
  size_t max_blocks_;
  size_t threads_per_block_;
  size_t major_sm_;
  size_t minor_sm_;
  size_t max_share_memory_;
  bool check_sm_{true};
  dim3 max_grid_size_;
#ifndef _MSC_VER
  static pthread_rwlock_t rwlock_;
#else
  static std::mutex instanceLock;
#endif
};

#define CUDA_BLOCKS(device_id, total_threads) \
  mindspore::device::gpu::GPUdeviceInfo::GetInstance(device_id)->blocks_num(total_threads)
#define CUDA_BLOCKS_CAL(device_id, total_threads, block_size) \
  mindspore::device::gpu::GPUdeviceInfo::GetInstance(device_id)->blocks_num(total_threads, block_size)
#define CUDA_THREADS(device_id) mindspore::device::gpu::GPUdeviceInfo::GetInstance(device_id)->threads_num()
#define CUDA_THREADS_MAXSIZE(device_id, size) \
  mindspore::device::gpu::GPUdeviceInfo::GetInstance(device_id)->threads_num(size)
#define CUDA_BLOCKS_MAXSIZE(device_id, size) \
  mindspore::device::gpu::GPUdeviceInfo::GetInstance(device_id)->blocks_max_num(size)
#define CUDA_MAJOR_SM(device_id) mindspore::device::gpu::GPUdeviceInfo::GetInstance(device_id)->major_sm()
#define CUDA_CAP(device_id) mindspore::device::gpu::GPUdeviceInfo::GetInstance(device_id)->cuda_cap()
#define CUDA_SHARED_MEM_PER_BLOCK(device_id) \
  mindspore::device::gpu::GPUdeviceInfo::GetInstance(device_id)->share_memory_size()
#define CUDA_GRIDS_MAXSIZE(device_id) mindspore::device::gpu::GPUdeviceInfo::GetInstance(device_id)->grids_max_size()

#define MINIUM_SM 6
#define RECOMMEND_SM 7
#define AMPER_ARCH_SM 8
#define SUPPORTED_CAP 5.3
}  // namespace gpu
}  // namespace device
}  // namespace mindspore

inline cudaError_t GetCudaStatus() {
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    return status;
  }
  return cudaSuccess;
}

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_CUDA_DEVICE_INFO_H_
