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

#ifndef MINDSPORE_CCSRC_DEVICE_GPU_CUDA_COMMON_H_
#define MINDSPORE_CCSRC_DEVICE_GPU_CUDA_COMMON_H_

#include <algorithm>
#include "device/gpu/gpu_device_manager.h"

namespace mindspore {
namespace device {
namespace gpu {
class CudaCommon {
 public:
  inline int threads_num() const { return threads_per_block_; }
  inline int major_sm() const { return major_sm_; }
  inline int blocks_num(const int total_threads) const {
    return std::min(((total_threads - 1) / threads_per_block_) + 1, max_blocks_);
  }

  static CudaCommon &GetInstance() {
    static CudaCommon instance;
    return instance;
  }

 private:
  CudaCommon() {
    uint32_t device_id = GPUDeviceManager::GetInstance().cur_device_id();
    cudaDeviceProp prop;
    (void)cudaGetDeviceProperties(&prop, device_id);
    threads_per_block_ = prop.maxThreadsPerBlock;
    max_blocks_ = prop.multiProcessorCount;
    major_sm_ = prop.major;
  }
  ~CudaCommon() = default;
  CudaCommon(const CudaCommon &) = delete;
  CudaCommon &operator=(const CudaCommon &) = delete;

  int max_blocks_;
  int threads_per_block_;
  int major_sm_;
};
#define GET_BLOCKS(total_threads) mindspore::device::gpu::CudaCommon::GetInstance().blocks_num(total_threads)
#define GET_THREADS mindspore::device::gpu::CudaCommon::GetInstance().threads_num()
#define GET_MAJOR_SM mindspore::device::gpu::CudaCommon::GetInstance().major_sm()
#define MINIUM_SM 7
}  // namespace gpu
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_DEVICE_GPU_CUDA_COMMON_H_
