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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_CUDA_COMMON_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_CUDA_COMMON_H_
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_device_info.h"

namespace mindspore {
namespace device {
namespace gpu {
class CudaCommon {
 public:
  inline int threads_num() const { return CUDA_THREADS(device_id_); }
  inline int threads_num(int size) const { return CUDA_THREADS_MAXSIZE(device_id_, size); }
  inline int major_sm() const { return CUDA_MAJOR_SM(device_id_); }
  inline float cuda_cap() const { return CUDA_CAP(device_id_); }
  inline int blocks_num(const int total_threads) const { return CUDA_BLOCKS(device_id_, total_threads); }
  size_t share_memory_size() const { return CUDA_SHARED_MEM_PER_BLOCK(device_id_); }
  void set_check_sm(const bool &flag) { GPUdeviceInfo::GetInstance(device_id_)->set_check_sm(flag); }
  bool check_sm() const { return GPUdeviceInfo::GetInstance(device_id_)->check_sm(); }

  static CudaCommon &GetInstance();

 private:
  CudaCommon();
  ~CudaCommon() = default;
  CudaCommon(const CudaCommon &) = delete;
  CudaCommon &operator=(const CudaCommon &) = delete;

  uint32_t device_id_;
};
#define GET_BLOCKS(total_threads) mindspore::device::gpu::CudaCommon::GetInstance().blocks_num(total_threads)
#define GET_THREADS mindspore::device::gpu::CudaCommon::GetInstance().threads_num()
#define GET_THREADS_MAXSIZE(size) mindspore::device::gpu::CudaCommon::GetInstance().threads_num(size)
#define GET_MAJOR_SM mindspore::device::gpu::CudaCommon::GetInstance().major_sm()
#define GET_CUDA_CAP mindspore::device::gpu::CudaCommon::GetInstance().cuda_cap()
#define SHARED_MEM_PER_BLOCK mindspore::device::gpu::CudaCommon::GetInstance().share_memory_size()
}  // namespace gpu
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_CUDA_COMMON_H_
