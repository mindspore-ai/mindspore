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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_RL_TENSORS_QUEUE_BASE_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_RL_TENSORS_QUEUE_BASE_H_

#include <vector>
#include <mutex>
#include <condition_variable>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/rl/tensors_queue_gpu_kernel.h"
#include "plugin/device/gpu/hal/device/gpu_tensor_array.h"
#include "runtime/device/tensor_array_manager.h"

namespace mindspore {
namespace kernel {
using mindspore::device::TensorsQueueMgr;
using mindspore::device::gpu::GPUTensorsQueue;
using mindspore::device::gpu::GPUTensorsQueuePtr;

class TensorsQueueBaseMod : public DeprecatedNativeGpuKernelMod {
 public:
  TensorsQueueBaseMod() = default;
  ~TensorsQueueBaseMod() = default;

  virtual bool Init(const CNodePtr &kernel_node) = 0;
  inline GPUTensorsQueuePtr GetTensorsQueue(const CNodeWeakPtr &kernel_node, const std::vector<AddressPtr> &inputs,
                                            cudaStream_t cuda_stream) {
    auto handle_addr = GetDeviceAddress<int64_t>(inputs, 0);
    MS_EXCEPTION_IF_NULL(handle_addr);

    int64_t handle = 0;
    CHECK_CUDA_RET_WITH_EXCEPT(
      kernel_node, cudaMemcpyAsync(&handle, handle_addr, sizeof(int64_t), cudaMemcpyDeviceToHost, cuda_stream),
      "Get handle to host failed");
    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node, cudaStreamSynchronize(cuda_stream),
                               "TensorsQueue cudaStreamSynchronized failed");

    auto tensors_q = std::dynamic_pointer_cast<GPUTensorsQueue>(TensorsQueueMgr::GetInstance().GetTensorsQueue(handle));
    MS_EXCEPTION_IF_NULL(tensors_q);
    return tensors_q;
  }

 protected:
  void InitSizeLists() {}
  // Lock the operation: Get, Pop, Size and Clear.
  static std::mutex tq_mutex_;
  static std::condition_variable read_cdv_;
  static std::condition_variable write_cdv_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_RL_TENSORS_QUEUE_BASE_H_
