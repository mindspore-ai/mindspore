/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_CONTROL_RECV_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_CONTROL_RECV_GPU_KERNEL_H_

#include <vector>
#include "include/utils/utils.h"
#include "mindspore/ops/infer/stream_recv.h"
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
class RecvGpuKernelMod : public NativeGpuKernelMod {
 public:
  RecvGpuKernelMod() {}
  ~RecvGpuKernelMod() override = default;

  bool Launch(const std::vector<KernelTensor *> &, const std::vector<KernelTensor *> &,
              const std::vector<KernelTensor *> &, void *stream_ptr) override {
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamWaitEvent(reinterpret_cast<cudaStream_t>(stream_ptr), wait_event_, 0),
                                       "Waiting cuda event failed.");
    return true;
  }

  bool Init(const std::vector<KernelTensor *> &, const std::vector<KernelTensor *> &) override {
    auto prim = primitive_;
    MS_ERROR_IF_NULL(prim);
    wait_event_ = reinterpret_cast<cudaEvent_t>(GetValue<uintptr_t>(prim->GetAttr(kAttrWaitEvent)));
    return true;
  }

 private:
  cudaEvent_t wait_event_{nullptr};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_CONTROL_RECV_GPU_KERNEL_H_
