/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_KERNEL_GPU_CONTROL_RECV_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_KERNEL_GPU_CONTROL_RECV_GPU_KERNEL_H_

#include <vector>
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
class RecvGpuKernel : public GpuKernel {
 public:
  RecvGpuKernel() {}
  ~RecvGpuKernel() override = default;

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &, const std::vector<AddressPtr> &, const std::vector<AddressPtr> &,
              void *) override {
    CHECK_CUDA_RET_WITH_EXCEPT(cudaStreamWaitEvent(wait_stream_, wait_event_, 0), "Waiting cuda event failed.");
    return true;
  }
  bool Init(const CNodePtr &kernel_node) override {
    wait_stream_ = reinterpret_cast<cudaStream_t>(GetAttr<uintptr_t>(kernel_node, "wait_event_stream"));
    wait_event_ = reinterpret_cast<cudaEvent_t>(GetAttr<uintptr_t>(kernel_node, "wait_event"));
    InitSizeLists();
    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
    return;
  }

 private:
  cudaStream_t wait_stream_{nullptr};
  cudaEvent_t wait_event_{nullptr};

  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_KERNEL_GPU_CONTROL_RECV_GPU_KERNEL_H_
