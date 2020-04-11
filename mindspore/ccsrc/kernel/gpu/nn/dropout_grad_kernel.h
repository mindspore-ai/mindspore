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

#ifndef MINDSPORE_CCSRC_KERNEL_GPU_NN_DROPOUT_GRAD_KERNEL_H_
#define MINDSPORE_CCSRC_KERNEL_GPU_NN_DROPOUT_GRAD_KERNEL_H_

#include <vector>
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
class DropoutGradGpuFwdKernel : public GpuKernel {
 public:
  DropoutGradGpuFwdKernel();
  ~DropoutGradGpuFwdKernel() override;

  const std::vector<size_t> &GetInputSizeList() const override;
  const std::vector<size_t> &GetOutputSizeList() const override;
  const std::vector<size_t> &GetWorkspaceSizeList() const override;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, uintptr_t stream_ptr) override;
  bool Init(const CNodePtr &kernel_node) override;

 protected:
  void InitResource() override;
  void InitSizeLists() override;

 private:
  void DestroyResource() noexcept;

  cudnnHandle_t cudnn_handle_;
  bool is_null_input_;
  size_t num_count_;
  float drop_prob_;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};

MS_REG_GPU_KERNEL(DropoutGrad, DropoutGradGpuFwdKernel)
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_KERNEL_GPU_NN_DROPOUT_GRAD_KERNEL_H_
