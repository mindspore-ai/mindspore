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

#ifndef MINDSPORE_CCSRC_KERNEL_GPU_FAKEQUANT_GPUKERNEL_H_
#define MINDSPORE_CCSRC_KERNEL_GPU_FAKEQUANT_GPUKERNEL_H_

#include <vector>
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
class FakeQuantGpuKernel : public GpuKernel {
 public:
  FakeQuantGpuKernel();
  ~FakeQuantGpuKernel() = default;

  const std::vector<size_t> &GetInputSizeList() const override;
  const std::vector<size_t> &GetOutputSizeList() const override;
  const std::vector<size_t> &GetWorkspaceSizeList() const override;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override;
  bool Init(const CNodePtr &kernel) override;

 protected:
  void InitSizeLists() override;

 private:
  size_t input_size_;
  size_t min_size_;
  size_t max_size_;
  size_t output_size_;
  size_t workspace_size_;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;

  int num_bits_;
  float quant_min_;
  float quant_max_;
  int quant_num_;
  int quant_delay_;
  bool ema_;
  float ema_decay_;
  int global_step_;
  bool training_;
  bool narrow_range_;
  bool symmetric_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_KERNEL_GPU_FAKEQUANT_GPUKERNEL_H_
