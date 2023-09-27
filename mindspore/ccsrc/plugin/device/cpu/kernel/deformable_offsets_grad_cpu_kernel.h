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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_DEFORMABLE_OFFSETS_GRAD_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_DEFORMABLE_OFFSETS_GRAD_CPU_KERNEL_H_

#include <algorithm>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <utility>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"
#include "mindspore/core/ops/grad/deformable_offsets_grad.h"

namespace mindspore {
namespace kernel {
using OpsDeformableOffsetsGradPtr = std::shared_ptr<ops::DeformableOffsetsGrad>;
struct DeformableOffsetGradDims {
  size_t x_n = 0;
  size_t x_h = 0;
  size_t x_w = 0;
  size_t offset_h = 0;
  size_t offset_w = 0;
  size_t grad_h = 0;
  size_t grad_w = 0;
  size_t kernel_h = 0;
  size_t kernel_w = 0;
  size_t pad_top = 0;
  size_t pad_left = 0;
  size_t stride_h = 0;
  size_t stride_w = 0;
  size_t dilation_h = 0;
  size_t dilation_w = 0;
  size_t deformable_group = 0;
  size_t deformable_group_channel = 0;
};

class DeformableOffsetsGradCpuKernelMod : public NativeCpuKernelMod,
                                          public MatchKernelHelper<DeformableOffsetsGradCpuKernelMod> {
 public:
  DeformableOffsetsGradCpuKernelMod() : deformable_kernel_operator_(nullptr), data_format_(kOpFormat_NCHW) {
    ResetResource();
  }
  ~DeformableOffsetsGradCpuKernelMod() override = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  const std::vector<std::pair<KernelAttr, KernelRunFunc>> &GetFuncList() const override;

  std::vector<KernelAttr> GetOpSupport() override { return MatchKernelHelper::OpSupport(); }

 private:
  void ResetResource() noexcept;

  void CheckInOutNum(size_t inputs_num, size_t outputs_num) const;

  void GetDataFormat();

  void SetDims(const std::vector<KernelTensor *> &inputs);

  template <typename T>
  bool LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs, const std::vector<kernel::KernelTensor *> &,
                    const std::vector<kernel::KernelTensor *> &outputs);

  template <typename T>
  void DeformableOffsetGradNHWCKernel(size_t num_kernels, const DeformableOffsetGradDims &dims, const T *input_x,
                                      const T *input_offset, const T *input_grad, T *output_grad_x,
                                      T *output_grad_offset);
  template <typename T>
  void DeformableOffsetGradNCHWKernel(size_t num_kernels, const DeformableOffsetGradDims &dims, const T *input_x,
                                      const T *input_offset, const T *input_grad, T *output_grad_x,
                                      T *output_grad_offset);
  OpsDeformableOffsetsGradPtr deformable_kernel_operator_;
  std::string data_format_ = kOpFormat_NCHW;
  DeformableOffsetGradDims dims_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_DEFORMABLE_OFFSETS_GRAD_CPU_KERNEL_H_
