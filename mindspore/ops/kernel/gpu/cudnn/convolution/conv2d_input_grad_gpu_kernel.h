/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_CONV2D_INPUT_GRAD_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_CONV2D_INPUT_GRAD_GPU_KERNEL_H_

#include <algorithm>
#include <string>
#include <vector>
#include <map>
#include <utility>
#include <memory>

#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/kernel_constants.h"
#include "include/utils/utils.h"
#include "kernel/gpu/cudnn/convolution/conv_kernel_factory.h"

namespace mindspore {
namespace kernel {
class Conv2dInputGradGpuKernelMod : public NativeGpuKernelMod, public MatchKernelHelper<Conv2dInputGradGpuKernelMod> {
 public:
  Conv2dInputGradGpuKernelMod() {}
  ~Conv2dInputGradGpuKernelMod() override { DestroyResource(); }

  const std::vector<std::pair<KernelAttr, KernelRunFunc>> &GetFuncList() const override;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override;

  std::vector<KernelAttr> GetOpSupport() override { return OpSupport(); }

  std::vector<size_t> GetLaunchIgnoredInputAddressIdx() const override { return {kIndex2}; }

  void DestroyResource() noexcept override;

 private:
  template <typename T>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<KernelTensor *> &outputs);
  void ResetResource() noexcept;
  void InitResource() override;

  std::string kernel_name_{"Conv2dGradInput"};
  std::shared_ptr<AbstractConvolutionGpuKernel> conv_kernel_ptr{nullptr};
  ConvolutionArgs conv_args_;
  void *stream_ptr_{nullptr};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_CONV2D_INPUT_GRAD_GPU_KERNEL_H_
