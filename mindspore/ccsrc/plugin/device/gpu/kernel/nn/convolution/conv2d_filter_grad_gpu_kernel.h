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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_CONV2D_FILTER_GRAD_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_CONV2D_FILTER_GRAD_GPU_KERNEL_H_

#include <algorithm>
#include <string>
#include <vector>
#include <utility>
#include <map>
#include <memory>

#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/kernel_constants.h"
#include "plugin/device/gpu/kernel/nn/convolution/conv_kernel_factory.h"

namespace mindspore {
namespace kernel {
class Conv2dFilterGradGpuKernelMod : public NativeGpuKernelMod, public MatchKernelHelper<Conv2dFilterGradGpuKernelMod> {
 public:
  Conv2dFilterGradGpuKernelMod() {}
  ~Conv2dFilterGradGpuKernelMod() override { DestroyResource(); }

  const std::vector<std::pair<KernelAttr, KernelRunFunc>> &GetFuncList() const override;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs,
             const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override;

  std::vector<KernelAttr> GetOpSupport() override { return OpSupport(); }

  std::vector<size_t> GetLaunchIgnoredInputAddressIdx() const override { return {kIndex2}; }

  void DestroyResource() noexcept override;

 private:
  template <typename T>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<AddressPtr> &outputs);

  void InitResource() override;
  void ResetResource() noexcept;

  std::string kernel_name_{"Conv2dGradFilter"};
  std::shared_ptr<AbstractConvolutionGpuKernel> conv_kernel_ptr_{nullptr};
  ConvolutionArgs conv_args_;
  void *stream_ptr_{nullptr};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_CONV2D_FILTER_GRAD_GPU_KERNEL_H_
