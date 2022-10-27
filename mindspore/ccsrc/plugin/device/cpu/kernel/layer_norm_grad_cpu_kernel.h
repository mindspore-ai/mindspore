/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_LAYER_NORM_GRAD_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_LAYER_NORM_GRAD_CPU_KERNEL_H_

#include <memory>
#include <vector>
#include <map>
#include <utility>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/device/cpu/kernel/mkldnn/mkl_cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class LayerNormGradCpuKernelMod : public MKLCpuKernelMod {
 public:
  LayerNormGradCpuKernelMod() = default;
  ~LayerNormGradCpuKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;
  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  void InitWorkspaceSize(const std::vector<KernelTensorPtr> &inputs);

  template <typename T>
  void LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);
  using KernelFunc =
    std::function<void(LayerNormGradCpuKernelMod *, const std::vector<AddressPtr> &, const std::vector<AddressPtr> &)>;
  KernelFunc kernel_func_{};
  static std::vector<std::pair<KernelAttr, KernelFunc>> func_list_;

  float eps_{1e-12};
  size_t block_num_{1};
  size_t block_size_{1};
  size_t param_num_{1};
  size_t param_size_{1};
  enum input_list_ { X, Y_BACKPROP, SAVE_VARIANCE, SAVE_MEAN, SCALE };
  enum workspace_list_ { SCALE_BIAS, DIFF_SCALE_BIAS };
  enum output_list_ { DX, DSCALE, DBIAS };
  bool use_onednn_{false};
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_LAYER_NORM_GRAD_CPU_KERNEL_H_
