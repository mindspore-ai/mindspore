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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_ADAPTIVE_AVG_POOL2D_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_ADAPTIVE_AVG_POOL2D_GPU_KERNEL_H_

#include <vector>
#include <string>
#include <algorithm>
#include <utility>
#include <map>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/adaptive_avg_pool2d_impl.cuh"

namespace mindspore {
namespace kernel {

class AdaptiveAvgPool2DKernelMod : public NativeGpuKernelMod, public MatchKernelHelper<AdaptiveAvgPool2DKernelMod> {
 public:
  AdaptiveAvgPool2DKernelMod() = default;
  ~AdaptiveAvgPool2DKernelMod() override = default;

  const std::vector<std::pair<KernelAttr, KernelRunFunc>> &GetFuncList() const override;

  std::vector<KernelAttr> GetOpSupport() override { return OpSupport(); }

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs,
             const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override;

 private:
  template <typename T>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<AddressPtr> &outputs);

  uint len_{0};
  uint input_height_{0};
  uint input_width_{0};
  uint output_height_{0};
  uint output_width_{0};
  uint size_{0};
  bool is_null_input_{false};
  std::string kernel_name_{"AdaptiveAvgPool2D"};
  void *stream_ptr_{nullptr};
};
}  // namespace kernel
}  // namespace mindspore

#endif  //  MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_ADAPTIVE_AVG_POOL2D_GPU_KERNEL_H_
