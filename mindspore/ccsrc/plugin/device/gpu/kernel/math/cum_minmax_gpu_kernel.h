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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_MATH_CUM_MINMAX_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_MATH_CUM_MINMAX_GPU_KERNEL_H_

#include <vector>
#include <memory>
#include <utility>
#include <string>
#include <map>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cum_minmax_impl.cuh"
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class CumMinMaxGpuKernelMod : public NativeGpuKernelMod {
 public:
  CumMinMaxGpuKernelMod() = default;
  explicit CumMinMaxGpuKernelMod(const CumOpType &cum_op_type) : cum_op_type_(cum_op_type) {}
  ~CumMinMaxGpuKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    return kernel_func_(this, inputs, workspace, outputs, stream_ptr);
  }

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T, typename S>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<AddressPtr> &outputs, void *stream_ptr);

  using CumMinMaxLaunchFunc =
    std::function<bool(CumMinMaxGpuKernelMod *, const std::vector<AddressPtr> &, const std::vector<AddressPtr> &,
                       const std::vector<AddressPtr> &, void *)>;
  static std::map<CumOpType, std::vector<std::pair<KernelAttr, CumMinMaxLaunchFunc>>> func_list_;
  CumOpType cum_op_type_{CUM_OP_INVALID_TYPE};
  CumMinMaxLaunchFunc kernel_func_;
  int64_t axis_{0};
  size_t inner_size_{1};
  size_t axis_size_{1};
  size_t outer_size_{1};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_MATH_CUM_MINMAX_GPU_KERNEL_H_
