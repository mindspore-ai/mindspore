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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_MAX_POOL_WITH_ARGMAX_V2_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_MAX_POOL_WITH_ARGMAX_V2_CPU_KERNEL_H_

#include <map>
#include <algorithm>
#include <functional>
#include <numeric>
#include <memory>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>
#include <utility>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class MaxPoolWithArgmaxV2CpuKernelMod : public NativeCpuKernelMod {
 public:
  MaxPoolWithArgmaxV2CpuKernelMod() = default;
  ~MaxPoolWithArgmaxV2CpuKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override {
    return kernel_func_(this, inputs, workspace, outputs);
  }

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

  template <typename DATA_T, typename INDICES_T>
  void MaxPoolWithArgmaxV2SingleCompute(DATA_T *input, DATA_T *output_y, INDICES_T *output_argmax, int64_t i,
                                        int64_t in_channel, int64_t iH, int64_t iW, int64_t oH, int64_t oW, int64_t kH,
                                        int64_t kW, int64_t sH, int64_t sW, int64_t pH, int64_t pW, int64_t dH,
                                        int64_t dW) const;

 private:
  std::vector<int64_t> GetValidAttr(const std::vector<int64_t> &src_attr);
  template <typename DATA_T, typename INDICES_T>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                    const std::vector<AddressPtr> &outputs);
  using MaxPoolWithArgmaxV2Func =
    std::function<bool(MaxPoolWithArgmaxV2CpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                       const std::vector<kernel::AddressPtr> &, const std::vector<kernel::AddressPtr> &)>;
  static std::vector<std::pair<KernelAttr, MaxPoolWithArgmaxV2Func>> func_list_;
  MaxPoolWithArgmaxV2Func kernel_func_;
  std::vector<int64_t> x_shape_;
  std::vector<int64_t> y_shape_;
  std::vector<int64_t> argmax_shape_;
  std::vector<int64_t> ksize_list_;
  std::vector<int64_t> strides_list_;
  std::vector<int64_t> pads_list_;
  std::vector<int64_t> dilation_list_;
  TypeId x_dtype_{kTypeUnknown};
  TypeId argmax_dtype_{kTypeUnknown};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_MAX_POOL_WITH_ARGMAX_V2_CPU_KERNEL_H_
