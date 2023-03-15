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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_MAX_POOL_GRAD_WITH_ARGMAX_V2_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_MAX_POOL_GRAD_WITH_ARGMAX_V2_CPU_KERNEL_H_

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
class MaxPoolGradWithArgmaxV2CpuKernelMod : public NativeCpuKernelMod {
 public:
  MaxPoolGradWithArgmaxV2CpuKernelMod() = default;
  ~MaxPoolGradWithArgmaxV2CpuKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override {
    return kernel_func_(this, inputs, workspace, outputs);
  }

  std::vector<int64_t> GetValidAttr(const std::vector<int64_t> &src_attr);

  template <typename DATA_T>
  void OutPutInitKernel(DATA_T *output, size_t length) const;

  void CheckPadsValue(size_t k_width, size_t p_width, size_t k_height, size_t p_height) const;

  void CheckDilationValue(size_t d_width, size_t in_width, size_t d_height, size_t in_height) const;

  template <typename DATA_T, typename INDICES_T>
  void MaxPoolGradWithArgmaxV2SingleCompute(DATA_T *input_grad, INDICES_T *input_argmax, DATA_T *output_y, size_t iH,
                                            size_t iW, size_t oH, size_t oW, size_t kH, size_t kW, size_t sH, size_t sW,
                                            size_t pH, size_t pW, size_t dH, size_t dW);

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename DATA_T, typename INDICES_T>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                    const std::vector<AddressPtr> &outputs);

  using MaxPoolGradWithArgmaxV2Func =
    std::function<bool(MaxPoolGradWithArgmaxV2CpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                       const std::vector<kernel::AddressPtr> &, const std::vector<kernel::AddressPtr> &)>;

  static std::vector<std::pair<KernelAttr, MaxPoolGradWithArgmaxV2Func>> func_list_;
  MaxPoolGradWithArgmaxV2Func kernel_func_;
  std::vector<int64_t> x_shape_;
  std::vector<int64_t> y_shape_;
  std::vector<int64_t> grads_shape_;
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
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_MAX_POOL_GRAD_WITH_ARGMAX_V2_CPU_KERNEL_H_
