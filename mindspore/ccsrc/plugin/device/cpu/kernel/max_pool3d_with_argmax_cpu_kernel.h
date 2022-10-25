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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_MAX_POOL3D_WITH_ARGMAX_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_MAX_POOL3D_WITH_ARGMAX_CPU_KERNEL_H_

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
class MaxPool3DWithArgmaxCpuKernelMod : public NativeCpuKernelMod {
 public:
  MaxPool3DWithArgmaxCpuKernelMod() = default;
  ~MaxPool3DWithArgmaxCpuKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override {
    return kernel_func_(this, inputs, workspace, outputs);
  }

  template <typename DATA_T>
  bool CheckIfLessOne(const std::vector<DATA_T> &inputs);

  template <typename DATA_T>
  bool CheckIfLessZero(const std::vector<DATA_T> &inputs);

  void CheckPadsValue(size_t k_width, size_t p_width, size_t k_height, size_t p_height, size_t k_depth, size_t p_depth);

  void CheckDilationValue(size_t d_width, size_t in_width, size_t d_height, size_t in_height, size_t d_depth,
                          size_t in_depth);

  template <typename DATA_T, typename INDICES_T>
  void MaxPool3DWithArgmaxSingleCompute(DATA_T *input, DATA_T *output_y, INDICES_T *output_argmax, int64_t iD,
                                        int64_t iH, int64_t iW, int64_t oD, int64_t oH, int64_t oW, int64_t kD,
                                        int64_t kH, int64_t kW, int64_t sD, int64_t sH, int64_t sW, int64_t pD,
                                        int64_t pH, int64_t pW, int64_t dD, int64_t dH, int64_t dW);

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename DATA_T, typename INDICES_T>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                    const std::vector<AddressPtr> &outputs);
  using MaxPool3DWithArgmaxFunc =
    std::function<bool(MaxPool3DWithArgmaxCpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                       const std::vector<kernel::AddressPtr> &, const std::vector<kernel::AddressPtr> &)>;
  static std::vector<std::pair<KernelAttr, MaxPool3DWithArgmaxFunc>> func_list_;
  MaxPool3DWithArgmaxFunc kernel_func_;
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
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_MAX_POOL3D_WITH_ARGMAX_CPU_KERNEL_H_
