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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_LUUNPACK_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_LUUNPACK_CPU_KERNEL_H_
#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <utility>
#include <map>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
constexpr size_t kInputNum = 2;
constexpr size_t kOutputNum = 3;
namespace kernel {
class LuUnpackCpuKernelMod : public NativeCpuKernelMod {
 public:
  LuUnpackCpuKernelMod() = default;
  ~LuUnpackCpuKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs) override {
    return kernel_func_(this, inputs, outputs);
  }

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T_data, typename T_pivots>
  bool LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &outputs);

  template <typename T_data, typename T_pivots>
  static void LuUnpack(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &outputs,
                       int64_t Lu_data_dim1, int64_t Lu_pivots_dim, T_pivots *const Lu_pivots_working_ptr,
                       int64_t matrix_index, int64_t matrix_size, int64_t matrix_width, int64_t matrix_height,
                       int64_t pivots_stride, int64_t L_stride, int64_t U_stride, T_data *const P_eye);
  using LuUnpackFunc = std::function<bool(LuUnpackCpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                                          const std::vector<kernel::AddressPtr> &)>;

  std::vector<int64_t> input_0_shape_;
  std::vector<int64_t> input_1_shape_;
  static std::vector<std::pair<KernelAttr, LuUnpackFunc>> func_list_;
  LuUnpackFunc kernel_func_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_LUUNPACK_CPU_KERNEL_H_
