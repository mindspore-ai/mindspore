/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_FRACTIONAL_MAX_POOL_GRAD_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_FRACTIONAL_MAX_POOL_GRAD_CPU_KERNEL_H_

#include <unordered_map>
#include <vector>
#include <random>
#include <limits>
#include <algorithm>
#include <memory>
#include <utility>
#include "Eigen/Core"
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class FractionalMaxPoolGradCpuKernelMod : public DeprecatedNativeCpuKernelMod {
 public:
  FractionalMaxPoolGradCpuKernelMod() = default;
  ~FractionalMaxPoolGradCpuKernelMod() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override {
    return kernel_func_(this, inputs, outputs);
  }

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T>
  bool FractionalMaxPoolGradLaunch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);
  template <typename T>
  bool FractionalMaxPoolGradOutput(int tensor_in_num, T *output, T *out_backprop, int back_in_nums, int output_nums,
                                   std::vector<int64_t> tensor_out_index);
  template <typename T>
  bool FractionalMaxPoolGradCompute(
    const int64_t row_start, int64_t row_end, size_t col_seq_num, size_t b, size_t hs, int64_t *col_seq,
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> tensor_in_mat,
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> output_mat,
    Eigen::Map<Eigen::Matrix<int64_t, Eigen::Dynamic, Eigen::Dynamic>> output_index_mat);
  using FractionalMaxPoolGradFunc =
    std::function<bool(FractionalMaxPoolGradCpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                       const std::vector<kernel::AddressPtr> &)>;
  static std::vector<std::pair<KernelAttr, FractionalMaxPoolGradFunc>> func_list_;
  FractionalMaxPoolGradFunc kernel_func_;
  std::vector<size_t> tensor_in_shape_;
  std::vector<size_t> tensor_out_shape_;
  bool overlapping_{false};
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_FRACTIONAL_MAX_POOL_GRAD_CPU_KERNEL_H_
