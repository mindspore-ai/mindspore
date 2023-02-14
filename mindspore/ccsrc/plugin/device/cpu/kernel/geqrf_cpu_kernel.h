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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_GEQRF_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_GEQRF_KERNEL_H_

#include <memory>
#include <vector>
#include <utility>
#include <algorithm>
#include <map>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class GeqrfCpuKernelMod : public NativeCpuKernelMod {
 public:
  GeqrfCpuKernelMod() = default;
  ~GeqrfCpuKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override {
    return kernel_func_(this, inputs, outputs);
  };

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);

  using GeqrfLaunchFunc = std::function<bool(GeqrfCpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                                             const std::vector<kernel::AddressPtr> &)>;

  static std::vector<std::pair<KernelAttr, GeqrfLaunchFunc>> func_list_;
  GeqrfLaunchFunc kernel_func_;

  template <typename T>
  void Larfg(size_t n, size_t vm, size_t vn, T *x, T *tau);

  template <typename T>
  std::unique_ptr<T[]> Larf(size_t m, size_t n, T *x, T *tau, std::unique_ptr<T[]> workspace, size_t cm, size_t cn);

  template <typename T>
  void Geqrf(size_t m, size_t n, T *x, T *tau);
  size_t num_m = 0;
  size_t num_n = 0;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_GEQRF_KERNEL_H_
