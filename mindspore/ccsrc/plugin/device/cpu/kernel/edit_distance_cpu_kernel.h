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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_EDITDISTANCE_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_EDITDISTANCE_CPU_KERNEL_H_

#include <algorithm>
#include <map>
#include <memory>
#include <utility>
#include <vector>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class EditDistanceCpuKernelMod : public NativeCpuKernelMod {
 public:
  EditDistanceCpuKernelMod() = default;
  ~EditDistanceCpuKernelMod() override = default;

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

 private:
  template <typename T1, typename T2>
  bool LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<kernel::AddressPtr> &outputs);
  using EditDistanceFunc =
    std::function<bool(EditDistanceCpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                       const std::vector<kernel::AddressPtr> &, const std::vector<kernel::AddressPtr> &)>;
  bool normalize_{true};
  std::vector<std::vector<int64_t>> shapes_;
  EditDistanceFunc kernel_func_;
  static std::vector<std::pair<KernelAttr, EditDistanceFunc>> func_list_;
};

template <typename T, typename Cmp>
inline size_t LevenshteinDistance(const std::vector<T> &s, const std::vector<T> &t, const Cmp &cmp) {
  const size_t s_size = s.size();
  const size_t t_size = t.size();
  if (s_size == 0 || t_size == 0) {
    return std::max(s_size, t_size);
  }
  std::vector<std::vector<size_t>> dis(s_size + 1, std::vector<size_t>(t_size + 1, 0));
  for (size_t i = 0; i <= t_size; i++) {
    dis[0][i] = i;
  }
  for (size_t i = 0; i <= s_size; i++) {
    dis[i][0] = i;
  }
  for (size_t i = 1; i <= s_size; i++) {
    for (size_t j = 1; j <= t_size; j++) {
      dis[i][j] =
        std::min({dis[i - 1][j] + 1, dis[i][j - 1] + 1, dis[i - 1][j - 1] + (cmp(s[i - 1], t[j - 1]) ? 0 : 1)});
    }
  }
  return dis[s_size][t_size];
}

template <typename Container1, typename Container2, typename Cmp>
inline size_t LevenshteinDistance(const Container1 &s, const Container2 &t, const Cmp &cmp) {
  return LevenshteinDistance(std::vector<typename Container1::value_type>(s.data(), s.data() + s.size()),
                             std::vector<typename Container1::value_type>(t.data(), t.data() + t.size()), cmp);
}
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_EDITDISTANCE_CPU_KERNEL_H_
