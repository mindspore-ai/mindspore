/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_RANK_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_RANK_CPU_KERNEL_H_

#include <vector>
#include <string>
#include <limits>
#include "backend/kernel_compiler/cpu/cpu_kernel.h"
#include "backend/kernel_compiler/cpu/cpu_kernel_factory.h"
#include "backend/kernel_compiler/cpu/nnacl/op_base.h"

namespace mindspore {
namespace kernel {
namespace rank {
enum Method : int {
  Average,
  Max,
  Min,
  First,
  Dense,
  MethodNotDefined,
};
enum NaOption : int {
  Keep,
  Top,
  Bottom,
  OptionNotDefined,
};
}  // namespace rank
template <typename T>
class RankCpuKernel : public CPUKernel {
 public:
  RankCpuKernel() = default;
  ~RankCpuKernel() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;
  void InitInputOutputSize(const CNodePtr &kernel_node) override;

  void SetFunc();

  void Launch1DInt(const T *input_addr, size_t *sort_idx, T *values, const AxisIterator &iter,
                   float *output_addr) const;
  void Launch1DFloat(const T *input_addr, size_t *sort_idx, T *values, bool *is_nan, const AxisIterator &iter,
                     float *output_addr) const;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 private:
  inline void SortIndex(size_t *sort_idx, const T *values, const AxisIterator &iter) const {
    std::iota(sort_idx, sort_idx + iter.AxisSize(), 0);
    if (ascending_) {
      std::stable_sort(sort_idx, sort_idx + iter.AxisSize(),
                       [values](size_t lhs, size_t rhs) { return values[lhs] < values[rhs]; });
    } else {
      std::stable_sort(sort_idx, sort_idx + iter.AxisSize(),
                       [values](size_t lhs, size_t rhs) { return values[lhs] > values[rhs]; });
    }
  }
  inline T get_padding_value() const {
    if (ascending_ != (option_ == rank::NaOption::Top)) {
      return std::numeric_limits<T>::max();
    } else {
      return std::numeric_limits<T>::min();
    }
  }
  // shape info
  AxisIterator axisIterator_{};
  // parameters
  size_t axis_{0};
  rank::Method method_{rank::MethodNotDefined};
  std::function<void(size_t, int, int, const AxisIterator &, const size_t *const, float *const)> func_;
  rank::NaOption option_{rank::OptionNotDefined};
  bool ascending_{true};
  bool pct_{false};
};

MS_REG_CPU_KERNEL_T(Rank, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                    RankCpuKernel, float)

MS_REG_CPU_KERNEL_T(Rank, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat32),
                    RankCpuKernel, double)

MS_REG_CPU_KERNEL_T(Rank, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32), RankCpuKernel,
                    int32_t)

MS_REG_CPU_KERNEL_T(Rank, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat32), RankCpuKernel,
                    int64_t)

}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_RANK_CPU_KERNEL_H_
