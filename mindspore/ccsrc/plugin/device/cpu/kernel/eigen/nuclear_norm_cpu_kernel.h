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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_EIGEN_NUCLEAR_NORM_CPU_KERNEL_H
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_EIGEN_NUCLEAR_NORM_CPU_KERNEL_H

#include <utility>
#include <vector>

#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class NuclearNormCpuKernelMod : public DeprecatedNativeCpuKernelMod {
 public:
  NuclearNormCpuKernelMod() = default;
  ~NuclearNormCpuKernelMod() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override {
    return kernel_func_(this, inputs, outputs);
  }

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T>
  void GivensL(T *S_, const size_t dim[2], size_t m, T a, T b);

  template <typename T>
  void GivensR(T *S_, const size_t dim[2], size_t m, T a, T b);

  template <typename T>
  void SVD_head(size_t i, std::vector<T> *house_vec, const size_t dim[2], T *U_, T *S_);

  template <typename T>
  void SVD(const size_t dim[2], T *U_, T *S_, T *V_, T eps);

  template <typename T>
  void SVD_tail(const size_t dim[2], T *U_, T *S_, T *V_, T eps);

  template <typename T>
  void SVD_tail_cal(const size_t dim[2], T *U_, T *S_, T *V_, T eps, size_t n, size_t k0, T alpha, T beta, T S_max);

  template <typename T>
  void svd(int *M, int *N, T *A, int *LDA, T *S, T *U, int *LDU, T *VT, int *LDVT);

  template <typename T>
  void svd_tail(int *M, int *N, T *S, T *S_, T *U, T *VT, T *U_, T *V_, const size_t dim[2], int *LDU, int *LDVT);

  template <typename T>
  T ComputeMatrixNuclearNorm(int dim0, int dim1, T mat[]);

  template <typename T, int32_t RANK>
  bool ComputeTensorNuclearNorm(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);

  template <typename T>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);
  using NuclearNormFunc = std::function<bool(NuclearNormCpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                                             const std::vector<kernel::AddressPtr> &)>;
  static std::vector<std::pair<KernelAttr, NuclearNormFunc>> func_list_;
  NuclearNormFunc kernel_func_;

  std::vector<int64_t> input_shape;
  TypeId input_dtype{kTypeUnknown};
  std::vector<int64_t> dim = {0, 1};
  bool keepdim = false;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_EIGEN_NUCLEAR_NORM_CPU_KERNEL_H
