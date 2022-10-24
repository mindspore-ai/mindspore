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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_EIGEN_NUCLEAR_NORM_CPU_KERNEL_H
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_EIGEN_NUCLEAR_NORM_CPU_KERNEL_H

#include <utility>
#include <vector>
#include <map>

#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class NuclearNormCpuKernelMod : public NativeCpuKernelMod, public MatchKernelHelper<NuclearNormCpuKernelMod> {
 public:
  NuclearNormCpuKernelMod() = default;
  ~NuclearNormCpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override {
    MS_EXCEPTION_IF_NULL(kernel_func_);
    return kernel_func_(this, inputs, workspace, outputs);
  }

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

  const std::vector<std::pair<KernelAttr, KernelRunFunc>> &GetFuncList() const override;

 protected:
  std::vector<KernelAttr> GetOpSupport() override { return OpSupport(); }

  template <typename T>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<AddressPtr> &outputs);

 private:
  template <typename T>
  void GivensL(T *S_, const size_t dim[], const size_t m, const T a, const T b);

  template <typename T>
  void GivensR(T *S_, const size_t dim[], const size_t m, const T a, const T b);

  template <typename T>
  void SVD_head(size_t i, std::vector<T> *house_vec, const size_t dim[], T *U_, T *S_);

  template <typename T>
  void SVD(const size_t dim[], T *U_, T *S_, T *V_, T eps);

  template <typename T>
  void SVD_tail(const size_t dim[], T *U_, T *S_, T *V_, T eps);

  template <typename T>
  void SVD_tail_cal(const size_t dim[], T *U_, T *S_, T *V_, const T eps, const size_t n, const size_t k0, T alpha,
                    T beta, const T S_max);

  template <typename T>
  void svd(int *M, int *N, const T *A, const int *LDA, T *S, T *U, const int *LDU, T *VT, const int *LDVT);

  template <typename T>
  void svd_tail(const int *M, const int *N, T *S, const T *S_, T *U, T *VT, const T *U_, const T *V_,
                const size_t dim[], const int *LDU, const int LDVT);

  template <typename T>
  T ComputeMatrixNuclearNorm(size_t dim0, size_t dim1, const T *mat);

  template <typename T, int32_t RANK>
  bool ComputeTensorNuclearNorm(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);

  std::vector<int64_t> input_shape;
  std::vector<int64_t> dim_ = {0, 1};
  bool keepdim = false;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_EIGEN_NUCLEAR_NORM_CPU_KERNEL_H
