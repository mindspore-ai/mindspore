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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_EIGEN_MATRIX_SOLVE_LS_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_EIGEN_MATRIX_SOLVE_LS_CPU_KERNEL_H_

#include <vector>
#include <utility>
#include <complex>
#include <map>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class MatrixSolveLsCpuKernelMod : public NativeCpuKernelMod {
 public:
  MatrixSolveLsCpuKernelMod() = default;
  ~MatrixSolveLsCpuKernelMod() override = default;

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
  bool LaunchKernelAcessCheck(void);
  template <typename T>
  bool LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &workspace,
                    const std::vector<kernel::AddressPtr> &outputs);
  using MatrixSolveLsFunc =
    std::function<bool(MatrixSolveLsCpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                       const std::vector<kernel::AddressPtr> &, const std::vector<kernel::AddressPtr> &)>;
  static std::vector<std::pair<KernelAttr, MatrixSolveLsFunc>> func_list_;
  MatrixSolveLsFunc kernel_func_;

  template <typename T>
  void RealCholeskySingleCompute(T *aptr, T *bptr, T *xptr, double *l2, int64_t m, int64_t k, int64_t n);

  template <typename T>
  bool RealCholesky(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &outputs);

  template <typename T>
  void RealQrSingleCompute(T *aptr, T *bptr, T *xptr, int64_t m, int64_t k, int64_t n);

  template <typename T>
  bool RealQr(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &outputs);

  template <typename T>
  void ComplexCholeskySingleCompute(std::complex<T> *aptr, std::complex<T> *bptr, std::complex<T> *xptr, double *l2,
                                    int64_t m, int64_t k, int64_t n);

  template <typename T>
  bool ComplexCholesky(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &outputs);

  template <typename T>
  void ComplexQrSingleCompute(std::complex<T> *aptr, std::complex<T> *bptr, std::complex<T> *xptr, int64_t m, int64_t k,
                              int64_t n);

  template <typename T>
  bool ComplexQr(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &outputs);

  bool qr_chole_{true};
  TypeId matrix_dtype_{kTypeUnknown};
  TypeId rhs_dtype_{kTypeUnknown};
  std::vector<int64_t> matrix_shape_;
  std::vector<int64_t> rhs_shape_;
  std::vector<int64_t> l2_shape_;
  std::vector<int64_t> output_shape_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_EIGEN_MATRIX_SOLVE_LS_CPU_KERNEL_H_
