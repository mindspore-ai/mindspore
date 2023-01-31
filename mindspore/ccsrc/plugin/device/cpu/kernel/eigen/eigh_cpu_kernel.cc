/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/eigen/eigh_cpu_kernel.h"
#include <algorithm>
#include <tuple>
#include <map>
#include <type_traits>
#include "plugin/device/cpu/kernel/eigen/eigen_common_utils.h"
#include "Eigen/Eigenvalues"
#include "utils/ms_utils.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kInputsNum = 1;
constexpr size_t kOutputsNum = 2;
}  // namespace

bool EighCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                            const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->GetPrim()->name();
  dtype_ = inputs[0]->GetDtype();

  compute_eigen_vectors_ = GetValue<bool>(base_operator->GetAttr(C_EIEH_VECTOR));
  lower_ = GetValue<bool>(base_operator->GetAttr(LOWER));

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "Eigh does not support this kernel data type: " << kernel_attr;
  }
  kernel_func_ = std::get<1>(func_list_[index]);
  const size_t kTwoIdx = 2;
  init_io_func_ = std::get<kTwoIdx>(func_list_[index]);
  return true;
}

int EighCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                             const std::vector<KernelTensorPtr> &outputs,
                             const std::map<uint32_t, tensor::TensorPtr> &) {
  auto ret = KernelMod::Resize(base_operator, inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }

  auto A_shape = inputs[0]->GetShapeVector();
  m_ = LongToSize(A_shape[kDim0]);
  init_io_func_(this);
  return KRET_OK;
}

template <typename T>
void SolveSelfAdjointMatrix(const Map<MatrixSquare<T>> &A, Map<MatrixSquare<T>> *output, bool compute_eigen_vectors,
                            size_t m, T *output_v_addr = nullptr) {
  Eigen::SelfAdjointEigenSolver<MatrixSquare<T>> solver(
    A, compute_eigen_vectors ? Eigen::ComputeEigenvectors : Eigen::EigenvaluesOnly);
  output->noalias() = solver.eigenvalues();
  if (compute_eigen_vectors && output_v_addr != nullptr) {
    Map<MatrixSquare<T>> outputv(output_v_addr, m, m);
    outputv.noalias() = solver.eigenvectors();
  }
}

template <typename T>
void SolveComplexMatrix(const Map<MatrixSquare<T>> &A, Map<MatrixSquare<T>> *output, bool compute_eigen_vectors,
                        size_t m, T *output_v_addr = nullptr) {
  Eigen::ComplexEigenSolver<MatrixSquare<T>> solver(A, compute_eigen_vectors);
  output->noalias() = solver.eigenvalues();
  if (compute_eigen_vectors && output_v_addr != nullptr) {
    Map<MatrixSquare<T>> outputv(output_v_addr, m, m);
    outputv.noalias() = solver.eigenvectors();
  }
}

template <typename T>
void EighCpuKernelMod::InitIOFunc() {
  (void)workspace_size_list_.emplace_back(m_ * m_ * sizeof(T));
}

template <typename T>
bool EighCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                    const std::vector<AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputsNum, kernel_name_);
  auto A_addr = reinterpret_cast<T *>(inputs[0]->addr);
  // is the Matrix a symmetric matrix(true lower triangle, false upper triangle)
  auto output_addr = reinterpret_cast<T *>(outputs[0]->addr);
  T *a_Work_dir = reinterpret_cast<T *>(workspace[0]->addr);
  Map<MatrixSquare<T>> A(A_addr, m_, m_);
  Map<MatrixSquare<T>> A_(a_Work_dir, m_, m_);
  Map<MatrixSquare<T>> output(output_addr, m_, 1);
  // selfadjoint matrix
  if (lower_) {
    A_ = A.template selfadjointView<Lower>();
  } else {
    A_ = A.template selfadjointView<Upper>();
  }
  T *output_v_addr = nullptr;
  if (compute_eigen_vectors_) {
    output_v_addr = reinterpret_cast<T *>(outputs[1]->addr);
  }
  if constexpr (std::is_same<T, float>::value || std::is_same<T, double>::value) {
    SolveSelfAdjointMatrix(A_, &output, compute_eigen_vectors_, m_, output_v_addr);
  } else {
    SolveComplexMatrix(A_, &output, compute_eigen_vectors_, m_, output_v_addr);
  }
  return true;
}

std::vector<std::tuple<KernelAttr, EighCpuKernelMod::EighFunc, EighCpuKernelMod::EighInitFunc>>
  EighCpuKernelMod::func_list_ = {
    {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &EighCpuKernelMod::LaunchKernel<float>, &EighCpuKernelMod::InitIOFunc<float>},
    {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &EighCpuKernelMod::LaunchKernel<double>, &EighCpuKernelMod::InitIOFunc<double>},
    {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
     &EighCpuKernelMod::LaunchKernel<float_complex>, &EighCpuKernelMod::InitIOFunc<float_complex>},
    {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
     &EighCpuKernelMod::LaunchKernel<double_complex>, &EighCpuKernelMod::InitIOFunc<double_complex>}};

std::vector<KernelAttr> EighCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::tuple<KernelAttr, EighFunc, EighInitFunc> &item) { return std::get<0>(item); });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Eigh, EighCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
