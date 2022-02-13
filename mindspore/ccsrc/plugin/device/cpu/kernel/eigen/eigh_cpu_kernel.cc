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

template <typename T>
void EighCpuKernelMod<T>::InitKernel(const CNodePtr &kernel_node) {
  kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
  dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
  compute_eigen_vectors_ = AnfAlgo::GetNodeAttr<bool>(kernel_node, C_EIEH_VECTOR);
  lower_ = AnfAlgo::GetNodeAttr<bool>(kernel_node, LOWER);
  auto A_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  if (A_shape.size() != kShape2dDims) {
    MS_LOG(EXCEPTION) << "Wrong array shape. For '" << kernel_name_ << "', a should be 2D, but got [" << A_shape.size()
                      << "] dimensions.";
  }
  if (A_shape[kDim0] != A_shape[kDim1]) {
    MS_LOG(EXCEPTION) << "Wrong array shape. For '" << kernel_name_
                      << "', a should be a squre matrix like [N X N], but got [" << A_shape[kDim0] << " X "
                      << A_shape[kDim1] << "].";
  }
  m_ = A_shape[kDim0];
}

template <typename T>
void SolveSelfAdjointMatrix(const Map<MatrixSquare<T>> &A, Map<MatrixSquare<T>> *output, Map<MatrixSquare<T>> *outputv,
                            bool compute_eigen_vectors) {
  Eigen::SelfAdjointEigenSolver<MatrixSquare<T>> solver(A);
  output->noalias() = solver.eigenvalues();
  if (compute_eigen_vectors) {
    outputv->noalias() = solver.eigenvectors();
  }
}

template <typename T>
void SolveComplexMatrix(const Map<MatrixSquare<T>> &A, Map<MatrixSquare<T>> *output, Map<MatrixSquare<T>> *outputv,
                        bool compute_eigen_vectors) {
  Eigen::ComplexEigenSolver<MatrixSquare<T>> solver(A);
  output->noalias() = solver.eigenvalues();
  if (compute_eigen_vectors) {
    outputv->noalias() = solver.eigenvectors();
  }
}

template <typename T>
bool EighCpuKernelMod<T>::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                 const std::vector<AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputsNum, kernel_name_);
  auto A_addr = reinterpret_cast<T *>(inputs[0]->addr);
  // is the Matrix a symmetric matrix(true lower triangle, false upper triangle)
  auto output_addr = reinterpret_cast<T *>(outputs[0]->addr);
  auto output_v_addr = reinterpret_cast<T *>(outputs[1]->addr);
  Map<MatrixSquare<T>> A(A_addr, m_, m_);
  Map<MatrixSquare<T>> A_(A_addr, m_, m_);
  Map<MatrixSquare<T>> output(output_addr, m_, 1);
  Map<MatrixSquare<T>> outputv(output_v_addr, m_, m_);
  // selfadjoint matrix
  if (lower_) {
    A_ = A.template selfadjointView<Lower>();
  } else {
    A_ = A.template selfadjointView<Upper>();
  }
  // Real scalar eigen solver
  if constexpr (std::is_same_v<T, float>) {
    SolveSelfAdjointMatrix(A_, &output, &outputv, compute_eigen_vectors_);
  } else if constexpr (std::is_same_v<T, double>) {
    SolveSelfAdjointMatrix(A_, &output, &outputv, compute_eigen_vectors_);
  } else {
    // complex eigen solver
    SolveComplexMatrix(A_, &output, &outputv, compute_eigen_vectors_);
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
