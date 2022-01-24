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

#include "backend/kernel_compiler/cpu/eigen/eig_cpu_kernel.h"
#include <type_traits>
#include "backend/kernel_compiler/cpu/eigen/eigen_common_utils.h"
#include "utils/ms_utils.h"
#include "Eigen/Eigenvalues"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kInputsNum = 1;
constexpr size_t kOutputsNum = 2;
}  // namespace

template <typename T, typename C>
void EigCpuKernelMod<T, C>::InitKernel(const CNodePtr &kernel_node) {
  dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
  compute_eigen_vectors = AnfAlgo::GetNodeAttr<bool>(kernel_node, C_EIEH_VECTOR);
  auto A_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  if (A_shape.size() != kShape2dDims || A_shape[0] != A_shape[1]) {
    MS_LOG(EXCEPTION) << "wrong array shape, A should be a  matrix, but got [" << A_shape[0] << " X " << A_shape[1]
                      << "]";
  }
  m_ = A_shape[0];
}

template <typename T, typename C>
void SolveGenericRealScalaMatrix(const Map<MatrixSquare<T>> &A, Map<MatrixSquare<C>> *output,
                                 Map<MatrixSquare<C>> *outputv, bool compute_eigen_vectors) {
  Eigen::EigenSolver<MatrixSquare<T>> solver(A);
  output->noalias() = solver.eigenvalues();
  if (compute_eigen_vectors) {
    outputv->noalias() = solver.eigenvectors();
  }
}

template <typename T, typename C>
void SolveComplexMatrix(const Map<MatrixSquare<T>> &A, Map<MatrixSquare<C>> *output, Map<MatrixSquare<C>> *outputv,
                        bool compute_eigen_vectors) {
  Eigen::ComplexEigenSolver<MatrixSquare<T>> solver(A);
  output->noalias() = solver.eigenvalues();
  if (compute_eigen_vectors) {
    outputv->noalias() = solver.eigenvectors();
  }
}

template <typename T, typename C>
bool EigCpuKernelMod<T, C>::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                   const std::vector<AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputsNum, kernel_name_);

  auto A_addr = reinterpret_cast<T *>(inputs[0]->addr);
  // is the Matrix a symmetric matrix(0, all, general matxi, -1 lower triangle, 1 upper triangle)
  auto output_addr = reinterpret_cast<C *>(outputs[0]->addr);
  auto output_v_addr = reinterpret_cast<C *>(outputs[1]->addr);
  Map<MatrixSquare<T>> A(A_addr, m_, m_);
  Map<MatrixSquare<C>> output(output_addr, m_, 1);
  Map<MatrixSquare<C>> outputv(output_v_addr, m_, m_);
  // Real scalar eigen solver
  if constexpr (std::is_same_v<T, float>) {
    SolveGenericRealScalaMatrix(A, &output, &outputv, compute_eigen_vectors);
  } else if constexpr (std::is_same_v<T, double>) {
    SolveGenericRealScalaMatrix(A, &output, &outputv, compute_eigen_vectors);
  } else {
    // complex eigen solver
    SolveComplexMatrix(A, &output, &outputv, compute_eigen_vectors);
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
