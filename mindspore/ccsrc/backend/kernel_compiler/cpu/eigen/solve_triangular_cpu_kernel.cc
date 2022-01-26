/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "backend/kernel_compiler/cpu/eigen/solve_triangular_cpu_kernel.h"
#include <Eigen/Dense>
#include <vector>
#include <string>
namespace mindspore {
namespace kernel {
using Eigen::ColMajor;
using Eigen::Dynamic;
using Eigen::Lower;
using Eigen::Map;
using Eigen::MatrixBase;
using Eigen::RowMajor;
using Eigen::UnitLower;
using Eigen::UnitUpper;
using Eigen::Upper;
template <typename T, int Major>
using Matrix = Eigen::Matrix<T, Dynamic, Dynamic, Major>;
constexpr auto kSolveTriangularInputsNum = 2;
constexpr auto kSolveTriangularOutputsNum = 1;
constexpr auto kAVectorxDimNum = 1;
constexpr auto kAMatrixDimNum = 2;
template <typename T>
void SolveTriangularCpuKernelMod<T>::InitKernel(const CNodePtr &kernel_node) {
  auto A_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  auto b_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);

  if (A_shape.size() != kAMatrixDimNum) {
    MS_LOG(EXCEPTION) << "Wrong array shape, A should be 2D, but got [" << A_shape.size() << "] dimensions.";
  }
  if (A_shape[kDim0] != A_shape[kDim1]) {
    MS_LOG(EXCEPTION) << "Wrong array shape, A should be a squre matrix, but got [" << A_shape[kDim0] << " X "
                      << A_shape[kDim1] << "].";
  }
  m_ = A_shape[kDim0];

  if (b_shape.size() != kAVectorxDimNum && b_shape.size() != kAMatrixDimNum) {
    MS_LOG(EXCEPTION) << "Wrong array shape, b should be 1D or 2D, but got [" << b_shape.size() << "] dimensions.";
  }
  if (SizeToInt(b_shape[kDim0]) != m_) {
    MS_LOG(EXCEPTION) << "Wrong array shape, b should match the shape of A, excepted [" << m_ << "] but got ["
                      << b_shape[kDim0] << "].";
  }
  if (b_shape.size() == kAVectorxDimNum || (b_shape.size() == kAMatrixDimNum && b_shape[kDim1] == 1)) {
    n_ = 1;
  } else {
    n_ = b_shape[kDim1];
  }
  lower_ = AnfAlgo::GetNodeAttr<bool>(kernel_node, LOWER);
  unit_diagonal_ = AnfAlgo::GetNodeAttr<bool>(kernel_node, UNIT_DIAGONAL);
  const std::string trans = AnfAlgo::GetNodeAttr<std::string>(kernel_node, TRANS);
  if (trans == "N") {
    trans_ = false;
  } else if (trans == "T") {
    trans_ = true;
  } else if (trans == "C") {
    trans_ = true;
  } else {
    MS_LOG(EXCEPTION) << "Trans should be in [N, T, C], but got [" << trans << "].";
  }
}

template <typename Derived_A, typename Derived_b, typename T>
inline void solve(const MatrixBase<Derived_A> &A, const MatrixBase<Derived_b> &b, T *output_addr, int m, int n,
                  bool lower, bool unit_diagonal) {
  Map<Matrix<T, RowMajor>> output(output_addr, m, n);
  if (unit_diagonal) {
    if (lower) {
      output.noalias() = A.template triangularView<UnitLower>().solve(b);
    } else {
      output.noalias() = A.template triangularView<UnitUpper>().solve(b);
    }
  } else {
    if (lower) {
      output.noalias() = A.template triangularView<Lower>().solve(b);
    } else {
      output.noalias() = A.template triangularView<Upper>().solve(b);
    }
  }
}

template <typename T>
bool SolveTriangularCpuKernelMod<T>::Launch(const std::vector<AddressPtr> &inputs,
                                            const std::vector<AddressPtr> & /* workspace */,
                                            const std::vector<AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kSolveTriangularInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kSolveTriangularOutputsNum, kernel_name_);

  auto A_addr = reinterpret_cast<T *>(inputs[0]->addr);
  auto b_addr = reinterpret_cast<T *>(inputs[1]->addr);
  auto output_addr = reinterpret_cast<T *>(outputs[0]->addr);

  Map<Matrix<T, RowMajor>> b(b_addr, m_, n_);

  if (trans_) {
    Map<Matrix<T, ColMajor>> A(A_addr, m_, m_);
    solve(A, b, output_addr, m_, n_, !lower_, unit_diagonal_);
  } else {
    Map<Matrix<T, RowMajor>> A(A_addr, m_, m_);
    solve(A, b, output_addr, m_, n_, lower_, unit_diagonal_);
  }

  return true;
}
}  // namespace kernel
}  // namespace mindspore
