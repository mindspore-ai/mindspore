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

#include "plugin/device/cpu/kernel/eigen/matrix_triangular_solve_cpu_kernel.h"
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
constexpr size_t kRowIndex = 2;
constexpr size_t kColIndex = 1;
template <typename T>
void MatrixTriangularSolveCpuKernelMod<T>::InitShape(const CNodePtr &kernel_node) {
  auto a_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  auto b_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
  // Since the shape check is done in frontend, we can suppose that the shape of a, b here is valid.
  size_t a_dims = a_shape.size();
  size_t aRowIndex = a_dims - kRowIndex;
  m_ = a_shape[aRowIndex];
  size_t b_sims = b_shape.size();
  bool vector_b = b_sims == a_dims - 1;
  if (vector_b) {
    n_ = 1;
  } else {
    n_ = b_shape[b_sims - 1];
  }
  batch_ = 1;
  for (size_t batch = 0; batch < a_dims - kRowIndex; ++batch) {
    batch_ *= a_shape[batch];
  }
}

template <typename T>
void MatrixTriangularSolveCpuKernelMod<T>::InitKernel(const CNodePtr &kernel_node) {
  kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
  InitShape(kernel_node);
  if (AnfAlgo::HasNodeAttr(ADJOINT, kernel_node)) {
    // MatrixTriangularSolve attribute
    trans_ = AnfAlgo::GetNodeAttr<bool>(kernel_node, ADJOINT);
    if (AnfAlgo::HasNodeAttr(TRANS, kernel_node)) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << "', the attribute 'adjoint' and 'trans' could not exist at the same time.";
    }
  } else {
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
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', 'trans' should be in ['N', 'T', 'C'], but got [" << trans
                        << "].";
    }
  }
}

template <typename Derived_a, typename Derived_b, typename T>
inline void solve(const MatrixBase<Derived_a> &a, const MatrixBase<Derived_b> &b, T *output_addr, int m, int n,
                  bool lower, bool unit_diagonal) {
  Map<Matrix<T, RowMajor>> output(output_addr, m, n);
  if (unit_diagonal) {
    if (lower) {
      output.noalias() = a.template triangularView<UnitLower>().solve(b);
    } else {
      output.noalias() = a.template triangularView<UnitUpper>().solve(b);
    }
  } else {
    if (lower) {
      output.noalias() = a.template triangularView<Lower>().solve(b);
    } else {
      output.noalias() = a.template triangularView<Upper>().solve(b);
    }
  }
}

template <typename T>
bool MatrixTriangularSolveCpuKernelMod<T>::Launch(const std::vector<AddressPtr> &inputs,
                                                  const std::vector<AddressPtr> & /* workspace */,
                                                  const std::vector<AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kSolveTriangularInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kSolveTriangularOutputsNum, kernel_name_);

  auto a_addr = reinterpret_cast<T *>(inputs[0]->addr);
  auto b_addr = reinterpret_cast<T *>(inputs[1]->addr);
  auto output_addr = reinterpret_cast<T *>(outputs[0]->addr);

  size_t a_batch_size = m_ * m_;
  size_t b_batch_size = m_ * n_;
  size_t output_batch_size = m_ * n_;

  for (size_t i = 0; i < batch_; ++i) {
    T *a_batch_addr = a_addr + i * a_batch_size;
    T *b_batch_addr = b_addr + i * b_batch_size;
    T *output_batch_addr = output_addr + i * output_batch_size;

    Map<Matrix<T, RowMajor>> b(b_batch_addr, m_, n_);
    if (trans_) {
      Map<Matrix<T, ColMajor>> a(a_batch_addr, m_, m_);
      solve(a, b, output_batch_addr, m_, n_, !lower_, unit_diagonal_);
    } else {
      Map<Matrix<T, RowMajor>> a(a_batch_addr, m_, m_);
      solve(a, b, output_batch_addr, m_, n_, lower_, unit_diagonal_);
    }
  }

  return true;
}
}  // namespace kernel
}  // namespace mindspore
