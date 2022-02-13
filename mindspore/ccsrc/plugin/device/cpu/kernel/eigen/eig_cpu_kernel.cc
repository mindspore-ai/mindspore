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

#include "plugin/device/cpu/kernel/eigen/eig_cpu_kernel.h"
#include <type_traits>
#include "plugin/device/cpu/kernel/eigen/eigen_common_utils.h"
#include "utils/ms_utils.h"
#include "Eigen/Eigenvalues"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kInputsNum = 1;
constexpr size_t kOutputsNumNV = 1;
constexpr size_t kOutputsNumV = 2;
}  // namespace

template <typename T, typename C>
void EigCpuKernelMod<T, C>::InitMatrixInfo(const std::vector<size_t> &shape) {
  if (shape.size() < kShape2dDims) {
    MS_LOG_EXCEPTION << "For '" << kernel_name_ << "', the rank of parameter 'a' must be at least 2, but got "
                     << shape.size() << " dimensions.";
  }
  row_size_ = shape[shape.size() - kDim1];
  col_size_ = shape[shape.size() - kDim2];
  if (row_size_ != col_size_) {
    MS_LOG_EXCEPTION << "For '" << kernel_name_
                     << "', the shape of parameter 'a' must be a square matrix, but got last two dimensions is "
                     << row_size_ << " and " << col_size_;
  }
  batch_size_ = 1;
  for (auto i : shape) {
    batch_size_ *= i;
  }
  batch_size_ /= (row_size_ * col_size_);
}

template <typename T, typename C>
void EigCpuKernelMod<T, C>::InitKernel(const CNodePtr &kernel_node) {
  kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
  // If compute_v_ is true, then: w, v = Eig(a)
  // If compute_v_ is false, then: w = Eig(a)
  if (AnfAlgo::HasNodeAttr(COMPUTE_V, kernel_node)) {
    compute_v_ = AnfAlgo::GetNodeAttr<bool>(kernel_node, COMPUTE_V);
  }
  size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
  CHECK_KERNEL_INPUTS_NUM(input_num, kInputsNum, kernel_name_);
  size_t output_num = AnfAlgo ::GetOutputTensorNum(kernel_node);
  auto expect_output_num = compute_v_ ? kOutputsNumV : kOutputsNumNV;
  CHECK_KERNEL_OUTPUTS_NUM(output_num, expect_output_num, kernel_name_);
  auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  InitMatrixInfo(input_shape);
}

template <typename T, typename C>
bool EigCpuKernelMod<T, C>::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                   const std::vector<AddressPtr> &outputs) {
  auto input_addr = reinterpret_cast<T *>(inputs[0]->addr);
  auto output_w_addr = reinterpret_cast<C *>(outputs[0]->addr);
  auto output_v_addr = compute_v_ ? reinterpret_cast<C *>(outputs[1]->addr) : nullptr;

  for (size_t batch = 0; batch < batch_size_; ++batch) {
    T *a_addr = input_addr + batch * row_size_ * col_size_;
    C *w_addr = output_w_addr + batch * row_size_;
    Map<MatrixSquare<T>> a(a_addr, row_size_, col_size_);
    Map<MatrixSquare<C>> w(w_addr, row_size_, 1);
    auto eigen_option = compute_v_ ? Eigen::ComputeEigenvectors : Eigen::EigenvaluesOnly;
    Eigen::ComplexEigenSolver<MatrixSquare<T>> solver(a, eigen_option);
    w = solver.eigenvalues();
    if (compute_v_) {
      C *v_addr = output_v_addr + batch * row_size_ * col_size_;
      Map<MatrixSquare<C>> v(v_addr, row_size_, col_size_);
      v = solver.eigenvectors();
    }
    if (solver.info() != Eigen::Success) {
      MS_LOG_WARNING << "For '" << kernel_name_
                     << "', the computation was not successful. Eigen::ComplexEigenSolver returns 'NoConvergence'.";
    }
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
