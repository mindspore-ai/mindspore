/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include <Eigen/Eigenvalues>
#include "utils/ms_utils.h"

namespace mindspore {
namespace kernel {

namespace {
constexpr size_t kInputsNum = 2;
constexpr size_t kOutputsNum = 2;
constexpr size_t kDefaultShape = 1;
constexpr auto kAMatrixDimNum = 2;

}  // namespace
using Eigen::Dynamic;
using Eigen::EigenSolver;
using Eigen::Lower;
using Eigen::Map;
using Eigen::MatrixBase;
using Eigen::RowMajor;
using Eigen::Upper;
template <typename T>
using MatrixSquare = Eigen::Matrix<T, Dynamic, Dynamic, RowMajor>;

template <typename T, typename C>
void EighCPUKernel<T, C>::InitKernel(const CNodePtr &kernel_node) {
  MS_LOG(INFO) << "init eigen value kernel";
  MS_EXCEPTION_IF_NULL(kernel_node);
  dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);

  compute_eigen_vectors = AnfAlgo::GetNodeAttr<bool>(kernel_node, C_EIEH_VECTOR);

  auto A_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  CHECK_KERNEL_INPUTS_NUM(A_shape.size(), kAMatrixDimNum, AnfAlgo::GetCNodeName(kernel_node));

  if (A_shape[kDim0] != A_shape[kDim1]) {
    MS_LOG(EXCEPTION) << "wrong array shape, A should be a   matrix, but got [" << A_shape[kDim0] << " X "
                      << A_shape[kDim1] << "]";
  }
  m_ = A_shape[kDim0];
}

template <typename T, typename C>
bool EighCPUKernel<T, C>::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                 const std::vector<AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputsNum, kernel_name_);

  auto A_addr = reinterpret_cast<T *>(inputs[0]->addr);
  // is the Matrix a symmetric matrix(0, all, general matxi, -1 lower triangle, 1 upper triangle)
  auto symmetric_type = reinterpret_cast<int *>(inputs[1]->addr);
  auto output_addr = reinterpret_cast<C *>(outputs[0]->addr);
  auto output_v_addr = reinterpret_cast<C *>(outputs[1]->addr);
  Map<MatrixSquare<T>> A(A_addr, m_, m_);
  Map<MatrixSquare<C>> output(output_addr, m_, 1);
  Map<MatrixSquare<C>> outputv(output_v_addr, m_, m_);
  if (*symmetric_type != 0) {
    if (*symmetric_type < 0) {
      A = A.template selfadjointView<Lower>();
    } else {
      A = A.template selfadjointView<Upper>();
    }
    Eigen::SelfAdjointEigenSolver<MatrixSquare<T>> solver(A);
    output.noalias() = solver.eigenvalues();
    if (compute_eigen_vectors) {
      outputv.noalias() = solver.eigenvectors();
    }
  } else {
    // this is for none symmetric matrix eigenvalue and eigen vectors, it should support complex
    Eigen::EigenSolver<MatrixSquare<T>> solver(A);
    output.noalias() = solver.eigenvalues();
    if (compute_eigen_vectors) {
      outputv.noalias() = solver.eigenvectors();
    }
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
