/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/eigen/matmul_double_cpu_kernel_func.h"
#include <Eigen/Dense>
#include <vector>
#include <map>
#include "mindspore/core/ops/mat_mul.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kMatMulInputsNum = 2;
constexpr size_t kMatMulOutputsNum = 1;
constexpr auto kAMatrixDimNum = 2;
const size_t kIndexOffset = 2;
using Eigen::ColMajor;
using Eigen::Map;
using Eigen::MatrixBase;
using Eigen::Ref;
using Eigen::RowMajor;
template <int Major>
using DoubleMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Major>;
}  // namespace

template <typename Derived>
inline void matmul_b(const MatrixBase<Derived> &A, double *b_addr, double *output_addr, size_t b_row_, size_t b_col_,
                     size_t out_row_, size_t out_col_, bool trans_b) {
  Map<DoubleMatrix<RowMajor>> output(output_addr, out_row_, out_col_);
  if (trans_b) {
    Map<DoubleMatrix<ColMajor>> b(b_addr, b_col_, b_row_);
    output.noalias() = A * b;
  } else {
    Map<DoubleMatrix<RowMajor>> b(b_addr, b_row_, b_col_);
    output.noalias() = A * b;
  }
}

void MatmulDoubleCpuKernelFunc::InitFunc(const BaseOperatorPtr &base_operator,
                                         const std::vector<KernelTensorPtr> &inputs,
                                         const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  auto kernel_ptr = std::dynamic_pointer_cast<ops::MatMul>(base_operator);
  if (!kernel_ptr) {
    MS_LOG(ERROR) << "cast MatMul ops failed!";
  }

  trans_a_ = kernel_ptr->get_transpose_a();
  trans_b_ = kernel_ptr->get_transpose_b();
}

int MatmulDoubleCpuKernelFunc::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                      const std::vector<KernelTensorPtr> &outputs,
                                      const std::map<uint32_t, tensor::TensorPtr> &) {
  auto a_shape = inputs[kIndex0]->GetShapeVector();
  auto b_shape = inputs[kIndex1]->GetShapeVector();
  auto out_shape = outputs[kIndex0]->GetShapeVector();
  if (a_shape.size() != kAMatrixDimNum || b_shape.size() != kAMatrixDimNum || out_shape.size() != kAMatrixDimNum) {
    MS_LOG(EXCEPTION) << "The tensor rank of MatMul must be equal to 2.";
  }

  a_row_ = static_cast<size_t>(a_shape[kDim0]);
  a_col_ = static_cast<size_t>(a_shape[kDim1]);
  b_row_ = static_cast<size_t>(b_shape[kDim0]);
  b_col_ = static_cast<size_t>(b_shape[kDim1]);
  out_row_ = static_cast<size_t>(out_shape[kDim0]);
  out_col_ = static_cast<size_t>(out_shape[kDim1]);
  return KRET_OK;
}

bool MatmulDoubleCpuKernelFunc::RunFunc(const std::vector<kernel::AddressPtr> &inputs,
                                        const std::vector<kernel::AddressPtr> &,
                                        const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kMatMulInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kMatMulOutputsNum, kernel_name_);
  const auto a_addr = reinterpret_cast<double *>(inputs[0]->addr);
  const auto b_addr = reinterpret_cast<double *>(inputs[1]->addr);
  auto output_addr = reinterpret_cast<double *>(outputs[0]->addr);

  if (trans_a_) {
    Map<DoubleMatrix<ColMajor>> A(a_addr, a_col_, a_row_);
    matmul_b(A, b_addr, output_addr, b_row_, b_col_, out_row_, out_col_, trans_b_);
  } else {
    Map<DoubleMatrix<RowMajor>> A(a_addr, a_row_, a_col_);
    matmul_b(A, b_addr, output_addr, b_row_, b_col_, out_row_, out_col_, trans_b_);
  }

  return true;
}
}  // namespace kernel
}  // namespace mindspore
