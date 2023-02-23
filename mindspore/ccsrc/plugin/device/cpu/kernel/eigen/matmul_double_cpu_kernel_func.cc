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
}  // namespace

template <typename T>
void MatmulDoubleCpuKernelFunc::ComputeMatMulOutput(T *a_addr, T *b_addr, T *output_addr) const {
  using MatrixMap = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
  MatrixMap input0(a_addr, a_row_, a_col_);
  MatrixMap input1(b_addr, b_row_, b_col_);
  MatrixMap output(output_addr, out_row_, out_col_);
  if (kernel_name_ == prim::kPrimMatMul->name()) {
    if (trans_a_) {
      if (trans_b_) {
        output.noalias() = input0.adjoint() * input1.adjoint();
      } else {
        output.noalias() = input0.adjoint() * input1;
      }
    } else {
      if (trans_b_) {
        output.noalias() = input0 * input1.adjoint();
      } else {
        output.noalias() = input0 * input1;
      }
    }
  } else if (kernel_name_ == prim::kPrimBatchMatMul->name()) {
    if (trans_a_) {
      if (trans_b_) {
        output.noalias() = input0.adjoint() * input1.adjoint();
      } else {
        output.noalias() = input0.adjoint() * input1;
      }
    } else {
      if (trans_b_) {
        output.noalias() = input0 * input1.adjoint();
      } else {
        output.noalias() = input0 * input1;
      }
    }
  } else {
    MS_LOG(EXCEPTION) << "MatmulDoubleCpuKernelFunc support MatMul and BatchMatMul, but got " << kernel_name_ << ".";
  }
}

template <typename T>
void MatmulDoubleCpuKernelFunc::MatMul(const std::vector<kernel::AddressPtr> &inputs,
                                       const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kMatMulInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kMatMulOutputsNum, kernel_name_);
  if (batch_ > 1) {
    for (size_t index = 0; index < batch_; ++index) {
      const auto a_addr = reinterpret_cast<T *>(inputs[0]->addr) + index * a_row_ * a_col_;
      const auto b_addr = reinterpret_cast<T *>(inputs[1]->addr) + index * b_row_ * b_col_;
      auto output_addr = reinterpret_cast<T *>(outputs[0]->addr) + index * out_row_ * out_col_;
      ComputeMatMulOutput(a_addr, b_addr, output_addr);
    }
  } else {
    const auto a_addr = reinterpret_cast<T *>(inputs[0]->addr);
    const auto b_addr = reinterpret_cast<T *>(inputs[1]->addr);
    auto output_addr = reinterpret_cast<T *>(outputs[0]->addr);
    ComputeMatMulOutput(a_addr, b_addr, output_addr);
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

int MatmulDoubleCpuKernelFunc::Resize(const BaseOperatorPtr &, const std::vector<KernelTensorPtr> &inputs,
                                      const std::vector<KernelTensorPtr> &outputs,
                                      const std::map<uint32_t, tensor::TensorPtr> &) {
  auto a_shape = inputs[kIndex0]->GetShapeVector();
  auto b_shape = inputs[kIndex1]->GetShapeVector();
  auto out_shape = outputs[kIndex0]->GetShapeVector();
  if (a_shape.size() < kAMatrixDimNum || b_shape.size() < kAMatrixDimNum || out_shape.size() < kAMatrixDimNum) {
    MS_LOG(EXCEPTION) << "The tensor rank of MatMul must be greater than or equal to " << kAMatrixDimNum;
  }

  auto rank = a_shape.size();
  int64_t batch = 1;
  for (size_t i = 0; i < rank - kIndexOffset; ++i) {
    batch *= a_shape[i];
  }
  if (batch > 1 || rank > kIndexOffset) {
    batch_ = static_cast<size_t>(batch);
    rank_ = static_cast<size_t>(rank);
    a_row_ = static_cast<size_t>(a_shape[rank - kIndexOffset + kDim0]);
    a_col_ = static_cast<size_t>(a_shape[rank - kIndexOffset + kDim1]);
    b_row_ = static_cast<size_t>(b_shape[rank - kIndexOffset + kDim0]);
    b_col_ = static_cast<size_t>(b_shape[rank - kIndexOffset + kDim1]);
    out_row_ = static_cast<size_t>(out_shape[rank - kIndexOffset + kDim0]);
    out_col_ = static_cast<size_t>(out_shape[rank - kIndexOffset + kDim1]);
  } else {
    a_row_ = static_cast<size_t>(a_shape[kDim0]);
    a_col_ = static_cast<size_t>(a_shape[kDim1]);
    b_row_ = static_cast<size_t>(b_shape[kDim0]);
    b_col_ = static_cast<size_t>(b_shape[kDim1]);
    out_row_ = static_cast<size_t>(out_shape[kDim0]);
    out_col_ = static_cast<size_t>(out_shape[kDim1]);
  }
  dtype_ = inputs[kIndex0]->GetDtype();
  return KRET_OK;
}

bool MatmulDoubleCpuKernelFunc::RunFunc(const std::vector<kernel::AddressPtr> &inputs,
                                        const std::vector<kernel::AddressPtr> &,
                                        const std::vector<kernel::AddressPtr> &outputs) {
  if (dtype_ == kNumberTypeInt8) {
    MatMul<int8_t>(inputs, outputs);
  } else if (dtype_ == kNumberTypeInt16) {
    MatMul<int16_t>(inputs, outputs);
  } else if (dtype_ == kNumberTypeInt32) {
    MatMul<int32_t>(inputs, outputs);
  } else if (dtype_ == kNumberTypeInt64) {
    MatMul<int64_t>(inputs, outputs);
  } else if (dtype_ == kNumberTypeUInt8) {
    MatMul<uint8_t>(inputs, outputs);
  } else if (dtype_ == kNumberTypeUInt16) {
    MatMul<uint16_t>(inputs, outputs);
  } else if (dtype_ == kNumberTypeUInt32) {
    MatMul<uint32_t>(inputs, outputs);
  } else if (dtype_ == kNumberTypeUInt64) {
    MatMul<uint64_t>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat64) {
    MatMul<double>(inputs, outputs);
  } else if (dtype_ == kNumberTypeComplex64) {
    MatMul<complex64>(inputs, outputs);
  } else if (dtype_ == kNumberTypeComplex128) {
    MatMul<complex128>(inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << "Unsupported input data type.";
  }

  return true;
}
}  // namespace kernel
}  // namespace mindspore
