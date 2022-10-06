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

#include "plugin/device/cpu/kernel/eigen/cholesky_solve_cpu_kernel.h"
#include <Eigen/Dense>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
constexpr size_t kInputIndex0 = 0;
constexpr size_t kInputIndex1 = 1;
constexpr size_t kOutputIndex = 0;
constexpr size_t kDefalutRank = 2;
constexpr size_t kBatchRank = 3;
constexpr size_t kBatchIndex = 3;
constexpr size_t kRowIndex = 2;
constexpr size_t kColIndex = 1;
constexpr size_t kCholeskySolveInputNum = 2;
constexpr size_t kCholeskySolveOutputNum = 1;

void CholeskySolveCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  auto shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, kInputIndex0);
  if (IsDynamic(shape)) {
    return;
  }
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, kInputIndex0);
  std::vector<size_t> x1_shape = Convert2SizeT(shape);
  size_t rank = x1_shape.size();
  if (rank == kDefalutRank) {
    dim = x1_shape[rank - kRowIndex];
    rhs_dim = x1_shape[rank - kColIndex];
  } else {
    batch_size = x1_shape[rank - kBatchIndex];
    dim = x1_shape[rank - kRowIndex];
    rhs_dim = x1_shape[rank - kColIndex];
  }
  if (common::AnfAlgo::HasNodeAttr("upper", kernel_node)) {
    upper = common::AnfAlgo::GetNodeAttr<bool>(kernel_node, "upper");
  }
}

bool CholeskySolveCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                       const std::vector<kernel::AddressPtr> &,
                                       const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kCholeskySolveInputNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kCholeskySolveOutputNum, kernel_name_);
  if (dtype_ == kNumberTypeFloat32) {
    LaunchKernel<float>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat64) {
    LaunchKernel<double>(inputs, outputs);
  } else {
    MS_EXCEPTION(TypeError) << "Data type is " << TypeIdLabel(dtype_) << " which is not supported for CholeskySolve.";
  }
  return true;
}

template <typename T>
void CholeskySolveCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                             const std::vector<kernel::AddressPtr> &outputs) {
  T *rhsptr = reinterpret_cast<T *>(inputs[kInputIndex0]->addr);
  T *lhsptr = reinterpret_cast<T *>(inputs[kInputIndex1]->addr);
  T *outptr = reinterpret_cast<T *>(outputs[kOutputIndex]->addr);
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RHS(dim, rhs_dim);
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> LHS(dim, dim);
  for (size_t k = 0; k < batch_size; k++) {
    for (size_t i = 0; i < dim * rhs_dim; i++) {
      RHS.data()[i] = rhsptr[k * dim * rhs_dim + i];
    }
    for (size_t i = 0; i < dim * dim; i++) {
      LHS.data()[i] = lhsptr[k * dim * dim + i];
    }
    if (!upper) {
      LHS.template triangularView<Eigen::Lower>().solveInPlace(RHS);
      LHS.adjoint().template triangularView<Eigen::Upper>().solveInPlace(RHS);
    } else {
      LHS.adjoint().template triangularView<Eigen::Lower>().solveInPlace(RHS);
      LHS.template triangularView<Eigen::Upper>().solveInPlace(RHS);
    }
    for (size_t i = 0; i < dim * rhs_dim; i++) {
      outptr[k * dim * rhs_dim + i] = RHS.data()[i];
    }
  }
}

std::vector<KernelAttr> CholeskySolveCpuKernelMod::GetOpSupport() {
  static const std::vector<KernelAttr> support_list = {
    KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
    KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64)};
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, CholeskySolve, CholeskySolveCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
