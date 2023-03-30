/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include <utility>
#include <algorithm>
#include "plugin/device/cpu/kernel/tridiagonalsolve_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "ops/tridiagonal_solve.h"
#include "Eigen/Core"
#include "Eigen/LU"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t InputSize = 2;
constexpr size_t OutputSize = 1;
constexpr size_t AxisNumber = 3;
constexpr int ThirdColomnOfU = 2;
constexpr size_t LastSecondRowOfU = 2;
constexpr int64_t ParallelDataNumSameShape = 8 * 1024;
}  // namespace

bool TridiagonalSolveCPUKernelMod::Init(const BaseOperatorPtr &base_operator,
                                        const std::vector<KernelTensorPtr> &inputs,
                                        const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();

  diag_dtype_ = inputs.at(kIndex0)->GetDtype();

  auto kernel_ptr = std::dynamic_pointer_cast<ops::TridiagonalSolve>(base_operator);
  MS_EXCEPTION_IF_NULL(kernel_ptr);
  partial_pivoting_ = kernel_ptr->get_partial_pivoting();

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_EXCEPTION(TypeError) << "For TridiagonalSolve, does not support this kernel data type: " << kernel_attr << ".";
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int TridiagonalSolveCPUKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                         const std::vector<KernelTensorPtr> &inputs,
                                         const std::vector<KernelTensorPtr> &outputs,
                                         const std::map<uint32_t, tensor::TensorPtr> &) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  input0_shape = inputs.at(kIndex0)->GetShapeVector();
  input1_shape = inputs.at(kIndex1)->GetShapeVector();
  n_ = input0_shape[input0_shape.size() - 1];
  batch_ = input1_shape[input1_shape.size() - 1];
  diags_size_ = AxisNumber * batch_;
  rhs_size_ = batch_ * n_;
  return KRET_OK;
}

template <typename T>
bool TridiagonalSolveCPUKernelMod::DoComputeWithPartPivoting_(const std::vector<AddressPtr> &inputs,
                                                              const std::vector<AddressPtr> &outputs, size_t nth_batch,
                                                              int i) {
  T *a = reinterpret_cast<T *>(inputs[0]->addr);
  MS_EXCEPTION_IF_NULL(a);
  T *b = reinterpret_cast<T *>(inputs[1]->addr);
  MS_EXCEPTION_IF_NULL(b);
  T *value = reinterpret_cast<T *>(outputs[0]->addr);
  MS_EXCEPTION_IF_NULL(value);

  if (i == -1) {
    a += nth_batch * IntToSize(diags_size_);
    b += nth_batch * IntToSize(rhs_size_);
    value += nth_batch * IntToSize(rhs_size_);
  } else {
    a += i * diags_size_;
    b += i * rhs_size_;
    value += i * rhs_size_;
  }

  const T zero = 0;

  Eigen::Array<T, Eigen::Dynamic, AxisNumber> u(n_, AxisNumber);
  Eigen::Array<T, Eigen::Dynamic, 1> superdiag(n_);
  Eigen::Array<T, Eigen::Dynamic, 1> diag(n_);
  Eigen::Array<T, Eigen::Dynamic, 1> subdiag(n_);
  Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> rhs(n_, batch_);
  Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> x(n_, batch_);

  for (int i = 0; i < n_; i++) {
    for (int j = 0; j < batch_; j++) {
      rhs(i, j) = *(b + i * batch_ + j);
    }
  }
  for (int i = 0; i < n_; i++) {
    superdiag(i) = *(a + i);
    diag(i) = *(a + n_ + i);
    subdiag(i) = *(a + n_ + n_ + i);
  }

  u(0, 0) = diag(0);
  u(0, 1) = superdiag(0);
  x.row(0) = rhs.row(0);
  for (int i = 0; i < n_ - 1; ++i) {
    if (abs(u(i, 0)) >= abs(subdiag(i + 1))) {
      if (u(i, 0) == zero) {
        MS_EXCEPTION(ValueError) << "For TridiagonalSolve, the first element of diag should not be zero.";
      }
      const T factor = subdiag(i + 1) / u(i, 0);
      u(i + 1, 0) = diag(i + 1) - factor * u(i, 1);
      x.row(i + 1) = rhs.row(i + 1) - factor * x.row(i);
      if (i != n_ - ThirdColomnOfU) {
        u(i + 1, 1) = superdiag(i + 1);
        u(i, ThirdColomnOfU) = 0;
      }
    } else {
      const T factor = u(i, 0) / subdiag(i + 1);
      u(i, 0) = subdiag(i + 1);
      u(i + 1, 0) = u(i, 1) - factor * diag(i + 1);
      u(i, 1) = diag(i + 1);
      x.row(i + 1) = x.row(i) - factor * rhs.row(i + 1);
      x.row(i) = rhs.row(i + 1);
      if (i != n_ - ThirdColomnOfU) {
        u(i, ThirdColomnOfU) = superdiag(i + 1);
        u(i + 1, 1) = -factor * superdiag(i + 1);
      }
    }
  }
  if (u(n_ - 1, 0) == zero) {
    MS_EXCEPTION(ValueError) << "For TridiagonalSolve, the last element of diag should not be zero.";
  }
  x.row(n_ - 1) /= u(n_ - 1, 0);
  for (int j = 0; j < batch_; j++) {
    *(value + (n_ - 1) * batch_ + j) = x(n_ - 1, j);
  }
  x.row(n_ - LastSecondRowOfU) =
    (x.row(n_ - LastSecondRowOfU) - u(n_ - LastSecondRowOfU, 1) * x.row(n_ - 1)) / u(n_ - LastSecondRowOfU, 0);
  for (int j = 0; j < batch_; j++) {
    *(value + (n_ - LastSecondRowOfU) * batch_ + j) = x(n_ - LastSecondRowOfU, j);
  }

  for (int i = n_ - 3; i >= 0; --i) {
    x.row(i) = (x.row(i) - u(i, 1) * x.row(i + 1) - u(i, LastSecondRowOfU) * x.row(i + 1 + 1)) / u(i, 0);
    for (int j = 0; j < batch_; j++) {
      *(value + i * batch_ + j) = x(i, j);
    }
  }

  return true;
}

template <typename T>
bool TridiagonalSolveCPUKernelMod::DoComputeWithoutPartPivoting_(const std::vector<AddressPtr> &inputs,
                                                                 const std::vector<AddressPtr> &outputs,
                                                                 size_t nth_batch, int i) {
  T *a = reinterpret_cast<T *>(inputs[0]->addr);
  MS_EXCEPTION_IF_NULL(a);
  T *b = reinterpret_cast<T *>(inputs[1]->addr);
  MS_EXCEPTION_IF_NULL(b);
  T *value = reinterpret_cast<T *>(outputs[0]->addr);
  MS_EXCEPTION_IF_NULL(value);
  if (i == -1) {
    a += nth_batch * IntToSize(diags_size_);
    b += nth_batch * IntToSize(rhs_size_);
    value += nth_batch * IntToSize(rhs_size_);
  } else {
    a += i * diags_size_;
    b += i * rhs_size_;
    value += i * rhs_size_;
  }

  Eigen::Array<T, Eigen::Dynamic, AxisNumber> u(n_, AxisNumber);

  Eigen::Array<T, Eigen::Dynamic, 1> superdiag(n_);
  Eigen::Array<T, Eigen::Dynamic, 1> diag(n_);
  Eigen::Array<T, Eigen::Dynamic, 1> subdiag(n_);

  Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> rhs(n_, batch_);

  Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> x(n_, batch_);

  const T zero = 0;

  for (int i = 0; i < n_; i++) {
    for (int j = 0; j < batch_; j++) {
      rhs(i, j) = *(b + i * batch_ + j);
    }
  }

  for (int i = 0; i < n_; i++) {
    superdiag(i) = *(a + i);
    diag(i) = *(a + n_ + i);
    subdiag(i) = *(a + n_ + n_ + i);
  }

  if (diag(0) == zero) {
    MS_EXCEPTION(ValueError) << "For TridiagonalSolve, the first element of diag should not be zero.";
  }

  u(0) = superdiag(0) / diag(0);
  x.row(0) = rhs.row(0) / diag(0);
  for (int i = 1; i < n_; ++i) {
    auto denom = diag(i) - subdiag(i) * u(i - 1);
    if (denom == zero) {
      MS_EXCEPTION(ValueError) << "For TridiagonalSolve, the diag should not be zero.";
    }
    u(i) = superdiag(i) / denom;
    x.row(i) = (rhs.row(i) - subdiag(i) * x.row(i - 1)) / denom;
  }
  for (int i = n_ - 2; i >= 0; --i) {
    x.row(i) -= u(i) * x.row(i + 1);
  }

  for (int i = 0; i < n_; i++) {
    for (int j = 0; j < batch_; j++) {
      *(value + i * batch_ + j) = x(i, j);
    }
  }

  return true;
}

bool TridiagonalSolveCPUKernelMod::ChooseDataType_(const std::vector<AddressPtr> &inputs,
                                                   const std::vector<AddressPtr> &outputs, size_t nth_batch, int i) {
  if (partial_pivoting_) {
    if (dtype_ == kNumberTypeFloat32) {
      res_ = DoComputeWithPartPivoting_<float>(inputs, outputs, nth_batch, i);
    } else if (dtype_ == kNumberTypeFloat64) {
      res_ = DoComputeWithPartPivoting_<double>(inputs, outputs, nth_batch, i);
    } else if (dtype_ == kNumberTypeComplex64) {
      res_ = DoComputeWithPartPivoting_<std::complex<float>>(inputs, outputs, nth_batch, i);
    } else if (dtype_ == kNumberTypeComplex128) {
      res_ = DoComputeWithPartPivoting_<std::complex<double>>(inputs, outputs, nth_batch, i);
    } else {
      MS_EXCEPTION(TypeError) << "For TridiagonalSolve, kernel data type " << TypeIdLabel(dtype_) << " not support.";
    }
  } else {
    if (dtype_ == kNumberTypeFloat32) {
      res_ = DoComputeWithoutPartPivoting_<float>(inputs, outputs, nth_batch, i);
    } else if (dtype_ == kNumberTypeFloat64) {
      res_ = DoComputeWithoutPartPivoting_<double>(inputs, outputs, nth_batch, i);
    } else if (dtype_ == kNumberTypeComplex64) {
      res_ = DoComputeWithoutPartPivoting_<std::complex<float>>(inputs, outputs, nth_batch, i);
    } else if (dtype_ == kNumberTypeComplex128) {
      res_ = DoComputeWithoutPartPivoting_<std::complex<double>>(inputs, outputs, nth_batch, i);
    } else {
      MS_EXCEPTION(TypeError) << "For TridiagonalSolve, kernel data type " << TypeIdLabel(dtype_) << " not support.";
    }
  }
  return res_;
}

bool TridiagonalSolveCPUKernelMod::CheckInputValue_(const std::vector<kernel::AddressPtr> &inputs,
                                                    const std::vector<AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), InputSize, "TridiagonalSolve");
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), OutputSize, "TridiagonalSolve");

  size_t thedimensionofinput0_shape = 2;
  if (input0_shape.size() < thedimensionofinput0_shape) {
    MS_EXCEPTION(ValueError) << "For TridiagonalSolve, expected diags to have rank at least 2, got "
                             << input0_shape.size() << ".";
  }

  if (input1_shape.size() != input0_shape.size()) {
    MS_EXCEPTION(ValueError) << "For TridiagonalSolve, expected the rank of diagonals and rhs to be the same, but got "
                             << input0_shape.size() << " and " << input1_shape.size() << ".";
  }
  size_t numberforlastseconddim = 2;
  if (input0_shape[input0_shape.size() - numberforlastseconddim] != AxisNumber) {
    MS_EXCEPTION(ValueError) << "For TridiagonalSolve, expected 3 diagonals got "
                             << input0_shape[input0_shape.size() - numberforlastseconddim] << ".";
  }

  for (size_t i = 0; i < input0_shape.size() - numberforlastseconddim; i++) {
    if (input0_shape[i] != input1_shape[i]) {
      MS_EXCEPTION(ValueError) << "For TridiagonalSolve, batch shapes of diags and rhs are incompatible.";
    }
  }

  dtype_ = diag_dtype_;
  int dim_size1 = input0_shape[input0_shape.size() - 1];
  int rhs_size0 = input1_shape[input1_shape.size() - numberforlastseconddim];
  if (dim_size1 != rhs_size0) {
    MS_EXCEPTION(ValueError) << "For TridiagonalSolve, the length of diags and rhs are incompatible.";
  }
  return true;
}

template <typename T>
bool TridiagonalSolveCPUKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                                const std::vector<kernel::AddressPtr> &outputs) {
  res_ = CheckInputValue_(inputs, outputs);
  if (!res_) {
    return res_;
  }

  int64_t m = input0_shape[input0_shape.size() - 1];
  int64_t size_m3 = m * 3;
  int64_t k = input1_shape[input1_shape.size() - 1];

  diags_size_ = size_m3;
  rhs_size_ = m * k;

  if (size_m3 > 0) {
    size_t input_num = 1;
    for (size_t i = 0; i < input0_shape.size(); i++) {
      input_num *= static_cast<size_t>(input0_shape[i]);
    }
    size_t matrix_num = input_num / LongToSize(size_m3);
    int64_t data_size = SizeToLong(input_num);
    if (data_size >= ParallelDataNumSameShape) {
      auto shared_tridiagonalsolve = [&](size_t start, size_t end) {
        for (size_t nth_batch = start; nth_batch < end; nth_batch++)
          res_ = ChooseDataType_(inputs, outputs, nth_batch, -1);
      };
      CPUKernelUtils::ParallelFor(shared_tridiagonalsolve, matrix_num);
    } else {
      for (size_t nth_batch = 0; nth_batch < matrix_num; nth_batch++)
        res_ = ChooseDataType_(inputs, outputs, -1, nth_batch);
    }
  }
  return res_;
}

std::vector<std::pair<KernelAttr, TridiagonalSolveCPUKernelMod::TridiagonalSolveFunc>>
  TridiagonalSolveCPUKernelMod::func_list_ = {
    {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &TridiagonalSolveCPUKernelMod::LaunchKernel<float>},
    {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &TridiagonalSolveCPUKernelMod::LaunchKernel<double>},
    {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
     &TridiagonalSolveCPUKernelMod::LaunchKernel<std::complex<float>>},
    {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
     &TridiagonalSolveCPUKernelMod::LaunchKernel<std::complex<double>>}};

std::vector<KernelAttr> TridiagonalSolveCPUKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, TridiagonalSolveFunc> &pair) { return pair.first; });

  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, TridiagonalSolve, TridiagonalSolveCPUKernelMod);
}  // namespace kernel
}  // namespace mindspore
