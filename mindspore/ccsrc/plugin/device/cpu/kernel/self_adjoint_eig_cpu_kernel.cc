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

#include "plugin/device/cpu/kernel/self_adjoint_eig_cpu_kernel.h"
#include <algorithm>
#include <functional>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "Eigen/Dense"

namespace mindspore::kernel {
constexpr auto kSelfAdjopintEig = "SelfAdjopintEig";
constexpr const size_t kInputsNum = 1;
constexpr const size_t kOutputsNum = 2;

void SelfAdjointEigCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
  attr_ = common::AnfAlgo::GetNodeAttr<bool>(kernel_node, "compute_v");
  input_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
}

bool SelfAdjointEigCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                        const std::vector<kernel::AddressPtr> &,
                                        const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputsNum, kernel_name_);
  // CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputsNum, kernel_name_);
  if (dtype_ == kNumberTypeFloat32) {
    LaunchKernel<float>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat64) {
    LaunchKernel<double>(inputs, outputs);
  } else if (dtype_ == kNumberTypeComplex64) {
    LaunchKernel<std::complex<float>>(inputs, outputs);
  } else if (dtype_ == kNumberTypeComplex128) {
    LaunchKernel<std::complex<double>>(inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dtype of x must be float32 or float64, but got "
                      << TypeIdLabel(dtype_) << ".";
  }
  return true;
}

template <typename T>
bool SelfAdjointEigCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                              const std::vector<kernel::AddressPtr> &outputs) {
  auto *input = reinterpret_cast<T *>(inputs[kIndex0]->addr);
  auto *output0 = reinterpret_cast<T *>(outputs[kIndex0]->addr);
  auto *output1 = reinterpret_cast<T *>(outputs[kIndex1]->addr);
  bool attr0_ = attr_;
  // The size of each dimension
  std::vector<int64_t> shape = input_shape_;
  // rank
  auto input_dims = input_shape_.size();
  // Total number of elements
  size_t input_numelements = static_cast<size_t>(inputs[0]->size / sizeof(T));
  // The length of the line
  const int32_t m = shape[input_dims - 1];
  // The length of the column
  const int32_t n = shape[input_dims - 2];
  auto num_array = (SizeToLong(input_numelements)) / (m * n);
  using MatrixMap = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
  const size_t input_dim_min = 2;
  if (input_dims <= input_dim_min) {
    MatrixMap Input0(input, m, n);
    MatrixMap Output0(output0, m, 1);
    MatrixMap Output1(output1, m, n);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> es(
      Input0, attr0_ ? Eigen::ComputeEigenvectors : Eigen::EigenvaluesOnly);
    Output0 = es.eigenvalues().template cast<T>();
    for (int64_t i = 0; i < m; i++) {
      *(output0 + i) = Output0(i, 0);
    }
    if (attr0_) {
      Output1 = es.eigenvectors();
      for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < m; j++) {
          *(output1 + i * m + j) = Output1(i, j);
        }
      }
    }
  } else {
    for (int64_t batch = 0; batch < num_array; ++batch) {
      T *A = reinterpret_cast<T *>(new T[m * n]);
      T *B = reinterpret_cast<T *>(new T[m]);
      T *C = reinterpret_cast<T *>(new T[m * n]);
      // Get the address of the input and output matrix for each batch
      for (int64_t i = 0; i < m * n; ++i) {
        A[i] = input[batch * m * n + i];
        C[i] = output1[batch * m * n + i];
      }
      for (int64_t i = 0; i < m; ++i) {
        B[i] = output0[batch * m + i];
      }
      MatrixMap Input0(A, m, n);
      MatrixMap Output0(B, m, 1);
      MatrixMap Output1(C, m, n);
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> es(
        Input0, attr0_ ? Eigen::ComputeEigenvectors : Eigen::EigenvaluesOnly);
      Output0 = es.eigenvalues().template cast<T>();
      for (int64_t i = 0; i < m; i++) {
        *(output0 + batch * n + i) = Output0(i, 0);
      }
      if (attr0_) {
        Output1 = es.eigenvectors();
        for (int64_t i = 0; i < m; i++) {
          for (int64_t j = 0; j < m; j++) {
            *(output1 + batch * m * n + i * m + j) = Output1(i, j);
          }
        }
      }
    }
  }
  return true;
}
std::vector<std::pair<KernelAttr, SelfAdjointEigCpuKernelMod::SelfAdjointEigLaunchFunc>>
  SelfAdjointEigCpuKernelMod::func_list_ = {
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &SelfAdjointEigCpuKernelMod::LaunchKernel<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &SelfAdjointEigCpuKernelMod::LaunchKernel<double>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeComplex64)
       .AddOutputAttr(kNumberTypeComplex64)
       .AddOutputAttr(kNumberTypeComplex64),
     &SelfAdjointEigCpuKernelMod::LaunchKernel<std::complex<float>>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeComplex128)
       .AddOutputAttr(kNumberTypeComplex128)
       .AddOutputAttr(kNumberTypeComplex128),
     &SelfAdjointEigCpuKernelMod::LaunchKernel<std::complex<double>>}};

std::vector<KernelAttr> SelfAdjointEigCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, SelfAdjointEigLaunchFunc> &pair) { return pair.first; });

  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, SelfAdjointEig, SelfAdjointEigCpuKernelMod);
}  // namespace mindspore::kernel
