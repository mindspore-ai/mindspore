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

#include "plugin/device/cpu/kernel/eigen/orgqr_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "Eigen/Dense"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kTwo = 2;
constexpr size_t kInputsNum = 2;
constexpr size_t kOutputsNum = 1;
constexpr size_t kInputIndex0 = 0;
constexpr size_t kInputIndex1 = 1;
constexpr size_t kOutputIndex0 = 0;
}  // namespace

bool OrgqrCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                             const std::vector<KernelTensorPtr> &outputs) {
  MS_ERROR_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputsNum, kernel_name_);

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << kernel_name_ << " does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int OrgqrCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                              const std::vector<KernelTensorPtr> &outputs,
                              const std::map<uint32_t, tensor::TensorPtr> &) {
  auto ret = KernelMod::Resize(base_operator, inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }
  std::vector<int64_t> x_shape = inputs[kIndex0]->GetShapeVector();
  std::vector<int64_t> tau_shape = inputs[kIndex1]->GetShapeVector();
  int64_t shape_size = static_cast<int64_t>(x_shape.size());
  m_ = *(x_shape.end() - kTwo);
  n_ = *(x_shape.end() - 1);
  p_ = *(tau_shape.end() - 1);
  int64_t size_mn = m_ * n_;
  int64_t num_elements = 1;
  for (int64_t i = 0; i < shape_size; i++) {
    num_elements *= x_shape[static_cast<size_t>(i)];
  }
  martrix_num_ = num_elements / size_mn;
  return KRET_OK;
}

template <typename T>
bool OrgqrCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                     const std::vector<kernel::AddressPtr> &outputs) {
  T *x = reinterpret_cast<T *>(inputs[kInputIndex0]->addr);
  T *tau = reinterpret_cast<T *>(inputs[kInputIndex1]->addr);
  T *y = reinterpret_cast<T *>(outputs[kOutputIndex0]->addr);
  typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MartrixXd;
  typedef Eigen::Matrix<T, Eigen::Dynamic, 1> VectorXd;
  for (int64_t i = 0; i < martrix_num_; i++) {
    Eigen::Map<MartrixXd> martrix_y(y + i * m_ * n_, m_, n_);
    Eigen::Map<MartrixXd> martrix_x(x + i * m_ * n_, m_, n_);
    MartrixXd tmp = MartrixXd::Identity(m_, m_);
    Eigen::Map<VectorXd> vector_tau(tau + i * p_, p_, 1);
    for (int64_t k = 0; k < p_; k++) {
      VectorXd vector_v = martrix_x.block(k, k, m_ - k, 1);
      vector_v[0] = 1;
      tmp.rightCols(m_ - k) =
        tmp.rightCols(m_ - k) - vector_tau(k) * (tmp.rightCols(m_ - k) * vector_v) * vector_v.transpose();
    }
    martrix_y = tmp.leftCols(n_);
  }
  return true;
}

template <typename T>
bool OrgqrCpuKernelMod::LaunchComplexKernel(const std::vector<kernel::AddressPtr> &inputs,
                                            const std::vector<kernel::AddressPtr> &outputs) {
  T *x = reinterpret_cast<T *>(inputs[kInputIndex0]->addr);
  T *tau = reinterpret_cast<T *>(inputs[kInputIndex1]->addr);
  T *y = reinterpret_cast<T *>(outputs[kOutputIndex0]->addr);
  typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MartrixXd;
  typedef Eigen::Matrix<T, Eigen::Dynamic, 1> VectorXd;
  for (int64_t i = 0; i < martrix_num_; i++) {
    Eigen::Map<MartrixXd> martrix_y(y + i * m_ * n_, m_, n_);
    Eigen::Map<MartrixXd> martrix_x(x + i * m_ * n_, m_, n_);
    MartrixXd tmp = MartrixXd::Identity(m_, m_);
    Eigen::Map<VectorXd> vector_tau(tau + i * p_, p_, 1);
    for (int64_t k = 0; k < p_; k++) {
      VectorXd vector_v = martrix_x.block(k, k, m_ - k, 1);
      vector_v[0] = 1;
      tmp.rightCols(m_ - k) =
        tmp.rightCols(m_ - k) - vector_tau(k) * (tmp.rightCols(m_ - k) * vector_v) * vector_v.adjoint();
    }
    martrix_y = tmp.leftCols(n_);
  }
  return true;
}

std::vector<std::pair<KernelAttr, OrgqrCpuKernelMod::OrgqrLaunchFunc>> OrgqrCpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
   &OrgqrCpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
   &OrgqrCpuKernelMod::LaunchKernel<double>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeComplex64)
     .AddInputAttr(kNumberTypeComplex64)
     .AddOutputAttr(kNumberTypeComplex64),
   &OrgqrCpuKernelMod::LaunchComplexKernel<complex64>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeComplex128)
     .AddInputAttr(kNumberTypeComplex128)
     .AddOutputAttr(kNumberTypeComplex128),
   &OrgqrCpuKernelMod::LaunchComplexKernel<complex128>}};

std::vector<KernelAttr> OrgqrCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, OrgqrLaunchFunc> &pair) { return pair.first; });

  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Orgqr, OrgqrCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
