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

#include "plugin/device/cpu/kernel/eigen/qr_cpu_kernel.h"
#include <algorithm>
#include <string>
#include <utility>
#include "Eigen/Dense"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kAMatrixDimNumMin = 2;
constexpr size_t kQRInputsNum = 1;
constexpr size_t kQROutputsNum = 2;
constexpr size_t kPivotsIndex = 1;
constexpr size_t kPermutationIndex = 2;
constexpr size_t kRowIndex = 2;
constexpr size_t kColIndex = 1;
constexpr int64_t kParallelDataNums = 8 * 1024;
}  // namespace
void QrCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
  CHECK_KERNEL_INPUTS_NUM(input_num, kQRInputsNum, kernel_name_);
  size_t output_num = common::AnfAlgo::GetOutputTensorNum(kernel_node);
  CHECK_KERNEL_OUTPUTS_NUM(output_num, kQROutputsNum, kernel_name_);

  full_matrices_ = common::AnfAlgo::GetNodeAttr<bool>(kernel_node, "full_matrices");
  auto x_shape = Convert2SizeTClipNeg(common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0));
  if (x_shape.empty() || x_shape.size() < kAMatrixDimNumMin) {
    MS_LOG_EXCEPTION << "For '" << kernel_name_ << "', input x matrix shape must greater than or equal to 2, but got "
                     << x_shape.size();
  }
  m = x_shape[x_shape.size() - kRowIndex];
  n = x_shape[x_shape.size() - kColIndex];
  auto q_shape = Convert2SizeTClipNeg(common::AnfAlgo::GetOutputInferShape(kernel_node, 0));
  if (q_shape.empty() || q_shape.size() < kAMatrixDimNumMin) {
    MS_LOG_EXCEPTION << "For '" << kernel_name_ << "', output q matrix shape must greater than or equal to 2, but got "
                     << q_shape.size();
  }
  auto r_shape = Convert2SizeTClipNeg(common::AnfAlgo::GetOutputInferShape(kernel_node, 1));
  if (r_shape.empty() || r_shape.size() < kAMatrixDimNumMin) {
    MS_LOG_EXCEPTION << "For '" << kernel_name_ << "', output r matrix shape must greater than or equal to 2, but got "
                     << r_shape.size();
  }
  auto kernel_attr = GetKernelAttrFromNode(kernel_node);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "Qr does not support this kernel data type: " << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;
}

template <typename T>
bool QrCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                  const std::vector<kernel::AddressPtr> &,
                                  const std::vector<kernel::AddressPtr> &outputs) {
  auto input_x = reinterpret_cast<T *>(inputs[0]->addr);
  auto output_q = reinterpret_cast<T *>(outputs[0]->addr);
  auto output_r = reinterpret_cast<T *>(outputs[1]->addr);
  typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MartixXd;
  size_t p = std::min(m, n);
  size_t size_mn = m * n;
  size_t size_mm = m * m;
  size_t size_mp = m * p;
  size_t size_pn = p * n;
  if (size_mn > 0) {
    size_t input_num = static_cast<int64_t>(inputs[0]->size / sizeof(T));
    size_t matrix_num = input_num / size_mn;
    size_t data_size = input_num * sizeof(T);
    if (data_size <= kParallelDataNums) {
      for (size_t i = 0; i < matrix_num; i++) {
        Eigen::Map<MartixXd> martix_x(input_x + i * size_mn, m, n);
        Eigen::HouseholderQR<MartixXd> qr(martix_x);
        if (full_matrices_) {
          Eigen::Map<MartixXd> martix_q(output_q + i * size_mm, m, m);
          Eigen::Map<MartixXd> martix_r(output_r + i * size_mn, m, n);
          martix_q = qr.householderQ();
          martix_r = qr.matrixQR().template triangularView<Eigen::Upper>();
        } else {
          Eigen::Map<MartixXd> martix_q(output_q + i * size_mp, m, p);
          Eigen::Map<MartixXd> martix_r(output_r + i * size_pn, p, n);
          MartixXd tmp = MartixXd::Identity(m, p);
          martix_q = qr.householderQ() * tmp;
          auto qr_top = qr.matrixQR().block(0, 0, p, n);
          martix_r = qr_top.template triangularView<Eigen::Upper>();
        }
      }
    } else {
      auto task = [this, &input_x, &output_q, &output_r, p, size_mm, size_mn, size_mp, size_pn](size_t start,
                                                                                                size_t end) {
        for (size_t i = start; i < end; i++) {
          Eigen::Map<MartixXd> martix_x(input_x + i * size_mn, m, n);
          Eigen::HouseholderQR<MartixXd> qr(martix_x);
          if (full_matrices_) {
            Eigen::Map<MartixXd> martix_q(output_q + i * size_mm, m, m);
            Eigen::Map<MartixXd> martix_r(output_r + i * size_mn, m, n);
            martix_q = qr.householderQ();
            martix_r = qr.matrixQR().template triangularView<Eigen::Upper>();
          } else {
            Eigen::Map<MartixXd> martix_q(output_q + i * size_mp, m, p);
            Eigen::Map<MartixXd> martix_r(output_r + i * size_pn, p, n);
            MartixXd tmp = MartixXd::Identity(m, p);
            martix_q = qr.householderQ() * tmp;
            auto qr_top = qr.matrixQR().block(0, 0, p, n);
            martix_r = qr_top.template triangularView<Eigen::Upper>();
          }
        }
      };
      CPUKernelUtils::ParallelFor(task, matrix_num);
    }
  }
  return true;
}

std::vector<std::pair<KernelAttr, QrCpuKernelMod::QrFunc>> QrCpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
   &QrCpuKernelMod::LaunchKernel<Eigen::half>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
   &QrCpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
   &QrCpuKernelMod::LaunchKernel<double>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeComplex64)
     .AddOutputAttr(kNumberTypeComplex64)
     .AddOutputAttr(kNumberTypeComplex64),
   &QrCpuKernelMod::LaunchKernel<std::complex<float>>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeComplex128)
     .AddOutputAttr(kNumberTypeComplex128)
     .AddOutputAttr(kNumberTypeComplex128),
   &QrCpuKernelMod::LaunchKernel<std::complex<double>>}};

std::vector<KernelAttr> QrCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, QrFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Qr, QrCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
