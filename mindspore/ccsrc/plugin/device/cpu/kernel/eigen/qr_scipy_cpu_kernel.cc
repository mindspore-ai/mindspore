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

#include "plugin/device/cpu/kernel/eigen/qr_scipy_cpu_kernel.h"
#include <algorithm>
#include <string>
#include <utility>
#include "Eigen/Dense"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kAMatrixDimNum = 2;
constexpr size_t kQRInputsNum = 1;
constexpr size_t kQROutputsNum = 2;
constexpr size_t kPivotsIndex = 1;
constexpr size_t kPermutationIndex = 2;
constexpr size_t kRowIndex = 2;
constexpr size_t kColIndex = 1;
}  // namespace
void QRCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
  CHECK_KERNEL_INPUTS_NUM(input_num, kQRInputsNum, kernel_name_);
  size_t output_num = common::AnfAlgo::GetOutputTensorNum(kernel_node);
  CHECK_KERNEL_OUTPUTS_NUM(output_num, kQROutputsNum, kernel_name_);

  const auto mode = common::AnfAlgo::GetNodeAttr<std::string>(kernel_node, MODE);
  if (mode != "full" && mode != "r" && mode != "economic") {
    MS_LOG(EXCEPTION) << "mode must be in [full, r, economic], but got [" << mode << "].";
  }

  auto a_shape = Convert2SizeTClipNeg(common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0));
  CHECK_KERNEL_INPUTS_NUM(a_shape.size(), kAMatrixDimNum, kernel_name_);
  a_row_ = a_shape[kDim0];
  a_col_ = a_shape[kDim1];

  auto q_shape = Convert2SizeTClipNeg(common::AnfAlgo::GetOutputInferShape(kernel_node, 0));
  CHECK_KERNEL_INPUTS_NUM(q_shape.size(), kAMatrixDimNum, kernel_name_);
  q_row_ = q_shape[kDim0];
  q_col_ = q_shape[kDim1];

  auto r_shape = Convert2SizeTClipNeg(common::AnfAlgo::GetOutputInferShape(kernel_node, 1));
  CHECK_KERNEL_INPUTS_NUM(r_shape.size(), kAMatrixDimNum, kernel_name_);
  r_row_ = r_shape[kDim0];
  r_col_ = r_shape[kDim1];

  if (mode == "economic") {
    economic_ = true;
  }

  auto kernel_attr = GetKernelAttrFromNode(kernel_node);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "QR does not support this kernel data type: " << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;
}

template <typename T>
bool QRCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                  const std::vector<kernel::AddressPtr> &,
                                  const std::vector<kernel::AddressPtr> &outputs) {
  T *a_value = reinterpret_cast<T *>(inputs[0]->addr);
  Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> input_a(a_value, a_row_, a_col_);
  T *q_value = reinterpret_cast<T *>(outputs[0]->addr);
  Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> output_q(q_value, q_row_, q_col_);
  T *r_value = reinterpret_cast<T *>(outputs[1]->addr);
  Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> output_r(r_value, r_row_, r_col_);

  auto householder_qr = input_a.householderQr();
  if (economic_) {
    // r_row_ = std::min(a_row_, a_col_)
    output_r = Eigen::MatrixXd::Identity(r_row_, a_row_).template cast<T>() *
               householder_qr.matrixQR().template triangularView<Eigen::Upper>();
    // q_col_ = std::min(a_row_, a_col_)
    output_q = householder_qr.householderQ() * Eigen::MatrixXd::Identity(q_row_, q_col_).template cast<T>();
  } else {
    output_r = householder_qr.matrixQR().template triangularView<Eigen::Upper>();
    output_q = householder_qr.householderQ();
  }
  if (output_r.RowsAtCompileTime != 0 && output_r.ColsAtCompileTime != 0 && output_q.RowsAtCompileTime != 0 &&
      output_q.ColsAtCompileTime != 0) {
    return true;
  }
  MS_LOG_EXCEPTION << kernel_name_ << " output lu shape invalid.";
}

std::vector<std::pair<KernelAttr, QRCpuKernelMod::QRFunc>> QRCpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
   &QRCpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
   &QRCpuKernelMod::LaunchKernel<double>}};

std::vector<KernelAttr> QRCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, QRFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, QR, QRCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
