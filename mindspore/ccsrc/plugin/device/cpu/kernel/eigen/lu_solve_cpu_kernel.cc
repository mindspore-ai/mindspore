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

#include "plugin/device/cpu/kernel/eigen/lu_solve_cpu_kernel.h"
#include <algorithm>
#include <vector>
#include <string>
#include <utility>
#include "utils/ms_utils.h"
#include "plugin/device/cpu/kernel/eigen/eigen_common_utils.h"
#include "Eigen/LU"
namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kLUaIndex = 0;
constexpr size_t kLUbIndex = 1;
constexpr size_t kLuIndex = 0;
constexpr size_t kLUDefaultShape = 1;
constexpr size_t kRowIndex = 2;
constexpr size_t kColIndex = 1;
}  // namespace

int LUSolverCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                 const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  dtype_ = inputs[kIndex0]->dtype_id();
  auto a_shape = Convert2SizeTClipNeg(inputs[kLUaIndex]->GetShapeVector());
  auto b_shape = Convert2SizeTClipNeg(inputs[kLUbIndex]->GetShapeVector());
  if (a_shape.empty() || b_shape.empty()) {
    MS_LOG_EXCEPTION << kernel_name_ << " input a or b matrix shape invalid.";
  }
  if (a_shape.size() == kLUDefaultShape) {
    a_row_ = a_shape.front();
  } else {
    a_row_ = a_shape.at(a_shape.size() - kRowIndex);
    a_col_ = a_shape.at(a_shape.size() - kColIndex);
  }
  if (b_shape.size() == kLUDefaultShape) {
    b_row_ = b_shape.front();
  } else {
    b_row_ = b_shape.at(b_shape.size() - kRowIndex);
    b_col_ = b_shape.at(b_shape.size() - kColIndex);
  }
  const auto &output_lu_shape = outputs[kLuIndex]->GetShapeVector();
  if (output_lu_shape.empty()) {
    MS_LOG_EXCEPTION << kernel_name_ << " output lu shape invalid.";
  }
  if (output_lu_shape.size() == kLUDefaultShape) {
    out_row_ = output_lu_shape.front();
  } else {
    out_row_ = output_lu_shape.at(output_lu_shape.size() - kRowIndex);
    out_col_ = output_lu_shape.at(output_lu_shape.size() - kColIndex);
  }
  trans_ = GetValue<std::string>(primitive_->GetAttr(TRANS));

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "LUSolver does not support this kernel data type: " << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;
  return KRET_OK;
}

template <typename T>
bool LUSolverCpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
                                        const std::vector<KernelTensor *> &outputs) {
  T *a_value = reinterpret_cast<T *>(inputs[kLUaIndex]->device_ptr());
  Map<Matrix<T, RowMajor>> input_a(a_value, a_row_, a_col_);

  T *b_value = reinterpret_cast<T *>(inputs[kLUbIndex]->device_ptr());
  Map<Matrix<T, RowMajor>> input_b(b_value, b_row_, b_col_);
  T *output_lu_value = reinterpret_cast<T *>(outputs[kLuIndex]->device_ptr());
  Map<Matrix<T, RowMajor>> output_lu(output_lu_value, out_row_, out_col_);
  if (trans_ == "N") {
    output_lu.noalias() = input_a.template triangularView<UnitLower>().solve(input_b);
    output_lu.noalias() = input_a.template triangularView<Upper>().solve(output_lu);
  } else if (trans_ == "T") {
    output_lu.noalias() = input_a.template triangularView<Upper>().solve(input_b);
    output_lu.noalias() = input_a.template triangularView<UnitLower>().solve(output_lu);
  } else if (trans_ == "C") {
    MS_LOG_EXCEPTION << kernel_name_ << " trans_ flag is not supported C:  " << trans_;
  } else {
    MS_LOG_EXCEPTION << kernel_name_ << " trans_ flag is invalid:  " << trans_;
  }
  return true;
}

std::vector<std::pair<KernelAttr, LUSolverCpuKernelMod::LUSolverFunc>> LUSolverCpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
   &LUSolverCpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
   &LUSolverCpuKernelMod::LaunchKernel<double>}};

std::vector<KernelAttr> LUSolverCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, LUSolverFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, LUSolver, LUSolverCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
