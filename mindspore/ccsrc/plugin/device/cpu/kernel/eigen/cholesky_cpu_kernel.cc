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

#include "plugin/device/cpu/kernel/eigen/cholesky_cpu_kernel.h"
#include <algorithm>
#include <vector>
#include <utility>
#include "plugin/device/cpu/kernel/eigen/eigen_common_utils.h"
#include "utils/ms_utils.h"
#include "Eigen/Cholesky"
namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kInputsNum = 1;
constexpr size_t kInputIndex = 0;
constexpr size_t kOutputsNum = 1;
constexpr size_t kOutputIndex = 0;
constexpr size_t kRowIndex = 2;
constexpr size_t kColIndex = 1;
}  // namespace

void CholeskyCpuKernelMod::InitMatrixInfo(const std::vector<size_t> &shape, size_t *row, size_t *col) {
  if (shape.empty()) {
    MS_LOG_EXCEPTION << kernel_name_ << " input or output shape is empty which is invalid.";
  }
  constexpr size_t min_dim = 1;
  if (shape.size() <= min_dim) {
    MS_LOG_EXCEPTION << kernel_name_ << " input or output shape dim is " << shape.size() << " which is invalid.";
  }
  *row = shape.at(shape.size() - kRowIndex);
  *col = shape.at(shape.size() - kColIndex);
  if (*row != *col) {
    MS_LOG_EXCEPTION << kernel_name_ << " input shape is invalid. "
                     << "Cholesky expects a square matrix. but input or output shape is: " << *row << ", " << *col;
  }
  outer_batch_ = min_dim;
  for (const auto &sh : shape) {
    outer_batch_ *= sh;
  }
  outer_batch_ /= ((*row) * (*col));
}

bool CholeskyCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->GetPrim()->name();
  dtype_ = inputs[kIndex0]->GetDtype();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputsNum, kernel_name_);
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  if (base_operator->HasAttr("upper")) {
    flag_ = false;
    upper_ = GetValue<bool>(base_operator->GetAttr("upper"));
  }
  // If clean attribute exits, we will remain rand triangular data by clean flag, otherwise clean it to zero.
  if (base_operator->HasAttr(CLEAN)) {
    clean_ = GetValue<bool>(base_operator->GetAttr(CLEAN));
  }
  if (base_operator->HasAttr(LOWER)) {
    lower_ = GetValue<bool>(base_operator->GetAttr(LOWER));
  }
  return true;
}

int CholeskyCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs,
                                 const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }

  auto input_shape = LongVecToSizeVec(inputs[kInputIndex]->GetShapeVector());
  InitMatrixInfo(input_shape, &input_row_, &input_col_);
  auto output_shape = LongVecToSizeVec(inputs[kOutputIndex]->GetShapeVector());
  InitMatrixInfo(output_shape, &output_row_, &output_col_);
  return KRET_OK;
}

template <typename T>
bool CholeskyCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                        const std::vector<AddressPtr> &outputs) {
  T *batch_input_value = reinterpret_cast<T *>(inputs[kInputIndex]->addr);
  T *batch_output_value = reinterpret_cast<T *>(outputs[kOutputIndex]->addr);
  Eigen::LLT<Matrix<T, RowMajor>> llt;
  for (size_t batch = 0; batch < outer_batch_; ++batch) {
    T *input_value = batch_input_value + batch * input_row_ * input_col_;
    T *output_value = batch_output_value + batch * output_row_ * output_col_;
    Map<Matrix<T, RowMajor>> input(input_value, input_row_, input_col_);
    Map<Matrix<T, RowMajor>> output(output_value, output_row_, output_col_);
    (void)llt.compute(input);
    if (flag_) {
      if (clean_) {
        if (lower_) {
          output = llt.matrixL();
        } else {
          output = llt.matrixU();
        }
      } else {
        if (lower_) {
          output = llt.matrixLLT();
        } else {
          output = llt.matrixLLT().transpose();
        }
      }
    } else {
      if (!input.isApprox(input.transpose()) || llt.info() == Eigen::NumericalIssue) {
        MS_LOG_EXCEPTION << "Cholesky expects symmetric positive definite matrices as inputs.";
      }
      if (!upper_) {
        output = llt.matrixL();
      } else {
        output = llt.matrixU();
      }
    }
  }
  return true;
}

std::vector<std::pair<KernelAttr, CholeskyCpuKernelMod::CholeskyFunc>> CholeskyCpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
   &CholeskyCpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
   &CholeskyCpuKernelMod::LaunchKernel<double>}};

std::vector<KernelAttr> CholeskyCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, CholeskyFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Cholesky, CholeskyCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
