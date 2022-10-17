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

#include "plugin/device/cpu/kernel/matrix_diag_v3_cpu_kernel.h"
#include <set>
#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include "mindspore/core/ops/op_name.h"
#include "mindspore/core/ops/matrix_diag_v3.h"
#include "mindspore/core/utils/check_convert_utils.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kMatrixDiagV3InputsNum = 5;
constexpr size_t kMatrixDiagV3OutputsNum = 1;
constexpr size_t kIndexNumRow = 2;
constexpr size_t kIndexNumCol = 3;
constexpr size_t kIndexPaddingValue = 4;
static std::pair<int64_t, int64_t> ComputeTwo(int64_t diag_index, int64_t max_diag_len, int32_t num_rows,
                                              int32_t num_cols, bool align_superdiag, bool align_subdiag) {
  const int64_t zero = 0;
  bool left_align = (diag_index >= zero && align_superdiag) || (diag_index <= zero && align_subdiag);
  int64_t diag_len = std::min(num_rows + std::min(zero, diag_index), num_cols + std::min(zero, -diag_index));
  int64_t offset = (left_align) ? zero : (max_diag_len - diag_len);
  return {diag_len, offset};
}
}  // namespace

bool MatrixDiagV3CpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                    const std::vector<KernelTensorPtr> &outputs) {
  MS_ERROR_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  MS_LOG(WARNING) << "in new init " << kernel_name_;
  auto op_prim = std::dynamic_pointer_cast<ops::MatrixDiagV3>(base_operator);
  MS_ERROR_IF_NULL(op_prim);
  auto align = op_prim->get_align();
  if (!align.empty()) {
    align_ = align;
  }
  const std::set<std::string> support_list = {"LEFT_RIGHT", "RIGHT_LEFT", "LEFT_LEFT", "RIGHT_RIGHT"};
  (void)CheckAndConvertUtils::CheckString(ops::kAlign, align_, support_list, kernel_name_);
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "MatrixDiagV3 does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int MatrixDiagV3CpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                     const std::vector<KernelTensorPtr> &outputs,
                                     const std::map<uint32_t, tensor::TensorPtr> &) {
  MS_ERROR_IF_NULL(base_operator);
  int ret = KernelMod::Resize(base_operator, inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }
  MS_LOG(WARNING) << "in new resize ..";
  diagonal_data_type_ = inputs[kIndex0]->GetDtype();
  auto padding_type = inputs[kIndexPaddingValue]->GetDtype();
  auto output_data_type = outputs[kIndex0]->GetDtype();

  if (diagonal_data_type_ != padding_type) {
    MS_LOG(ERROR) << "For MatrixDiagV3, the data type of x need be same with padding_value.";
    return KRET_RESIZE_FAILED;
  }

  if (diagonal_data_type_ != output_data_type) {
    MS_LOG(ERROR) << "For MatrixDiagV3, The data type of x need be same with output.";
    return KRET_RESIZE_FAILED;
  }
  diagonal_shape_ = inputs[kIndex0]->GetShapeVector();
  k_shape_ = inputs[kIndex1]->GetShapeVector();
  const size_t dim_size_max = 1;
  if (k_shape_.size() > dim_size_max) {
    MS_LOG(ERROR) << "For MatrixDiagV3, k_dim_size can not be greater than 1, received " << k_shape_.size() << ".";
    return KRET_RESIZE_FAILED;
  }
  return KRET_OK;
}

template <typename T>
bool MatrixDiagV3CpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                            const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kMatrixDiagV3InputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kMatrixDiagV3OutputsNum, kernel_name_);
  lower_diag_index_ = 0;
  upper_diag_index_ = 0;
  num_rows_ = -1;
  num_cols_ = -1;
  const size_t diag_rank = diagonal_shape_.size();
  if (diag_rank < 1) {
    MS_LOG(EXCEPTION) << "For MatrixDiagV3, input x dims must be greater equal than 1 while got " << diag_rank << ".";
  }
  max_diag_len_ = diagonal_shape_[diag_rank - 1];
  // k
  auto *k_data = static_cast<int32_t *>(inputs[1]->addr);
  MS_EXCEPTION_IF_NULL(k_data);
  lower_diag_index_ = k_data[0];
  upper_diag_index_ = lower_diag_index_;
  size_t k_num = static_cast<size_t>(inputs[1]->size / sizeof(int32_t));
  const size_t k_num_max = 2;
  if (k_num == 0 || k_num > k_num_max) {
    MS_LOG(EXCEPTION) << "For MatrixDiagV3, k must have one or two elements, but received " << k_num << "elements.";
  }
  if (k_num == k_num_max) {
    upper_diag_index_ = k_data[1];
  }
  if (!(lower_diag_index_ <= upper_diag_index_)) {
    MS_LOG(EXCEPTION) << "For MatrixDiagV3, lower_diag_index must be smaller than upper_diag_index,received "
                      << lower_diag_index_ << " is larger than " << upper_diag_index_;
  }
  const int64_t num_diags = IntToLong(upper_diag_index_) - IntToLong(lower_diag_index_) + 1;
  // num_rows
  size_t num_rows_num = static_cast<size_t>(inputs[kIndexNumRow]->size / sizeof(int32_t));
  if (!(num_rows_num == 1)) {
    MS_LOG(EXCEPTION) << "For MatrixDiagV3, num_rows must have only one element, received " << num_rows_num
                      << " elements. ";
  }
  auto *num_rows_data = static_cast<int32_t *>(inputs[kIndexNumRow]->addr);
  MS_EXCEPTION_IF_NULL(num_rows_data);
  num_rows_ = num_rows_data[0];
  // num_cols
  size_t num_cols_num = static_cast<size_t>(inputs[kIndexNumCol]->size / sizeof(int32_t));
  if (!(num_cols_num == 1)) {
    MS_LOG(EXCEPTION) << "For MatrixDiagV3, num_cols must have only one element, received " << num_cols_num
                      << " elements. ";
  }
  auto *num_cols_data = static_cast<int32_t *>(inputs[kIndexNumCol]->addr);
  MS_EXCEPTION_IF_NULL(num_cols_data);
  num_cols_ = num_cols_data[0];

  const int32_t min_rows = max_diag_len_ + std::max(-upper_diag_index_, 0);
  const int32_t min_cols = max_diag_len_ + std::max(lower_diag_index_, 0);
  if (num_rows_ != -1 && num_rows_ < min_rows) {
    MS_LOG(EXCEPTION) << "For MatrixDiagV3, the number of rows is too small.";
  }
  if (num_cols_ != -1 && num_cols_ < min_cols) {
    MS_LOG(EXCEPTION) << "For MatrixDiagV3, the number of columns is too small.";
  }
  if (num_rows_ == -1 && num_cols_ == -1) {
    num_rows_ = std::max(min_rows, min_cols);
    num_cols_ = num_rows_;
  }
  if (num_rows_ == -1) {
    num_rows_ = min_rows;
  }
  if (num_cols_ == -1) {
    num_cols_ = min_cols;
  }
  if (num_rows_ != min_rows && num_cols_ != min_cols) {
    MS_LOG(EXCEPTION) << "For MatrixDiagV3, the number of rows or columns is not consistent with "
                         "the specified d_lower, d_upper, and diagonal.";
  }
  diag_elements_in_batch_ = num_diags * max_diag_len_;
  diag_batch_base_index_ = 0 * diag_elements_in_batch_;
  size_t num_element = static_cast<size_t>(outputs[0]->size / sizeof(T));
  num_batches_ = (SizeToLong(num_element)) / (num_rows_ * num_cols_);

  return DoLaunch<T>(inputs, outputs);
}

template <typename T>
bool MatrixDiagV3CpuKernelMod::DoLaunch(const std::vector<kernel::AddressPtr> &inputs,
                                        const std::vector<kernel::AddressPtr> &outputs) {
  align_superdiag_ = align_ == "LEFT_LEFT" || align_ == "LEFT_RIGHT";
  align_subdiag_ = align_ == "LEFT_LEFT" || align_ == "RIGHT_LEFT";
  // padding_value
  size_t padding_value_num = static_cast<size_t>(inputs[kIndexPaddingValue]->size / sizeof(T));
  if (!(padding_value_num == 1)) {
    MS_LOG(EXCEPTION) << "For MatrixDiagV3, padding_value must have only one element, received " << padding_value_num
                      << " elements. ";
  }
  auto *padding_value_data = static_cast<T *>(inputs[kIndexPaddingValue]->addr);
  MS_EXCEPTION_IF_NULL(padding_value_data);
  T padding_value = padding_value_data[0];

  auto *diagonal_data = static_cast<T *>(inputs[0]->addr);
  MS_EXCEPTION_IF_NULL(diagonal_data);
  auto *output_data = static_cast<T *>(outputs[0]->addr);
  MS_EXCEPTION_IF_NULL(output_data);
  int64_t elem = 0;
  for (int64_t index_array = 0; index_array < num_batches_; index_array++) {
    for (int64_t i = 0; i < num_rows_; i++) {
      for (int64_t j = 0; j < num_cols_; j++) {
        int64_t diag_index = j - i;
        int64_t diag_index_in_input = upper_diag_index_ - diag_index;
        int64_t diag_len, offset;
        std::tie(diag_len, offset) =
          ComputeTwo(diag_index, max_diag_len_, num_rows_, num_cols_, align_superdiag_, align_subdiag_);
        int64_t index_in_the_diagonal = j - std::max<int64_t>(diag_index, 0) + offset;
        if (lower_diag_index_ <= diag_index && diag_index <= upper_diag_index_) {
          size_t index =
            LongToSize(diag_batch_base_index_ + diag_index_in_input * max_diag_len_ + index_in_the_diagonal);
          output_data[LongToSize(elem)] = diagonal_data[index];
          elem++;
        } else {
          output_data[LongToSize(elem)] = padding_value;
          elem++;
        }
      }
    }
    diag_batch_base_index_ += diag_elements_in_batch_;
  }
  return true;
}

std::vector<std::pair<KernelAttr, MatrixDiagV3CpuKernelMod::MatrixDiagV3Func>> MatrixDiagV3CpuKernelMod::func_list_ = {
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt8)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt8)
     .AddOutputAttr(kNumberTypeInt8),
   &MatrixDiagV3CpuKernelMod::LaunchKernel<int8_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt16)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt16)
     .AddOutputAttr(kNumberTypeInt16),
   &MatrixDiagV3CpuKernelMod::LaunchKernel<int16_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt32),
   &MatrixDiagV3CpuKernelMod::LaunchKernel<int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt64),
   &MatrixDiagV3CpuKernelMod::LaunchKernel<int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt8)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeUInt8)
     .AddOutputAttr(kNumberTypeUInt8),
   &MatrixDiagV3CpuKernelMod::LaunchKernel<uint8_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt16)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeUInt16)
     .AddOutputAttr(kNumberTypeUInt16),
   &MatrixDiagV3CpuKernelMod::LaunchKernel<uint16_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeUInt32)
     .AddOutputAttr(kNumberTypeUInt32),
   &MatrixDiagV3CpuKernelMod::LaunchKernel<uint32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt64)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeUInt64)
     .AddOutputAttr(kNumberTypeUInt64),
   &MatrixDiagV3CpuKernelMod::LaunchKernel<uint64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16),
   &MatrixDiagV3CpuKernelMod::LaunchKernel<float16>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   &MatrixDiagV3CpuKernelMod::LaunchKernel<float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64),
   &MatrixDiagV3CpuKernelMod::LaunchKernel<double>}};

std::vector<KernelAttr> MatrixDiagV3CpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, MatrixDiagV3Func> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, MatrixDiagV3, MatrixDiagV3CpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
