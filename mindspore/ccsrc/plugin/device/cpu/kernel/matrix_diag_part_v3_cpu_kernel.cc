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

#include "plugin/device/cpu/kernel/matrix_diag_part_v3_cpu_kernel.h"
#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include "ops/op_name.h"
#include "mindspore/core/ops/matrix_diag_part_v3.h"
#include "mindspore/core/utils/check_convert_utils.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kMatrixDiagPartV3InputsNum = 3;
constexpr size_t kMatrixDiagPartV3OutputsNum = 1;
constexpr int64_t kParallelArrayNumSameShape = 2048;  // all cores running if data size is too large
constexpr int64_t ZERO = 0;
static std::pair<int64_t, int64_t> ComputeTwo(int64_t diag_index, int64_t max_diag_len, int64_t num_rows,
                                              int64_t num_cols, bool align_superdiag, bool align_subdiag) {
  bool left_align = (diag_index >= ZERO && align_superdiag) || (diag_index <= ZERO && align_subdiag);
  int64_t diag_len = std::min(num_rows + std::min(ZERO, diag_index), num_cols + std::min(ZERO, -diag_index));
  int64_t offset = (left_align) ? ZERO : (max_diag_len - diag_len);
  return {diag_len, offset};
}
}  // namespace

bool MatrixDiagPartV3CpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                        const std::vector<KernelTensorPtr> &inputs,
                                        const std::vector<KernelTensorPtr> &outputs) {
  MS_ERROR_IF_NULL(base_operator);
  const std::string input_str = "input number";
  const size_t input_number = 3;
  kernel_name_ = base_operator->name();
  (void)CheckAndConvertUtils::CheckInteger(input_str, SizeToLong(inputs.size()), kEqual, input_number, kernel_name_);
  auto op_prim = std::dynamic_pointer_cast<ops::MatrixDiagPartV3>(base_operator);
  MS_ERROR_IF_NULL(op_prim);
  auto align = op_prim->get_align();
  if (!align.empty()) {
    align_ = align;
  }
  (void)CheckAndConvertUtils::CheckString(ops::kAlign, align_, {"LEFT_RIGHT", "RIGHT_LEFT", "LEFT_LEFT", "RIGHT_RIGHT"},
                                          kernel_name_);
  return true;
}

int MatrixDiagPartV3CpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                         const std::vector<KernelTensorPtr> &inputs,
                                         const std::vector<KernelTensorPtr> &outputs,
                                         const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  int ret = KRET_OK;
  if ((ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost)) != 0) {
    return ret;
  }
  auto padding_dtype = inputs[kIndex2]->GetDtype();
  auto output_dtype = outputs[kIndex0]->GetDtype();
  input_dtype_ = inputs[kIndex0]->GetDtype();
  if (input_dtype_ != padding_dtype) {
    MS_LOG(ERROR) << "For MatrixDiagPartV3, the data type of x need be same with padding_value.";
    return KRET_RESIZE_FAILED;
  }
  if (input_dtype_ != output_dtype) {
    MS_LOG(ERROR) << "For MatrixDiagPartV3, the data type of x need be same with output.";
    return KRET_RESIZE_FAILED;
  }
  x_shape_ = inputs[kIndex0]->GetShapeVector();
  k_shape_ = inputs[kIndex1]->GetShapeVector();
  size_t k_dim_size = k_shape_.size();
  const size_t k_dim_size_max = 1;
  if (k_dim_size > k_dim_size_max) {
    MS_LOG(ERROR) << "For MatrixDiagPartV3, k_dim_size can not be greater than 1, received " << k_dim_size << ".";
    return KRET_RESIZE_FAILED;
  }

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel type: " << kernel_attr;
    return KRET_RESIZE_FAILED;
  }
  kernel_func_ = func_list_[index].second;
  return KRET_OK;
}

template <typename T>
bool MatrixDiagPartV3CpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                                const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kMatrixDiagPartV3InputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kMatrixDiagPartV3OutputsNum, kernel_name_);
  // k
  int64_t lower_diag_index = 0;
  upper_diag_index_ = 0;
  size_t k_len = static_cast<size_t>(inputs[1]->size / sizeof(int32_t));
  auto k_Data = static_cast<int32_t *>(inputs[1]->addr);
  MS_EXCEPTION_IF_NULL(k_Data);
  const size_t k_len_max = 2;
  if (k_len == 0 || k_len > k_len_max) {
    MS_LOG(EXCEPTION) << "For MatrixDiagPartV3, k must have one or two elements, but received " << k_len << "elements.";
  }
  lower_diag_index = k_Data[0];
  upper_diag_index_ = k_Data[0];
  if (k_len == k_len_max) {
    upper_diag_index_ = k_Data[1];
  }
  if (!(lower_diag_index <= upper_diag_index_)) {
    MS_LOG(EXCEPTION) << "For MatrixDiagPartV3, k[0] can not be larger than k[1] . ,received " << lower_diag_index
                      << " is larger than " << upper_diag_index_;
  }
  // x
  size_t input_dims = x_shape_.size();
  const size_t input_dim_min = 2;
  if (input_dims < input_dim_min) {
    MS_LOG(EXCEPTION) << "For MatrixDiagPartV3, input x dims must be greater equal than 2 while got " << input_dims
                      << ".";
  }
  num_cols_ = x_shape_[input_dims - 1];
  const size_t toCalRow = 2;
  num_rows_ = x_shape_[input_dims - toCalRow];
  size_t input_numelements = static_cast<size_t>(inputs[0]->size / sizeof(T));
  num_array_ = (SizeToLong(input_numelements)) / (num_rows_ * num_cols_);

  if (align_ == "LEFT_LEFT" || align_ == "LEFT_RIGHT") {
    align_superdiag_ = true;
  } else {
    align_superdiag_ = false;
  }
  if (align_ == "LEFT_LEFT" || align_ == "RIGHT_LEFT") {
    align_subdiag_ = true;
  } else {
    align_subdiag_ = false;
  }
  num_diags_ = upper_diag_index_ - lower_diag_index + 1;
  max_diag_len_ = std::min(num_rows_ + std::min(upper_diag_index_, ZERO), num_cols_ - std::max(lower_diag_index, ZERO));
  output_elements_in_batch_ = num_diags_ * max_diag_len_;
  data_num_ = num_array_ * output_elements_in_batch_;
  return DoLaunch<T>(inputs, outputs);
}

template <typename T>
bool MatrixDiagPartV3CpuKernelMod::DoLaunch(const std::vector<kernel::AddressPtr> &inputs,
                                            const std::vector<kernel::AddressPtr> &outputs) {
  // padding_value
  size_t padding_value_num = static_cast<size_t>(inputs[kIndex2]->size / sizeof(T));
  if (!(padding_value_num == 1)) {
    MS_LOG(EXCEPTION) << "For MatrixDiagPartV3, padding_value must have only one element, received "
                      << padding_value_num << " elements. ";
  }
  auto *padding_value_data = static_cast<T *>(inputs[kIndex2]->addr);
  MS_EXCEPTION_IF_NULL(padding_value_data);
  T padding_value = padding_value_data[0];
  auto output_data = static_cast<T *>(outputs[0]->addr);
  MS_EXCEPTION_IF_NULL(output_data);
  auto input_data = static_cast<T *>(inputs[0]->addr);
  MS_EXCEPTION_IF_NULL(input_data);
  size_t Num_array = LongToSize(num_array_);

  if (data_num_ >= kParallelArrayNumSameShape) {
    auto task = [this, &output_data, &input_data, padding_value](size_t start, size_t end) {
      int64_t out_begin_index = SizeToLong(start) * output_elements_in_batch_;
      for (size_t index_array = start; index_array < end; index_array++) {
        for (int64_t i = 0; i < num_diags_; i++) {
          int64_t offset = 0;
          int64_t diag_len = 0;
          int64_t diag_index = upper_diag_index_ - i;
          int64_t col_offset = std::max(ZERO, -diag_index);
          int64_t row_offset = std::max(ZERO, diag_index);
          std::tie(diag_len, offset) =
            ComputeTwo(diag_index, max_diag_len_, num_rows_, num_cols_, align_superdiag_, align_subdiag_);

          for (int64_t n = 0; n < diag_len; n++) {
            output_data[LongToSize(out_begin_index + offset + n)] =
              input_data[SizeToLong(index_array) * num_rows_ * num_cols_ + (n + col_offset) * num_cols_ + n +
                         row_offset];
          }
          const bool left_align = (offset == 0);
          const int64_t padding_start = (left_align) ? diag_len : 0;
          const int64_t padding_end = (left_align) ? max_diag_len_ : offset;
          int64_t n = padding_start;
          while (n < padding_end) {
            output_data[LongToSize(out_begin_index + n)] = padding_value;
            n += 1;
          }
          out_begin_index += max_diag_len_;
        }
      }
    };
    CPUKernelUtils::ParallelFor(task, Num_array);
  } else {
    // single core used if data size is not too large
    int64_t out_begin_index = 0;
    for (int64_t index_array = 0; index_array < num_array_; index_array++) {
      for (int64_t i = 0; i < num_diags_; i++) {
        int64_t offset = 0;
        int64_t diag_len = 0;
        int64_t diag_index = upper_diag_index_ - i;
        int64_t col_offset = std::max(ZERO, -diag_index);
        int64_t row_offset = std::max(ZERO, diag_index);
        std::tie(diag_len, offset) =
          ComputeTwo(diag_index, max_diag_len_, num_rows_, num_cols_, align_superdiag_, align_subdiag_);

        for (int64_t n = 0; n < diag_len; n++) {
          output_data[LongToSize(out_begin_index + offset + n)] =
            input_data[LongToSize(index_array * num_rows_ * num_cols_ + (n + col_offset) * num_cols_ + n + row_offset)];
        }
        const bool left_align = (offset == 0);
        const int64_t padding_start = (left_align) ? diag_len : 0;
        const int64_t padding_end = (left_align) ? max_diag_len_ : offset;
        int64_t n = padding_start;
        while (n < padding_end) {
          output_data[LongToSize(out_begin_index + n)] = padding_value;
          n += 1;
        }
        out_begin_index += max_diag_len_;
      }
    }
  }
  return true;
}

std::vector<std::pair<KernelAttr, MatrixDiagPartV3CpuKernelMod::MatrixDiagPartV3Func>>
  MatrixDiagPartV3CpuKernelMod::func_list_ = {{KernelAttr()
                                                 .AddInputAttr(kNumberTypeInt8)
                                                 .AddInputAttr(kNumberTypeInt32)
                                                 .AddInputAttr(kNumberTypeInt8)
                                                 .AddOutputAttr(kNumberTypeInt8),
                                               &MatrixDiagPartV3CpuKernelMod::LaunchKernel<int8_t>},
                                              {KernelAttr()
                                                 .AddInputAttr(kNumberTypeInt16)
                                                 .AddInputAttr(kNumberTypeInt32)
                                                 .AddInputAttr(kNumberTypeInt16)
                                                 .AddOutputAttr(kNumberTypeInt16),
                                               &MatrixDiagPartV3CpuKernelMod::LaunchKernel<int16_t>},
                                              {KernelAttr()
                                                 .AddInputAttr(kNumberTypeInt32)
                                                 .AddInputAttr(kNumberTypeInt32)
                                                 .AddInputAttr(kNumberTypeInt32)
                                                 .AddOutputAttr(kNumberTypeInt32),
                                               &MatrixDiagPartV3CpuKernelMod::LaunchKernel<int32_t>},
                                              {KernelAttr()
                                                 .AddInputAttr(kNumberTypeInt64)
                                                 .AddInputAttr(kNumberTypeInt32)
                                                 .AddInputAttr(kNumberTypeInt64)
                                                 .AddOutputAttr(kNumberTypeInt64),
                                               &MatrixDiagPartV3CpuKernelMod::LaunchKernel<int64_t>},
                                              {KernelAttr()
                                                 .AddInputAttr(kNumberTypeUInt8)
                                                 .AddInputAttr(kNumberTypeInt32)
                                                 .AddInputAttr(kNumberTypeUInt8)
                                                 .AddOutputAttr(kNumberTypeUInt8),
                                               &MatrixDiagPartV3CpuKernelMod::LaunchKernel<uint8_t>},
                                              {KernelAttr()
                                                 .AddInputAttr(kNumberTypeUInt16)
                                                 .AddInputAttr(kNumberTypeInt32)
                                                 .AddInputAttr(kNumberTypeUInt16)
                                                 .AddOutputAttr(kNumberTypeUInt16),
                                               &MatrixDiagPartV3CpuKernelMod::LaunchKernel<uint16_t>},
                                              {KernelAttr()
                                                 .AddInputAttr(kNumberTypeUInt32)
                                                 .AddInputAttr(kNumberTypeInt32)
                                                 .AddInputAttr(kNumberTypeUInt32)
                                                 .AddOutputAttr(kNumberTypeUInt32),
                                               &MatrixDiagPartV3CpuKernelMod::LaunchKernel<uint32_t>},
                                              {KernelAttr()
                                                 .AddInputAttr(kNumberTypeUInt64)
                                                 .AddInputAttr(kNumberTypeInt32)
                                                 .AddInputAttr(kNumberTypeUInt64)
                                                 .AddOutputAttr(kNumberTypeUInt64),
                                               &MatrixDiagPartV3CpuKernelMod::LaunchKernel<uint64_t>},
                                              {KernelAttr()
                                                 .AddInputAttr(kNumberTypeFloat16)
                                                 .AddInputAttr(kNumberTypeInt32)
                                                 .AddInputAttr(kNumberTypeFloat16)
                                                 .AddOutputAttr(kNumberTypeFloat16),
                                               &MatrixDiagPartV3CpuKernelMod::LaunchKernel<float16>},
                                              {KernelAttr()
                                                 .AddInputAttr(kNumberTypeFloat32)
                                                 .AddInputAttr(kNumberTypeInt32)
                                                 .AddInputAttr(kNumberTypeFloat32)
                                                 .AddOutputAttr(kNumberTypeFloat32),
                                               &MatrixDiagPartV3CpuKernelMod::LaunchKernel<float>},
                                              {KernelAttr()
                                                 .AddInputAttr(kNumberTypeFloat64)
                                                 .AddInputAttr(kNumberTypeInt32)
                                                 .AddInputAttr(kNumberTypeFloat64)
                                                 .AddOutputAttr(kNumberTypeFloat64),
                                               &MatrixDiagPartV3CpuKernelMod::LaunchKernel<double>}};

std::vector<KernelAttr> MatrixDiagPartV3CpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, MatrixDiagPartV3Func> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, MatrixDiagPartV3, MatrixDiagPartV3CpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
