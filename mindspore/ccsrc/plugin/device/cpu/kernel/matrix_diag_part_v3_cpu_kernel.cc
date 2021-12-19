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
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kMatrixDiagPartV3InputsNum = 3;
constexpr size_t kMatrixDiagPartV3OutputsNum = 1;
constexpr int64_t kParallelArrayNumSameShape = 2048;  // all cores running if data size is too large
constexpr size_t kIndexPaddingValue = 2;
constexpr int64_t ZERO = 0;
static std::pair<int64_t, int64_t> ComputeTwo(int64_t diag_index, int64_t max_diag_len, int64_t num_rows,
                                              int64_t num_cols, bool align_superdiag, bool align_subdiag) {
  bool left_align = (diag_index >= ZERO && align_superdiag) || (diag_index <= ZERO && align_subdiag);
  int64_t diag_len = std::min(num_rows + std::min(ZERO, diag_index), num_cols + std::min(ZERO, -diag_index));
  int64_t offset = (left_align) ? ZERO : (max_diag_len - diag_len);
  return {diag_len, offset};
}
}  // namespace

void MatrixDiagPartV3CpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);

  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);

  if (common::AnfAlgo::HasNodeAttr("align", kernel_node)) {
    align_ = common::AnfAlgo::GetNodeAttr<std::string>(kernel_node, "align");
    if (!(align_ == "" || align_ == "RIGHT_LEFT" || align_ == "RIGHT_RIGHT" || align_ == "LEFT_LEFT" ||
          align_ == "LEFT_RIGHT")) {
      MS_LOG(EXCEPTION) << "Attr 'align' of 'MatrixDiagPartV3' is not in: 'LEFT_RIGHT', "
                           "'RIGHT_LEFT', 'LEFT_LEFT', 'RIGHT_RIGHT'.";
    }
    if (align_ == "") align_ = "RIGHT_LEFT";
  } else {
    align_ = "RIGHT_LEFT";
  }

  auto padding_data_type = AnfAlgo::GetInputDeviceDataType(kernel_node, kIndexPaddingValue);
  input_dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
  auto output_data_type = AnfAlgo::GetOutputDeviceDataType(kernel_node, 0);

  if (padding_data_type != input_dtype_) {
    MS_LOG(EXCEPTION) << "For MatrixDiagPartV3, the data type of x need be same with padding_value.";
  }

  if (input_dtype_ != output_data_type) {
    MS_LOG(EXCEPTION) << "For MatrixDiagPartV3, the data type of x need be same with output.";
  }

  x_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  k_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, 1);
  size_t k_dim_size = k_shape_.size();
  const size_t k_dim_size_max = 1;
  if (k_dim_size > k_dim_size_max) {
    MS_LOG(EXCEPTION) << "For MatrixDiagPartV3, k_dim_size must not be greater than 1, received " << k_dim_size << ".";
  }

  auto kernel_attr = GetKernelAttrFromNode(kernel_node);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "MatrixDiagPartV3 does not support this kernel data type: " << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;
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
  auto k_Data = reinterpret_cast<int32_t *>(inputs[1]->addr);
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
    MS_LOG(EXCEPTION) << "For MatrixDiagPartV3, k[0] must not be larger than k[1] . ,received " << lower_diag_index
                      << " is larger than " << upper_diag_index_;
  }
  // x
  size_t input_dims = x_shape_.size();
  const size_t input_dim_min = 2;
  if (input_dims < input_dim_min) {
    MS_LOG(EXCEPTION) << "For MatrixDiagPartV3, input x dims must be greater equal than 2 while got " << input_dims
                      << ".";
  }
  num_cols_ = SizeToLong(x_shape_[input_dims - 1]);
  const size_t toCalRow = 2;
  num_rows_ = SizeToLong(x_shape_[input_dims - toCalRow]);
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
  size_t padding_value_num = static_cast<size_t>(inputs[kIndexPaddingValue]->size / sizeof(T));
  if (!(padding_value_num == 1)) {
    MS_LOG(EXCEPTION) << "For MatrixDiagPartV3, padding_value must have only one element, received "
                      << padding_value_num << " elements. ";
  }
  auto *padding_value_data = reinterpret_cast<T *>(inputs[kIndexPaddingValue]->addr);
  MS_EXCEPTION_IF_NULL(padding_value_data);
  T padding_value = padding_value_data[0];
  auto output_data = reinterpret_cast<T *>(outputs[0]->addr);
  MS_EXCEPTION_IF_NULL(output_data);
  auto input_data = reinterpret_cast<T *>(inputs[0]->addr);
  MS_EXCEPTION_IF_NULL(input_data);
  size_t Num_array = LongToSize(num_array_);

  if (data_num_ >= kParallelArrayNumSameShape) {
    auto task = [this, &output_data, &input_data, padding_value](size_t start, size_t end) {
      int64_t out_begin_index = SizeToLong(start * output_elements_in_batch_);
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
            output_data[LongToSize(out_begin_index + offset + n)] = input_data[LongToSize(
              index_array * num_rows_ * num_cols_ + (n + col_offset) * num_cols_ + n + row_offset)];
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
  std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                 [](const std::pair<KernelAttr, MatrixDiagPartV3Func> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, MatrixDiagPartV3, MatrixDiagPartV3CpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
