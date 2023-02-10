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

#include "plugin/device/cpu/kernel/matrix_set_diag_v3_cpu_kernel.h"
#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kMatrixSetDiagV3InputsNum = 3;
constexpr size_t kMatrixSetDiagV3OutputsNum = 1;
constexpr size_t kParallelDataNum = 64 * 1024;
constexpr size_t kKLengthMax = 2;
constexpr size_t kIndexK = 2;
constexpr int64_t ZERO = 0;
}  // namespace

void MatrixSetDiagV3CpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);

  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);

  if (common::AnfAlgo::HasNodeAttr("align", kernel_node)) {
    align_ = common::AnfAlgo::GetNodeAttr<std::string>(kernel_node, "align");
    if (!(align_ == "" || align_ == "RIGHT_LEFT" || align_ == "RIGHT_RIGHT" || align_ == "LEFT_LEFT" ||
          align_ == "LEFT_RIGHT")) {
      MS_LOG(EXCEPTION) << "Attr 'align' of 'MatrixSetDiagV3' is not in: 'LEFT_RIGHT', "
                           "'RIGHT_LEFT', 'LEFT_LEFT', 'RIGHT_RIGHT'.";
    }
    if (align_ == "") {
      align_ = "RIGHT_LEFT";
    }
  } else {
    align_ = "RIGHT_LEFT";
  }

  auto diagonal_data_type = AnfAlgo::GetInputDeviceDataType(kernel_node, 1);
  input_dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
  auto output_data_type = AnfAlgo::GetOutputDeviceDataType(kernel_node, 0);

  if (diagonal_data_type != input_dtype_) {
    MS_LOG(EXCEPTION) << "For MatrixSetDiagV3, the data type of x need be same diagonal.";
  }

  if (input_dtype_ != output_data_type) {
    MS_LOG(EXCEPTION) << "For MatrixSetDiagV3, the data type of x need be same with output.";
  }

  x_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  diagonal_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, 1);
  k_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, kIndexK);
  size_t k_dim_size = k_shape_.size();
  const size_t k_dim_size_max = 1;
  if (k_dim_size > k_dim_size_max) {
    MS_LOG(EXCEPTION) << "For MatrixSetDiagV3, k_dim_size can not be greater than 1, received " << k_dim_size << ".";
  }

  auto kernel_attr = GetKernelAttrFromNode(kernel_node);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "MatrixSetDiagV3 does not support this kernel data type: " << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;
}

template <typename T>
bool MatrixSetDiagV3CpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                               const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kMatrixSetDiagV3InputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kMatrixSetDiagV3OutputsNum, kernel_name_);
  size_t input_dims = x_shape_.size();
  const size_t input_dim_min = 2;
  const size_t toCalRow = 2;
  if (input_dims < input_dim_min) {
    MS_LOG(EXCEPTION) << "For MatrixSetDiagV3, input x dims must be greater equal than 2 while got " << input_dims
                      << ".";
  }
  input_columns_ = static_cast<size_t>(x_shape_[input_dims - 1]);
  input_rows_ = static_cast<size_t>(x_shape_[input_dims - toCalRow]);
  input_numelements_ = static_cast<size_t>(inputs[0]->size / sizeof(T));

  size_t diagonal_dims = diagonal_shape_.size();
  diagonal_columns_ = static_cast<size_t>(diagonal_shape_[diagonal_dims - 1]);
  diagonal_rows_ = 1;
  if (diagonal_dims > 1) {
    diagonal_rows_ = static_cast<size_t>(diagonal_shape_[diagonal_dims - toCalRow]);
  }

  k_len_ = static_cast<size_t>(inputs[kIndexK]->size / sizeof(int32_t));
  k_lower_ = 0;
  k_upper_ = 0;
  auto k_Data = static_cast<int32_t *>(inputs[kIndexK]->addr);
  MS_EXCEPTION_IF_NULL(k_Data);
  if (k_len_ == 0 || k_len_ > kKLengthMax) {
    MS_LOG(EXCEPTION) << "For MatrixSetDiagV3, k must have only one or two elements, received " << k_len_
                      << "elements.";
  }
  k_lower_ = k_Data[0];
  k_upper_ = k_Data[0];
  if (k_len_ == kKLengthMax) {
    k_upper_ = k_Data[1];
  }
  if (!(k_lower_ <= k_upper_)) {
    MS_LOG(EXCEPTION) << "For MatrixSetDiagV3, k[0] can not be larger than k[1] ,received " << k_lower_
                      << " is larger than " << k_upper_;
  }
  max_diag_len_ = std::min(static_cast<int64_t>(input_rows_) + std::min(k_upper_, ZERO),
                           static_cast<int64_t>(input_columns_) + std::min(-k_lower_, ZERO));

  return DoLaunch<T>(inputs, outputs);
}

template <typename T>
void MatrixSetDiagV3CpuKernelMod::singleCal(const std::vector<kernel::AddressPtr> &inputs,
                                            const std::vector<kernel::AddressPtr> &outputs) {
  auto output_data = static_cast<T *>(outputs[0]->addr);
  MS_EXCEPTION_IF_NULL(output_data);
  auto diagonal_data = static_cast<T *>(inputs[1]->addr);
  MS_EXCEPTION_IF_NULL(diagonal_data);
  auto input_data = static_cast<T *>(inputs[0]->addr);
  MS_EXCEPTION_IF_NULL(input_data);
  if (k_len_ == 1 || (k_len_ == kKLengthMax && k_lower_ == k_upper_)) {
    for (size_t elem = 0; elem < input_numelements_; ++elem) {
      int64_t t = SizeToLong(elem % (input_rows_ * input_columns_));
      int64_t index = SizeToLong(elem / (input_rows_ * input_columns_));
      int64_t m = t / static_cast<int64_t>(input_columns_);
      int64_t n = t % static_cast<int64_t>(input_columns_);
      int64_t x = n - std::max(k_upper_, ZERO);
      if (n - m == k_upper_) {
        output_data[elem] = diagonal_data[LongToSize(index * static_cast<int64_t>(diagonal_columns_) + x)];
      } else {
        output_data[elem] = input_data[elem];
      }
    }
  } else {
    for (size_t elem = 0; elem < input_numelements_; ++elem) {
      int64_t t = SizeToLong(elem % (input_rows_ * input_columns_));
      int64_t index = SizeToLong(elem / (input_rows_ * input_columns_));
      int64_t m = t / static_cast<int64_t>(input_columns_);
      int64_t n = t % static_cast<int64_t>(input_columns_);
      int64_t d = n - m;
      if (d >= k_lower_ && d <= k_upper_) {
        int64_t x = k_upper_ - d;
        int64_t offset = 0;
        if (((align_ == "RIGHT_LEFT" || align_ == "RIGHT_RIGHT") && d >= 0) ||
            ((align_ == "LEFT_RIGHT" || align_ == "RIGHT_RIGHT") && d <= 0)) {
          offset = max_diag_len_ - std::min(static_cast<int64_t>(input_columns_) - std::max(d, ZERO),
                                            static_cast<int64_t>(input_rows_) + std::min(d, ZERO));
        }
        int64_t y = n - std::max(d, ZERO) + offset;
        size_t position =
          LongToSize(index * static_cast<int64_t>(diagonal_rows_) * static_cast<int64_t>(diagonal_columns_) +
                     x * static_cast<int64_t>(diagonal_columns_) + y);
        output_data[elem] = diagonal_data[position];
      } else {
        output_data[elem] = input_data[elem];
      }
    }
  }
}

template <typename T>
bool MatrixSetDiagV3CpuKernelMod::DoLaunch(const std::vector<kernel::AddressPtr> &inputs,
                                           const std::vector<kernel::AddressPtr> &outputs) {
  auto output_data = static_cast<T *>(outputs[0]->addr);
  MS_EXCEPTION_IF_NULL(output_data);
  auto diagonal_data = static_cast<T *>(inputs[1]->addr);
  MS_EXCEPTION_IF_NULL(diagonal_data);
  auto input_data = static_cast<T *>(inputs[0]->addr);
  MS_EXCEPTION_IF_NULL(input_data);

  // 64K boundary value to determine whether to use all cores
  size_t input_size = inputs[0]->size;
  if (input_size < kParallelDataNum) {
    singleCal<T>(inputs, outputs);
  } else {
    auto task = [this, &diagonal_data, &output_data, &input_data](size_t start, size_t end) {
      if (k_len_ == 1 || (k_len_ == kKLengthMax && k_lower_ == k_upper_)) {
        for (size_t elem = start; elem < end; ++elem) {
          int64_t t = SizeToLong(elem % (input_rows_ * input_columns_));
          int64_t index = SizeToLong(elem / (input_rows_ * input_columns_));
          int64_t m = t / static_cast<int64_t>(input_columns_);
          int64_t n = t % static_cast<int64_t>(input_columns_);
          int64_t x = n - std::max(k_upper_, ZERO);
          if (n - m == k_upper_) {
            output_data[elem] = diagonal_data[LongToSize(index * static_cast<int64_t>(diagonal_columns_) + x)];
          } else {
            output_data[elem] = input_data[elem];
          }
        }
      } else {
        for (size_t elem = start; elem < end; ++elem) {
          int64_t t = SizeToLong(elem % (input_rows_ * input_columns_));
          int64_t index = SizeToLong(elem / (input_rows_ * input_columns_));
          int64_t m = t / static_cast<int64_t>(input_columns_);
          int64_t n = t % static_cast<int64_t>(input_columns_);
          int64_t d = n - m;
          if (d >= k_lower_ && d <= k_upper_) {
            int64_t x = k_upper_ - d;
            int64_t offset = 0;
            if (((align_ == "RIGHT_LEFT" || align_ == "RIGHT_RIGHT") && d >= 0) ||
                ((align_ == "LEFT_RIGHT" || align_ == "RIGHT_RIGHT") && d <= 0)) {
              offset = max_diag_len_ - std::min(static_cast<int64_t>(input_columns_) - std::max(d, ZERO),
                                                static_cast<int64_t>(input_rows_) + std::min(d, ZERO));
            }
            int64_t y = n - std::max(d, ZERO) + offset;
            size_t position =
              LongToSize(index * static_cast<int64_t>(diagonal_rows_) * static_cast<int64_t>(diagonal_columns_) +
                         x * static_cast<int64_t>(diagonal_columns_) + y);
            output_data[elem] = diagonal_data[position];
          } else {
            output_data[elem] = input_data[elem];
          }
        }
      }
    };
    CPUKernelUtils::ParallelFor(task, input_numelements_);
  }
  return true;
}

std::vector<std::pair<KernelAttr, MatrixSetDiagV3CpuKernelMod::MatrixSetDiagV3Func>>
  MatrixSetDiagV3CpuKernelMod::func_list_ = {{KernelAttr()
                                                .AddInputAttr(kNumberTypeInt8)
                                                .AddInputAttr(kNumberTypeInt8)
                                                .AddInputAttr(kNumberTypeInt32)
                                                .AddOutputAttr(kNumberTypeInt8),
                                              &MatrixSetDiagV3CpuKernelMod::LaunchKernel<int8_t>},
                                             {KernelAttr()
                                                .AddInputAttr(kNumberTypeInt16)
                                                .AddInputAttr(kNumberTypeInt16)
                                                .AddInputAttr(kNumberTypeInt32)
                                                .AddOutputAttr(kNumberTypeInt16),
                                              &MatrixSetDiagV3CpuKernelMod::LaunchKernel<int16_t>},
                                             {KernelAttr()
                                                .AddInputAttr(kNumberTypeInt32)
                                                .AddInputAttr(kNumberTypeInt32)
                                                .AddInputAttr(kNumberTypeInt32)
                                                .AddOutputAttr(kNumberTypeInt32),
                                              &MatrixSetDiagV3CpuKernelMod::LaunchKernel<int32_t>},
                                             {KernelAttr()
                                                .AddInputAttr(kNumberTypeInt64)
                                                .AddInputAttr(kNumberTypeInt64)
                                                .AddInputAttr(kNumberTypeInt32)
                                                .AddOutputAttr(kNumberTypeInt64),
                                              &MatrixSetDiagV3CpuKernelMod::LaunchKernel<int64_t>},
                                             {KernelAttr()
                                                .AddInputAttr(kNumberTypeUInt8)
                                                .AddInputAttr(kNumberTypeUInt8)
                                                .AddInputAttr(kNumberTypeInt32)
                                                .AddOutputAttr(kNumberTypeUInt8),
                                              &MatrixSetDiagV3CpuKernelMod::LaunchKernel<uint8_t>},
                                             {KernelAttr()
                                                .AddInputAttr(kNumberTypeUInt16)
                                                .AddInputAttr(kNumberTypeUInt16)
                                                .AddInputAttr(kNumberTypeInt32)
                                                .AddOutputAttr(kNumberTypeUInt16),
                                              &MatrixSetDiagV3CpuKernelMod::LaunchKernel<uint16_t>},
                                             {KernelAttr()
                                                .AddInputAttr(kNumberTypeUInt32)
                                                .AddInputAttr(kNumberTypeUInt32)
                                                .AddInputAttr(kNumberTypeInt32)
                                                .AddOutputAttr(kNumberTypeUInt32),
                                              &MatrixSetDiagV3CpuKernelMod::LaunchKernel<uint32_t>},
                                             {KernelAttr()
                                                .AddInputAttr(kNumberTypeUInt64)
                                                .AddInputAttr(kNumberTypeUInt64)
                                                .AddInputAttr(kNumberTypeInt32)
                                                .AddOutputAttr(kNumberTypeUInt64),
                                              &MatrixSetDiagV3CpuKernelMod::LaunchKernel<uint64_t>},
                                             {KernelAttr()
                                                .AddInputAttr(kNumberTypeFloat16)
                                                .AddInputAttr(kNumberTypeFloat16)
                                                .AddInputAttr(kNumberTypeInt32)
                                                .AddOutputAttr(kNumberTypeFloat16),
                                              &MatrixSetDiagV3CpuKernelMod::LaunchKernel<float16>},
                                             {KernelAttr()
                                                .AddInputAttr(kNumberTypeFloat32)
                                                .AddInputAttr(kNumberTypeFloat32)
                                                .AddInputAttr(kNumberTypeInt32)
                                                .AddOutputAttr(kNumberTypeFloat32),
                                              &MatrixSetDiagV3CpuKernelMod::LaunchKernel<float>},
                                             {KernelAttr()
                                                .AddInputAttr(kNumberTypeFloat64)
                                                .AddInputAttr(kNumberTypeFloat64)
                                                .AddInputAttr(kNumberTypeInt32)
                                                .AddOutputAttr(kNumberTypeFloat64),
                                              &MatrixSetDiagV3CpuKernelMod::LaunchKernel<double>}};

std::vector<KernelAttr> MatrixSetDiagV3CpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, MatrixSetDiagV3Func> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, MatrixSetDiagV3, MatrixSetDiagV3CpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
