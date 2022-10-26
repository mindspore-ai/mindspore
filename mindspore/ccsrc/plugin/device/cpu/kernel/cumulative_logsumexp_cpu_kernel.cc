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

#include "plugin/device/cpu/kernel/cumulative_logsumexp_cpu_kernel.h"
#include <cmath>
#include <string>
#include <thread>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "plugin/device/cpu/kernel/mkldnn/mkl_cpu_kernel.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kCumulativeLogsumexpInputsNum = 2;
constexpr size_t kCumulativeLogsumexpOutputsNum = 1;
constexpr size_t kAxisDimension = 1;
constexpr size_t kAxisShapeSize = 1;
const float float16_exclusive_data = -65504e+0;
const float float_exclusive_data = -3.4028235e+38;
const double double_exclusive_data = -1.7976931348623157e+308;
}  // namespace

void CumulativeLogsumexpCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  shape_ = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
  exclusive_ = common::AnfAlgo::GetNodeAttr<bool>(kernel_node, EXCLUSIVE);
  reverse_ = common::AnfAlgo::GetNodeAttr<bool>(kernel_node, REVERSE);
}

bool CumulativeLogsumexpCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                             const std::vector<kernel::AddressPtr> &,
                                             const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kCumulativeLogsumexpInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kCumulativeLogsumexpOutputsNum, kernel_name_);
  if (dtype_ == kNumberTypeFloat64) {
    LaunchKernel<double>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat32 || dtype_ == kNumberTypeFloat) {
    LaunchKernel<float>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat16) {
    LaunchKernel<float16>(inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', kernel data type " << TypeIdLabel(dtype_) << "not support.";
  }
  return true;
}

template <typename t>
void CumulativeLogsumexpCpuKernelMod::CumulativeProcess(const t *input_data, t *output_data, const uint32_t outer,
                                                        const uint32_t inner, const uint32_t depth) {
  for (size_t outer_index = 0; outer_index < outer; ++outer_index) {
    size_t outer_index_adj;
    if (reverse_) {
      outer_index_adj = (outer - 1) - outer_index;
    } else {
      outer_index_adj = outer_index;
    }
    for (size_t inner_index = 0; inner_index < inner; ++inner_index) {
      double one = 1;
      double temp = 0;
      size_t inner_index_adj;
      if (reverse_) {
        inner_index_adj = (inner - 1) - inner_index;
      } else {
        inner_index_adj = inner_index;
      }
      for (size_t depth_index = 0; depth_index < depth; ++depth_index) {
        size_t depth_index_adj;
        if (reverse_) {
          depth_index_adj = (depth - 1) - depth_index;
        } else {
          depth_index_adj = depth_index;
        }
        size_t index = outer_index_adj;
        index += inner_index_adj * depth * outer;
        index += depth_index_adj * outer;
        if (exclusive_) {
          if (depth_index == 0) {
            if (dtype_ == kNumberTypeFloat16) {
              output_data[index] = static_cast<t>(float16_exclusive_data);
            } else if (dtype_ == kNumberTypeFloat32) {
              output_data[index] = static_cast<t>(float_exclusive_data);
            } else {
              output_data[index] = static_cast<t>(double_exclusive_data);
            }
            temp = static_cast<double>(input_data[index]);
          } else {
            output_data[index] = static_cast<t>(temp);
            double a = temp;
            double b, min, max;
            b = static_cast<double>(input_data[index]);
            min = (a < b) ? a : b;
            max = (a >= b) ? a : b;
            temp = log(one + exp(min - max)) + max;
          }
        } else {
          if (depth_index == 0) {
            output_data[index] = input_data[index];
            temp = static_cast<double>(input_data[index]);
          } else {
            double a = temp;
            double b, min, max;
            b = static_cast<double>(input_data[index]);
            min = (a < b) ? a : b;
            max = (a >= b) ? a : b;
            output_data[index] = static_cast<t>(log(one + exp(min - max)) + max);
            temp = log(one + exp(min - max)) + max;
          }
        }
      }
    }
  }
}

template <typename T>
void CumulativeLogsumexpCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                   const std::vector<AddressPtr> &outputs) {
  auto *input_data = static_cast<T *>(inputs[kIndex0]->addr);
  auto axis_ = static_cast<int32_t *>(inputs[kIndex1]->addr);
  auto *output_data = static_cast<T *>(outputs[kIndex0]->addr);
  size_t lens = inputs[kIndex0]->size > 0 ? static_cast<size_t>(inputs[kIndex0]->size / sizeof(T)) : 1;
  auto task = [this, input_data, axis_, output_data](const size_t start, const size_t end) {
    int32_t x_rank = SizeToInt(shape_.size());
    if (axis_[0] >= x_rank || axis_[0] < -x_rank) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << ", 'axis' must be in range [" << -x_rank << ", " << x_rank
                        << "), but got: " << axis_[0];
    }
    if (axis_[0] < 0) {
      axis_[0] += x_rank;
    }
    uint32_t inner = 1;
    uint32_t depth = shape_[IntToSize(axis_[0])];
    uint32_t outer = 1;
    for (size_t i = 0; i < IntToSize(axis_[0]); i++) {
      inner *= shape_[i];
    }
    for (size_t i = IntToSize(axis_[0]) + 1; i < shape_.size(); i++) {
      outer *= shape_[i];
    }
    CumulativeProcess<T>(input_data, output_data, outer, inner, depth);
  };
  ParallelLaunchAutoSearch(task, lens, this, &parallel_search_info_);
}

std::vector<KernelAttr> CumulativeLogsumexpCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list = {
    KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat16),
    KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
    KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat64)};
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, CumulativeLogsumexp, CumulativeLogsumexpCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
