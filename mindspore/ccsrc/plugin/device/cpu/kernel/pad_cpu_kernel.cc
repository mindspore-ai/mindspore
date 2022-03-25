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

#include "plugin/device/cpu/kernel/pad_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kPadInputsNum = 1;
constexpr size_t kPadOutputsNum = 1;
constexpr size_t kPadElemSize = 2;
}  // namespace

void PadCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  paddings_ = common::AnfAlgo::GetNodeAttr<std::vector<std::vector<int64_t>>>(kernel_node, "paddings");
  dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
  input_shape_ = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  std::vector<size_t> output_shape = AnfAlgo::GetOutputDeviceShape(kernel_node, 0);

  input_rank_ = input_shape_.size();
  if (paddings_.size() != input_rank_) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the dimension of 'paddings' should be equal to the rank of the input, but got the "
                         "dimension of 'paddings': "
                      << paddings_.size() << ", and the rank of the input: " << input_rank_;
  }

  for (size_t i = 0; i < paddings_.size(); i++) {
    if (paddings_[i].size() != kPadElemSize) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << "', each element in 'paddings' should have size 2, but got: " << paddings_[i].size();
    }
    flattened_paddings_.push_back(paddings_[i][0]);
    flattened_paddings_.push_back(paddings_[i][1]);
  }

  for (size_t i = 0; i < input_rank_; i++) {
    input_size_ *= input_shape_[i];
    output_size_ *= (input_shape_[i] + IntToSize(flattened_paddings_[kPadElemSize * i]) +
                     IntToSize(flattened_paddings_[(kPadElemSize * i) + 1]));
  }

  if (input_rank_ < 1) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the rank of input should be greater than or equal to 1, but got the rank of input: "
                      << input_rank_;
  }
  if (output_shape.size() != input_rank_) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the rank of input should be equal to the rank of output, but got the rank of input: "
                      << input_rank_ << ", and the rank of output: " << output_shape.size();
  }
  strides_.resize(input_rank_);
  strides_[input_rank_ - 1] = 1;
  for (int32_t i = SizeToInt(input_rank_) - 2; i >= 0; i--) {
    size_t ind = IntToSize(i);
    strides_[ind] = output_shape[ind + 1] * strides_[ind + 1];
  }
}

bool PadCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &,
                             const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kPadInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kPadOutputsNum, kernel_name_);
  if (dtype_ == kNumberTypeFloat16) {
    LaunchKernel<float16>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat32) {
    LaunchKernel<float>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat64) {
    LaunchKernel<double>(inputs, outputs);
  } else if (dtype_ == kNumberTypeInt32) {
    LaunchKernel<int>(inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the dtype of 'input_x' should be float16, float32, float64, or int32, but got "
                      << TypeIdLabel(dtype_);
  }
  return true;
}

template <typename T>
bool PadCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs) {
  const auto *inputs_addr = reinterpret_cast<T *>(inputs[0]->addr);
  auto *outputs_addr = reinterpret_cast<T *>(outputs[0]->addr);
  if (memset_s(outputs_addr, outputs[0]->size, 0, outputs[0]->size) != EOK) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', output buffer memset failed.";
  }

  auto task = [&inputs_addr, &outputs_addr, this](size_t start, size_t end) {
    for (size_t gt_id = start; gt_id < end; ++gt_id) {
      size_t linear_index = gt_id;
      size_t padded_linear_index = 0;
      for (size_t i = input_rank_; i >= 1; i--) {
        size_t unravel_dimension = input_shape_[i - 1];
        size_t unraveled_index = linear_index % unravel_dimension;
        padded_linear_index +=
          ((unraveled_index + IntToSize(flattened_paddings_[kPadElemSize * (i - 1)])) * strides_[i - 1]);
        linear_index -= unraveled_index;
        linear_index /= unravel_dimension;
      }
      outputs_addr[padded_linear_index] = inputs_addr[gt_id];
    }
  };
  ParallelLaunchAutoSearch(task, input_size_, this, &parallel_search_info_);
  return true;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Pad, PadCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
