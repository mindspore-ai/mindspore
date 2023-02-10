/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/pad_and_shift_cpu_kernel.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kPadAndShiftInputsNum = 3;
constexpr size_t kPadAndShiftOutputsNum = 1;
}  // namespace

void PadAndShiftCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  node_wpt_ = kernel_node;
  input_x_dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
  type_size_ = GetTypeByte(TypeIdToType(input_x_dtype_));
  auto indices_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  batch_size_ = SizeOf(indices_shape);
  MS_LOG(INFO) << "PadAndShift batch_size:" << batch_size_;
  auto cum_sum_arr_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
  if (cum_sum_arr_shape.size() != 1) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of 'cum_sum_arr' must be 1, but got "
                  << cum_sum_arr_shape.size() << ".";
  }
  cum_sum_size_ = LongToSize(cum_sum_arr_shape[0]);
  is_need_retrieve_output_shape_ = true;
}

bool PadAndShiftCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                     const std::vector<kernel::AddressPtr> &,
                                     const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kPadAndShiftInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kPadAndShiftOutputsNum, kernel_name_);
  if (input_x_dtype_ == kNumberTypeInt32) {
    LaunchKernel<int>(inputs, outputs);
  } else if (input_x_dtype_ == kNumberTypeInt64) {
    LaunchKernel<int64_t>(inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dtype of 'input_x' must be int32 or int64, but got "
                      << input_x_dtype_;
  }
  return true;
}

template <typename T>
void PadAndShiftCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                           const std::vector<kernel::AddressPtr> &outputs) {
  T *input_x = reinterpret_cast<T *>(inputs[0]->addr);
  T *cum_sum_arr = reinterpret_cast<T *>(inputs[1]->addr);
  T shift_idx = *reinterpret_cast<T *>(inputs[2]->addr);
  T *output = reinterpret_cast<T *>(outputs[0]->addr);

  if (shift_idx < 0 || shift_idx >= static_cast<T>(cum_sum_size_)) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', shift index must be large than 0 and less than cumsum size, but got shift index: "
                      << shift_idx << " and cumsum size: " << cum_sum_size_;
  }
  size_t output_size = static_cast<size_t>(cum_sum_arr[cum_sum_size_ - 1]);
  size_t shift_size = static_cast<size_t>(cum_sum_arr[shift_idx]);
  size_t valid_size = static_cast<size_t>(cum_sum_arr[shift_idx + 1] - shift_size);
  int ret = memset_s(output, outputs[0]->size, -1, type_size_ * output_size);
  if (ret != EOK) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', memset_s error. Error no: " << ret;
  }
  ret = memcpy_s(output + shift_size, valid_size * type_size_, input_x, valid_size * type_size_);
  if (ret != EOK) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', memcpy_s error. Error no: " << ret;
  }
  std::vector<size_t> out_shape;
  (void)out_shape.emplace_back(output_size);
  auto node_ = node_wpt_.lock();
  if (!node_) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', node_wpt_(kernel_node) is expired. Error no: " << node_;
  }
  auto output_nums = AnfAlgo::GetOutputTensorNum(node_);
  std::vector<TypeId> dtypes(output_nums);
  for (size_t i = 0; i < output_nums; i++) {
    dtypes[i] = AnfAlgo::GetOutputDeviceDataType(node_, i);
  }
  common::AnfAlgo::SetOutputInferTypeAndShape(dtypes, {Convert2Long(out_shape)}, node_.get());
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, PadAndShift, PadAndShiftCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
