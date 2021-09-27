/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "backend/kernel_compiler/cpu/pad_and_shift_cpu_kernel.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kPadAndShiftInputsNum = 3;
constexpr size_t kPadAndShiftOutputsNum = 1;
}  // namespace

void PadAndShiftCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
  node_wpt_ = kernel_node;
  input_x_dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
  type_size_ = GetTypeByte(TypeIdToType(input_x_dtype_));
  auto indices_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  batch_size_ = 1;
  for (size_t i = 0; i < indices_shape.size(); ++i) {
    batch_size_ *= indices_shape[i];
  }
  MS_LOG(INFO) << "PadAndShift batch_size:" << batch_size_;
  auto cum_sum_arr_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
  if (cum_sum_arr_shape.size() != 1) {
    MS_LOG(ERROR) << "The shape of cum_sum_arr must be 1.";
  }
  cum_sum_size_ = cum_sum_arr_shape[0];
}

bool PadAndShiftCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                  const std::vector<kernel::AddressPtr> &,
                                  const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kPadAndShiftInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kPadAndShiftOutputsNum, kernel_name_);
  if (input_x_dtype_ == kNumberTypeInt32) {
    LaunchKernel<int>(inputs, outputs);
  } else if (input_x_dtype_ == kNumberTypeInt64) {
    LaunchKernel<int64_t>(inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << "Dtype of input_x only support int32, int64";
  }
  return true;
}

template <typename T>
void PadAndShiftCPUKernel::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                        const std::vector<kernel::AddressPtr> &outputs) {
  T *input_x = reinterpret_cast<T *>(inputs[0]->addr);
  T *cum_sum_arr = reinterpret_cast<T *>(inputs[1]->addr);
  T shift_idx = *reinterpret_cast<T *>(inputs[2]->addr);
  T *output = reinterpret_cast<T *>(outputs[0]->addr);

  if (shift_idx >= static_cast<T>(cum_sum_size_)) {
    MS_LOG(EXCEPTION) << "Shift index must small than cumsum size.";
  }
  size_t output_size = static_cast<size_t>(cum_sum_arr[cum_sum_size_ - 1]);
  size_t shift_size = static_cast<size_t>(cum_sum_arr[shift_idx]);
  size_t valid_size = static_cast<size_t>(cum_sum_arr[shift_idx + 1] - shift_size);
  int ret = memset_s(output, outputs[0]->size, -1, type_size_ * output_size);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "memset_s error, errorno" << ret;
  }
  ret = memcpy_s(output + shift_size, valid_size * type_size_, input_x, valid_size * type_size_);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "memcpy_s error, errorno" << ret;
  }
  std::vector<size_t> out_shape;
  (void)out_shape.emplace_back(output_size);
  auto node_ = node_wpt_.lock();
  if (!node_) {
    MS_LOG(EXCEPTION) << "node_wpt_ is expired.";
  }
  auto output_nums = AnfAlgo::GetOutputTensorNum(node_);
  std::vector<TypeId> dtypes(output_nums);
  for (size_t i = 0; i < output_nums; i++) {
    dtypes[i] = AnfAlgo::GetOutputDeviceDataType(node_, i);
  }
  AnfAlgo::SetOutputInferTypeAndShape(dtypes, {out_shape}, node_.get());
}
}  // namespace kernel
}  // namespace mindspore
