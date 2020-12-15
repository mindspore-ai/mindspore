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

#include "backend/kernel_compiler/cpu/unsorted_segment_sum_cpu_kernel.h"
#include <string>
#include "runtime/device/cpu/cpu_device_address.h"
#include "common/thread_pool.h"

namespace mindspore {
namespace kernel {
void UnsortedSegmentSumCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
  if (input_num != 2) {
    MS_LOG(EXCEPTION) << "Input number is " << input_num << ", but UnsortedSegmentSum needs 2 input.";
  }
  size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
  if (output_num != 1) {
    MS_LOG(EXCEPTION) << "Output number is " << output_num << ", but UnsortedSegmentSum needs 1 output.";
  }
  dtype_ = AnfAlgo::GetPrevNodeOutputInferDataType(kernel_node, 0);
  segment_ids_dtype_ = AnfAlgo::GetPrevNodeOutputInferDataType(kernel_node, 1);
  auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  auto segment_ids_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
  auto output_shape = AnfAlgo::GetOutputInferShape(kernel_node, 0);
  for (size_t i = 0; i < input_shape.size(); ++i) {
    unit_num_ *= input_shape[i];
    if (i >= segment_ids_shape.size()) {
      input_dim1_ *= input_shape[i];
    }
  }
  output_dim0_ = output_shape[0];
  for (size_t j = 1; j < output_shape.size(); j++) {
    output_dim1_ *= output_shape[j];
  }
}

bool UnsortedSegmentSumCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                         const std::vector<kernel::AddressPtr> & /*workspace*/,
                                         const std::vector<kernel::AddressPtr> &outputs) {
  bool ret{true};
  if (dtype_ == kNumberTypeInt32 && segment_ids_dtype_ == kNumberTypeInt32) {
    ret = LaunchKernel<int, int>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat32 && segment_ids_dtype_ == kNumberTypeInt32) {
    ret = LaunchKernel<float, int>(inputs, outputs);
  } else if (dtype_ == kNumberTypeInt32 && segment_ids_dtype_ == kNumberTypeInt64) {
    ret = LaunchKernel<int, int64_t>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat32 && segment_ids_dtype_ == kNumberTypeInt64) {
    ret = LaunchKernel<float, int64_t>(inputs, outputs);
  } else {
    MS_LOG(ERROR) << "Only support input_x int32 and float32, indices int32 and int64";
    return false;
  }
  return ret;
}

template <typename S, typename T>
bool UnsortedSegmentSumCPUKernel::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                               const std::vector<kernel::AddressPtr> &outputs) {
  S *input_addr = reinterpret_cast<S *>(inputs[0]->addr);
  T *indices_addr = reinterpret_cast<T *>(inputs[1]->addr);
  S *output_addr = reinterpret_cast<S *>(outputs[0]->addr);
  auto ret = memset_s(output_addr, outputs[0]->size, 0, outputs[0]->size);
  if (ret != EOK) {
    MS_LOG(ERROR) << "Output buff memset fail. ret:" << ret;
    return false;
  }
  for (size_t i = 0; i < unit_num_; ++i) {
    size_t j = i / input_dim1_;
    size_t k = i % input_dim1_;

    T index = indices_addr[j];
    if (index < 0 || index >= SizeToInt(output_dim0_)) {
      continue;
    }
    size_t output_index = index * output_dim1_ + k;
    output_addr[output_index] += input_addr[i];
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
