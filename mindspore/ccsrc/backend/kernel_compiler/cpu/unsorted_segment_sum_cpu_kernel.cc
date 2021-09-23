/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "runtime/device/cpu/cpu_device_address.h"
#include "common/thread_pool.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kUnsortedSegmentInputsNum = 2;
constexpr size_t kUnsortedSegmentOutputsNum = 1;
}  // namespace

void UnsortedSegmentSumCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
  dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
  segment_ids_dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 1);
  auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  auto segment_ids_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
  auto output_shape = AnfAlgo::GetOutputInferShape(kernel_node, 0);
  if (output_shape.empty()) {
    MS_LOG(EXCEPTION) << "Output shape size is 0";
  }
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
                                         const std::vector<kernel::AddressPtr> &,
                                         const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kUnsortedSegmentInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kUnsortedSegmentOutputsNum, kernel_name_);
  void *input_addr = inputs[0]->addr;
  void *indices_addr = inputs[1]->addr;
  void *output_addr = outputs[0]->addr;
  auto ret = memset_s(output_addr, outputs[0]->size, 0, outputs[0]->size);
  if (ret != EOK) {
    MS_LOG(ERROR) << "Output buff memset fail. ret:" << ret;
    return false;
  }

  if (dtype_ == kNumberTypeInt32 && segment_ids_dtype_ == kNumberTypeInt32) {
    ret = UnsortedSegmentSum(int, int, static_cast<const int *>(input_addr), SizeToInt(unit_num_),
                             SizeToInt(input_dim1_), static_cast<const int *>(indices_addr),
                             static_cast<int *>(output_addr), SizeToInt(output_dim0_), SizeToInt(output_dim1_));
  } else if (dtype_ == kNumberTypeFloat32 && segment_ids_dtype_ == kNumberTypeInt32) {
    ret = UnsortedSegmentSum(float, int, static_cast<const float *>(input_addr), SizeToInt(unit_num_),
                             SizeToInt(input_dim1_), static_cast<const int *>(indices_addr),
                             static_cast<float *>(output_addr), SizeToInt(output_dim0_), SizeToInt(output_dim1_));
  } else if (dtype_ == kNumberTypeInt32 && segment_ids_dtype_ == kNumberTypeInt64) {
    ret = UnsortedSegmentSum(int, int64_t, static_cast<const int *>(input_addr), SizeToInt(unit_num_),
                             SizeToInt(input_dim1_), static_cast<const int64_t *>(indices_addr),
                             static_cast<int *>(output_addr), SizeToInt(output_dim0_), SizeToInt(output_dim1_));
  } else if (dtype_ == kNumberTypeFloat32 && segment_ids_dtype_ == kNumberTypeInt64) {
    ret = UnsortedSegmentSum(float, int64_t, static_cast<const float *>(input_addr), SizeToInt(unit_num_),
                             SizeToInt(input_dim1_), static_cast<const int64_t *>(indices_addr),
                             static_cast<float *>(output_addr), SizeToInt(output_dim0_), SizeToInt(output_dim1_));
  } else {
    MS_LOG(ERROR) << "Only support input_x int32 and float32, indices int32 and int64";
    return false;
  }

  if (ret != EOK) {
    MS_LOG(ERROR) << "unsortedSegmentSum failed. ret:" << ret;
    return false;
  }

  return true;
}
}  // namespace kernel
}  // namespace mindspore
