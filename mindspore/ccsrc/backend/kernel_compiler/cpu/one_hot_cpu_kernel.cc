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

#include "backend/kernel_compiler/cpu/one_hot_cpu_kernel.h"
#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kOneHotInputsNum = 3;
constexpr size_t kOneHotOutputsNum = 1;
}  // namespace

void OneHotCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
  auto output_shape = AnfAlgo::GetOutputInferShape(kernel_node, 0);
  if (output_shape.size() < 2) {
    MS_LOG(EXCEPTION) << "Invalid output shape size: " << output_shape.size();
  }
  int64_t axis = AnfAlgo::GetNodeAttr<int64_t>(kernel_node, AXIS);
  if (axis != -1 && LongToSize(axis) >= output_shape.size()) {
    MS_LOG(EXCEPTION) << "Invalid axis: " << axis;
  }

  if (axis == -1) {
    axis_ = output_shape.size() - 1;
  } else {
    axis_ = LongToSize(axis);
  }
  depth_ = output_shape[axis_];
  stride_ = 1;
  for (size_t i = axis_ + 1; i < output_shape.size(); ++i) {
    stride_ *= output_shape[i];
  }
}

bool OneHotCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &,
                             const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kOneHotInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOneHotOutputsNum, kernel_name_);
  const auto *indices = reinterpret_cast<int *>(inputs[0]->addr);
  auto on_value = reinterpret_cast<float *>(inputs[1]->addr)[0];
  auto off_value = reinterpret_cast<float *>(inputs[2]->addr)[0];
  auto *output = reinterpret_cast<float *>(outputs[0]->addr);
  size_t elem_num = inputs[0]->size / sizeof(int);

  auto task = [this, &indices, &on_value, &off_value, &output](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      size_t stride_num = i / stride_;
      size_t output_index = stride_num * depth_ * stride_ + i % stride_;
      size_t index = IntToSize(indices[i]);
      for (size_t j = 0; j < depth_; j++) {
        if (index == j) {
          output[output_index] = on_value;
        } else {
          output[output_index] = off_value;
        }
        output_index += stride_;
      }
    }
  };
  ParallelLaunchAutoSearch(task, elem_num, this, &parallel_search_info_);

  return true;
}
}  // namespace kernel
}  // namespace mindspore
