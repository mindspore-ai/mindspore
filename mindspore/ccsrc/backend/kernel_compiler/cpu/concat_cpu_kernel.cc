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

#include "backend/kernel_compiler/cpu/concat_cpu_kernel.h"
#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
template <typename T>
void ConcatCPUKernel<T>::InitKernel(const CNodePtr &kernel_node) {
  node_wpt_ = kernel_node;
  CheckParam(kernel_node);

  axis_ = LongToInt(AnfAlgo::GetNodeAttr<int64_t>(kernel_node, AXIS));
  auto input_1_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  if (axis_ < 0) {
    axis_ = axis_ + SizeToInt(input_1_shape.size());
  }
}

template <typename T>
bool ConcatCPUKernel<T>::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                const std::vector<kernel::AddressPtr> & /*workspace*/,
                                const std::vector<kernel::AddressPtr> &outputs) {
  auto node_ = node_wpt_.lock();
  if (!node_) {
    MS_LOG(EXCEPTION) << "node_wpt_ is expired.";
  }
  size_t input_num = AnfAlgo::GetInputTensorNum(node_);
  std::vector<std::vector<size_t>> input_flat_shape_list;
  for (size_t i = 0; i < input_num; i++) {
    auto input_shape_i = AnfAlgo::GetPrevNodeOutputInferShape(node_, i);
    auto flat_shape = CPUKernelUtils::FlatShapeByAxis(input_shape_i, axis_);
    input_flat_shape_list.push_back(flat_shape);
  }

  auto output_addr = reinterpret_cast<T *>(outputs[0]->addr);
  auto buff_size = outputs[0]->size;
  // each input's row of shape after flat are same
  auto before_axis = input_flat_shape_list[0][0];
  for (size_t i = 0; i < before_axis; ++i) {
    for (size_t j = 0; j < input_num; ++j) {
      auto input_j_addr = reinterpret_cast<T *>(inputs[j]->addr);
      auto copy_num = input_flat_shape_list[j][1];
      auto offset = copy_num * i;
      auto ret = memcpy_s(output_addr, buff_size, input_j_addr + offset, copy_num * sizeof(T));
      if (ret != EOK) {
        MS_LOG(EXCEPTION) << "memcpy failed.";
      }
      output_addr += copy_num;
      buff_size -= copy_num * sizeof(T);
    }
  }
  return true;
}

template <typename T>
void ConcatCPUKernel<T>::CheckParam(const CNodePtr &kernel_node) {
  size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
  if (output_num != 1) {
    MS_LOG(EXCEPTION) << "Output number is " << output_num << ", but ConcatCPUKernel needs 1 output.";
  }
}
}  // namespace kernel
}  // namespace mindspore
