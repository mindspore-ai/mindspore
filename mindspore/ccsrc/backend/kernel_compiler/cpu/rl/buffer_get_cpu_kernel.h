/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_RL_BUFFER_GET_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_RL_BUFFER_GET_CPU_KERNEL_H_
#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include "backend/kernel_compiler/cpu/cpu_kernel.h"
#include "backend/kernel_compiler/cpu/cpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
class BufferCPUGetKernel : public CPUKernel {
 public:
  BufferCPUGetKernel() : element_nums_(0), capacity_(0) {}

  ~BufferCPUGetKernel() override = default;
  void Init(const CNodePtr &kernel_node) {
    auto shapes = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, "buffer_elements");
    auto types = AnfAlgo::GetNodeAttr<std::vector<TypePtr>>(kernel_node, "buffer_dtype");
    capacity_ = AnfAlgo::GetNodeAttr<int64_t>(kernel_node, "capacity");
    element_nums_ = shapes.size();
    for (size_t i = 0; i < element_nums_; i++) {
      exp_element_list.push_back(shapes[i] * UnitSizeInBytes(types[i]->type_id()));
    }
    // buffer size
    for (auto i : exp_element_list) {
      input_size_list_.push_back(i * capacity_);
      output_size_list_.push_back(i);
    }
    // count, head, index
    input_size_list_.push_back(sizeof(int));
    input_size_list_.push_back(sizeof(int));
    input_size_list_.push_back(sizeof(int));
  }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs) {
    auto count_addr = GetDeviceAddress<int>(inputs, element_nums_);
    auto head_addr = GetDeviceAddress<int>(inputs, element_nums_ + 1);
    auto index_addr = GetDeviceAddress<int>(inputs, element_nums_ + 2);
    int index = index_addr[0];
    if (index_addr[0] < 0) index += count_addr[0];
    if (!(index >= 0 && index < count_addr[0])) {
      MS_LOG(ERROR) << "The index " << index_addr[0] << " is out of range:[ " << -1 * count_addr[0] << ", "
                    << count_addr[0] << ").";
    }
    int t = count_addr[0] - head_addr[0];
    if (index < t) {
      index += head_addr[0];
    } else {
      index -= t;
    }
    auto task = [&](size_t start, size_t end) {
      for (size_t i = start; i < end; i++) {
        auto buffer_addr = GetDeviceAddress<unsigned char>(inputs, i);
        auto item_addr = GetDeviceAddress<unsigned char>(outputs, i);
        size_t one_exp_len = output_size_list_[i];
        size_t dist_len = one_exp_len;
        if (memcpy_s(item_addr, one_exp_len, buffer_addr + IntToSize(index) * one_exp_len, dist_len) != EOK) {
          MS_LOG(EXCEPTION) << "Launch kernel error: memcpy failed";
        }
      }
    };
    CPUKernelUtils::ParallelFor(task, element_nums_);
    return true;
  }

  void InitKernel(const CNodePtr &kernel_node) { return; }

 protected:
  void InitSizeLists() { return; }

 private:
  size_t element_nums_;
  int64_t capacity_;
  std::vector<size_t> exp_element_list;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_RL_BUFFER_GET_CPU_KERNEL_H_
