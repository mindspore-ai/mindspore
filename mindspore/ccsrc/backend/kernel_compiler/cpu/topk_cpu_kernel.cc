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

#include "backend/kernel_compiler/cpu/topk_cpu_kernel.h"
#include <algorithm>
#include "runtime/device/cpu/cpu_device_address.h"
#include "common/thread_pool.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kTopKInputsNum = 2;
constexpr size_t kTopKOutputsNum = 2;
}  // namespace

template <typename T>
void TopKCPUKernel::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspaces,
                                 const std::vector<AddressPtr> &outputs) {
  if (inputs.size() != 2 || outputs.size() != 2) {
    MS_LOG(EXCEPTION) << "TopK needs 2 inputs and 2 outputs, but get inputs: " << inputs.size()
                      << "outputs: " << outputs.size();
  }
  if (inputs[0]->size != outer_size_ * inner_size_ * sizeof(T)) {
    MS_LOG(EXCEPTION) << "Error input data size!";
  }
  if (inputs[1]->size != sizeof(int)) {
    MS_LOG(EXCEPTION) << "Input K must be int!";
  }
  auto input = reinterpret_cast<T *>(inputs[0]->addr);
  int k = reinterpret_cast<int *>(inputs[1]->addr)[0];
  auto workspace = reinterpret_cast<size_t *>(workspaces[0]->addr);
  auto output = reinterpret_cast<T *>(outputs[0]->addr);
  auto indices = reinterpret_cast<int *>(outputs[1]->addr);
  if (k < 1) {
    MS_LOG(EXCEPTION) << "Input k must > 0!";
  }
  size_t k_num = IntToSize(std::min<int>(inner_size_, k));
  if (outputs[0]->size != outer_size_ * k_num * sizeof(T)) {
    MS_LOG(EXCEPTION) << "Error output data size!";
  }

  const std::function<bool(size_t, size_t)> comparator = [input](size_t index_1, size_t index_2) {
    return input[index_1] > input[index_2];
  };

  std::vector<common::Task> tasks;
  tasks.reserve(outer_size_);
  for (size_t i = 0; i < outer_size_; ++i) {
    (void)tasks.emplace_back([this, i, k_num, &comparator, input, workspace, indices, output]() {
      size_t *idx = workspace + i * inner_size_;
      auto base_input = i * inner_size_;
      std::iota(idx, idx + inner_size_, base_input);

      if (sorted_) {
        constexpr float fraction = 0.5;
        const size_t threshold = inner_size_ * fraction;
        // fall back to stable_sort
        if (k_num > threshold) {
          std::stable_sort(idx, idx + inner_size_, comparator);
        } else {
          std::nth_element(idx, idx + SizeToLong(k_num), idx + inner_size_, comparator);
          std::stable_sort(idx, idx + SizeToLong(k_num), comparator);
        }
      } else {
        std::nth_element(idx, idx + SizeToLong(k_num), idx + inner_size_, comparator);
      }

      auto base_output = i * k_num;
      for (size_t j = 0; j < k_num; ++j) {
        indices[base_output + j] = SizeToInt(idx[j]) - SizeToInt(base_input);
        output[base_output + j] = input[idx[j]];
      }
      return common::SUCCESS;
    });
  }
  (void)common::ThreadPool::GetInstance().SyncRun(tasks);
}

void TopKCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  auto x_shape_ = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  if (x_shape_.empty()) {
    MS_LOG(EXCEPTION) << "Input shape is empty";
  }
  for (size_t i = 0; i < x_shape_.size() - 1; ++i) {
    outer_size_ *= x_shape_[i];
  }
  inner_size_ = x_shape_[x_shape_.size() - 1];
  sorted_ = AnfAlgo::GetNodeAttr<bool>(kernel_node, "sorted");
  dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
}

void TopKCPUKernel::InitInputOutputSize(const CNodePtr &kernel_node) {
  CPUKernel::InitInputOutputSize(kernel_node);
  size_t element_size = outer_size_ * inner_size_;
  (void)workspace_size_list_.emplace_back((sizeof(size_t) * element_size));
}

bool TopKCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                           const std::vector<kernel::AddressPtr> &workspaces,
                           const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kTopKInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kTopKOutputsNum, kernel_name_);
  if (dtype_ == kNumberTypeFloat16) {
    LaunchKernel<float16>(inputs, workspaces, outputs);
  } else if (dtype_ == kNumberTypeFloat32) {
    LaunchKernel<float>(inputs, workspaces, outputs);
  } else {
    MS_LOG(EXCEPTION) << "Unsupported input data type: " << dtype_;
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
