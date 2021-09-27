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

#include "backend/kernel_compiler/cpu/searchsorted_cpu_kernel.h"

#include <vector>
#include <numeric>
#include <functional>

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kSearchSortedInputsNum = 2;
constexpr size_t kSearchSortedOutputsNum = 1;
}  // namespace

template <typename S, typename T>
void SearchSortedCPUKernel<S, T>::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
  right_ = AnfAlgo::GetNodeAttr<bool>(kernel_node, "right");
  sequence_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  values_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, 1);
  search_len = sequence_shape_.back();
}

template <typename S, typename T>
const S *SearchSortedCPUKernel<S, T>::CustomizedLowerBound(const S *seq_start, const S *seq_end, const S key) {
  while (seq_start < seq_end) {
    const S *mid = seq_start + ((seq_end - seq_start) / 2);
    if (!(key <= *mid)) {
      seq_start = mid + 1;
    } else {
      seq_end = mid;
    }
  }
  return seq_start;
}

template <typename S, typename T>
bool SearchSortedCPUKernel<S, T>::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                         const std::vector<kernel::AddressPtr> &,
                                         const std::vector<kernel::AddressPtr> &outputs) {
  CheckParam(inputs, outputs);
  auto sequence = reinterpret_cast<S *>(inputs[0]->addr);
  auto values = reinterpret_cast<S *>(inputs[1]->addr);
  auto output = reinterpret_cast<T *>(outputs[0]->addr);
  size_t elem_num = inputs[1]->size / sizeof(S);
  size_t seq_dim = sequence_shape_.size();
  size_t search_repeat = values_shape_.back();

  auto task = [this, &sequence, &values, &output, seq_dim, search_repeat](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      auto seq_start = (seq_dim == 1) ? sequence : sequence + (i / search_repeat) * search_len;
      auto result = right_ ? std::upper_bound(seq_start, seq_start + search_len, values[i]) - seq_start
                           : CustomizedLowerBound(seq_start, seq_start + search_len, values[i]) - seq_start;
      output[i] = static_cast<T>(result);
    }
  };
  CPUKernelUtils::ParallelFor(task, elem_num);
  return true;
}

template <typename S, typename T>
void SearchSortedCPUKernel<S, T>::CheckParam(const std::vector<AddressPtr> &inputs,
                                             const std::vector<AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kSearchSortedInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kSearchSortedOutputsNum, kernel_name_);

  if (outputs[0]->size / sizeof(T) != inputs[1]->size / sizeof(S)) {
    MS_LOG(EXCEPTION) << "The output dimensions " << outputs[0]->size << " must match the dimensions of input values "
                      << inputs[1]->size;
  }

  auto sequence = reinterpret_cast<S *>(inputs[0]->addr);
  int list_count = accumulate(sequence_shape_.begin(), sequence_shape_.end() - 1, 1, std::multiplies<int>());
  auto task = [this, &sequence](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      for (size_t j = 0; j < search_len - 1; j++) {
        if (sequence[i * search_len + j] > sequence[i * search_len + j + 1]) {
          MS_LOG(EXCEPTION) << "The input sequence must be sorted.";
        }
      }
    }
  };
  CPUKernelUtils::ParallelFor(task, IntToSize(list_count));
}
}  // namespace kernel
}  // namespace mindspore
