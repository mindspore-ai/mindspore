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
#include "plugin/device/cpu/kernel/sort_cpu_kernel.h"
#include "common/thread_pool.h"

namespace mindspore {
namespace kernel {
template <typename T>
void SortCpuKernelMod<T>::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
  size_t input_count = AnfAlgo::GetInputTensorNum(kernel_node);
  if (input_count != 1) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs should be 1, but got " << input_count
                      << " input(s).";
  }

  size_t output_count = AnfAlgo::GetOutputTensorNum(kernel_node);
  if (output_count != 2) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of outputs should be 2, but got " << output_count
                      << " output(s).";
  }

  auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  descending_ = AnfAlgo::GetNodeAttr<bool>(kernel_node, "descending");
  auto axis = AnfAlgo::GetNodeAttr<int64_t>(kernel_node, AXIS);
  size_t axis_t = axis < 0 ? LongToSize(axis + SizeToLong(input_shape.size())) : LongToSize(axis);
  if (axis_t >= input_shape.size()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the 'axis' should be less than the dimension of input tensor "
                      << input_shape.size() << "D, but got " << axis_t;
  }

  axisIterator_.Init(input_shape, axis_t);
}

template <typename T>
void SortCpuKernelMod<T>::InitInputOutputSize(const CNodePtr &kernel_node) {
  NativeCpuKernelMod::InitInputOutputSize(kernel_node);
  size_t element_size = axisIterator_.OuterSize() * axisIterator_.InnerSize() * axisIterator_.AxisSize();
  // id
  (void)workspace_size_list_.emplace_back((sizeof(size_t) * element_size));
}

template <typename T>
bool SortCpuKernelMod<T>::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                 const std::vector<AddressPtr> &outputs) {
  if (inputs.size() != 1) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs should be 1, but got " << inputs.size()
                      << " input(s).";
  }
  if (outputs.size() != 2) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of outputs should be 2, but got " << outputs.size()
                      << " output(s).";
  }
  if (inputs[0]->size != axisIterator_.OuterSize() * axisIterator_.AxisSize() * axisIterator_.InnerSize() * sizeof(T)) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the memory size of inputs error.";
  }
  auto input = reinterpret_cast<T *>(inputs[0]->addr);
  auto ids_addr = reinterpret_cast<size_t *>(workspace[0]->addr);
  auto output = reinterpret_cast<T *>(outputs[0]->addr);
  auto indices = reinterpret_cast<int *>(outputs[1]->addr);

  if (outputs[0]->size != inputs[0]->size) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the memory size of the first output should be equal to "
                         "the memory size of input, but got the memory size of the first output "
                      << outputs[0]->size << " and the memory size of input " << inputs[0]->size;
  }

  std::function<bool(size_t, size_t)> comparator;
  if (descending_) {
    comparator = [&input](size_t index_1, size_t index_2) { return input[index_1] > input[index_2]; };
  } else {
    comparator = [&input](size_t index_1, size_t index_2) { return input[index_1] < input[index_2]; };
  }

  std::vector<common::Task> tasks;
  tasks.reserve(axisIterator_.OuterSize() * axisIterator_.InnerSize());
  for (size_t i = 0; i < axisIterator_.OuterSize(); ++i) {
    for (size_t j = 0; j < axisIterator_.InnerSize(); ++j) {
      auto task = [this, i, j, ids_addr, input, indices, output, &comparator]() {
        AxisIterator iter(axisIterator_);
        iter.SetOffset(i, j);

        size_t offset = (i * iter.InnerSize() + j) * iter.AxisSize();
        size_t *idx = ids_addr + offset;
        for (size_t k = 0; k < iter.AxisSize(); ++k) {
          idx[k] = iter.GetPos(k);
        }

        std::stable_sort(idx, idx + iter.AxisSize(), comparator);

        for (size_t k = 0; k < iter.AxisSize(); ++k) {
          const auto index = iter.GetPos(k);
          indices[index] = SizeToInt(iter.RevertPos(idx[k]));
          output[index] = input[idx[k]];
        }
        return common::SUCCESS;
      };
      (void)tasks.emplace_back(task);
    }
  }
  ParallelLaunch(tasks);
  return true;
}
}  // namespace kernel
}  // namespace mindspore
