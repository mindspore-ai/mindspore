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

#include "backend/kernel_compiler/cpu/reduce_cpu_kernel.h"
#include <string>
#include <vector>
#include <algorithm>
#include <utility>

namespace mindspore {
namespace kernel {
template <typename T>
void ReduceCPUKernel<T>::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  input_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  auto axis_addr = AnfAlgo::GetCNodePrimitive(kernel_node)->GetAttr(AXIS);
  if (axis_addr->isa<ValueTuple>() || axis_addr->isa<ValueList>()) {
    axis_ = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, AXIS);
  } else if (axis_addr->isa<Int64Imm>()) {
    axis_.emplace_back(AnfAlgo::GetNodeAttr<int64_t>(kernel_node, AXIS));
  } else {
    MS_LOG(EXCEPTION) << "Attribute is invalid";
  }

  int dimension = input_shape_.size();
  std::transform(axis_.begin(), axis_.end(), axis_.begin(),
                 [dimension](const auto &a) { return a < 0 ? dimension + a : a; });
  sort(axis_.begin(), axis_.end());
  // Delete the duplicate axis.
  auto last = std::unique(axis_.begin(), axis_.end());
  axis_.erase(last, axis_.end());
  auto kernel_name = AnfAlgo::GetCNodeName(kernel_node);

  if constexpr (std::is_same<T, bool>::value) {
    if (kernel_name == "ReduceAll") {
      reduce_type_ = ReduceType::ReduceAll;
      reduce_func_ = [](const T *input, size_t pos, T *out) { *out &= input[pos]; };
    } else if (kernel_name == "ReduceAny") {
      reduce_type_ = ReduceType::ReduceAny;
      reduce_func_ = [](const T *input, size_t pos, T *out) { *out |= input[pos]; };
    } else {
      MS_LOG(EXCEPTION) << "Unsupported reduce operation: " << kernel_name_ << " for bool.";
    }
  } else {
    if (kernel_name == "ReduceMax") {
      reduce_type_ = ReduceType::ReduceMax;
      reduce_func_ = [](const T *input, size_t pos, T *out) { *out = std::max(input[pos], *out); };
    } else if (kernel_name == "ReduceMin") {
      reduce_type_ = ReduceType::ReduceMin;
      reduce_func_ = [](const T *input, size_t pos, T *out) { *out = std::min(input[pos], *out); };
    } else if (kernel_name == "ReduceSum") {
      reduce_type_ = ReduceType::ReduceSum;
      reduce_func_ = [](const T *input, size_t pos, T *out) { *out += input[pos]; };
    } else if (kernel_name == "ReduceMean") {
      reduce_type_ = ReduceType::ReduceMean;
      reduce_func_ = [](const T *input, size_t pos, T *out) { *out += input[pos]; };
    } else {
      MS_LOG(EXCEPTION) << "Unsupported reduce operation:  " << kernel_name;
    }
  }
}

template <typename T>
bool ReduceCPUKernel<T>::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                const std::vector<kernel::AddressPtr> & /*workspaces*/,
                                const std::vector<kernel::AddressPtr> &outputs) {
  size_t input_size = inputs[0]->size / sizeof(T);
  auto input_addr = reinterpret_cast<T *>(inputs[0]->addr);
  auto output_addr = reinterpret_cast<T *>(outputs[0]->addr);
  if (axis_.empty() || input_shape_.empty() || input_shape_.size() == 1) {
    // Get one ret
    *output_addr = input_addr[0];
    for (size_t i = 1; i < input_size; ++i) {
      reduce_func_(input_addr, i, output_addr);
    }
    if (reduce_type_ == ReduceType::ReduceMean) {
      *output_addr /= input_size;
    }
  } else {
    // Calculate transpose axes and stride
    int dimension = input_shape_.size();
    size_t stride = 1;
    std::vector<size_t> axes(input_shape_.size());
    size_t j = 0;
    size_t k = 0;
    for (int i = 0; i < dimension; ++i) {
      if (j == axis_.size() || i != axis_[j]) {
        axes[k] = i;
        ++k;
      } else {
        stride *= input_shape_[i];
        ++j;
      }
    }
    for (auto &it : axis_) {
      axes[k] = it;
      ++k;
    }
    // Calculate transpose shape
    std::vector<size_t> transpose_shape(input_shape_.size());
    for (int i = 0; i < dimension; ++i) {
      transpose_shape[i] = input_shape_[axes[i]];
    }
    size_t output_size = outputs[0]->size / sizeof(T);
    TransposeIterator base_iter(std::move(transpose_shape), std::move(axes), input_shape_);
    auto task = [this, &base_iter, input_addr, output_addr, stride](size_t start, size_t end) {
      auto iter = base_iter;
      iter.SetPos(start * stride);
      for (size_t i = start; i < end; ++i) {
        output_addr[i] = input_addr[iter.GetPos()];
        iter.GenNextPos();
        for (size_t j = 1; j < stride; ++j) {
          reduce_func_(input_addr, iter.GetPos(), &output_addr[i]);
          iter.GenNextPos();
        }
        if (reduce_type_ == ReduceType::ReduceMean) {
          output_addr[i] /= stride;
        }
      }
    };
    CPUKernelUtils::ParallelFor(task, output_size);
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
