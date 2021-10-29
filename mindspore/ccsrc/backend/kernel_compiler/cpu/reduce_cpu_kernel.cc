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
#include "nnacl/fp32/reduce_fp32.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kReduceSmallVectorSize = 200000;
constexpr size_t kReduceInputsNum = 1;
constexpr size_t kReduceOutputsNum = 1;
}  // namespace

template <typename T>
void ReduceCPUKernel<T>::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
  input_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  auto prim = AnfAlgo::GetCNodePrimitive(kernel_node);
  MS_EXCEPTION_IF_NULL(prim);
  auto axis_addr = prim->GetAttr(AXIS);
  if (axis_addr == nullptr) {
    MS_LOG(EXCEPTION) << "Miss attribute " << AXIS;
  }
  if (axis_addr->isa<ValueTuple>() || axis_addr->isa<ValueList>()) {
    axis_ = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, AXIS);
  } else if (axis_addr->isa<Int64Imm>()) {
    (void)axis_.emplace_back(AnfAlgo::GetNodeAttr<int64_t>(kernel_node, AXIS));
  } else {
    MS_LOG(EXCEPTION) << "Attribute is invalid";
  }

  int dimension = input_shape_.size();
  (void)std::transform(axis_.begin(), axis_.end(), axis_.begin(),
                       [dimension](const auto &a) { return a < 0 ? dimension + a : a; });
  sort(axis_.begin(), axis_.end());
  // Delete the duplicate axis.
  auto last = std::unique(axis_.begin(), axis_.end());
  axis_.erase(last, axis_.end());

  if constexpr (std::is_same<T, bool>::value) {
    if (kernel_name_ == prim::kPrimReduceAll->name()) {
      reduce_type_ = kReduceAll;
      reduce_func_ = [](const T *input, size_t pos, T *out) { *out &= input[pos]; };
    } else if (kernel_name_ == prim::kPrimReduceAny->name()) {
      reduce_type_ = kReduceAny;
      reduce_func_ = [](const T *input, size_t pos, T *out) { *out |= input[pos]; };
    } else {
      MS_LOG(EXCEPTION) << "Unsupported reduce operation: " << kernel_name_ << " for bool.";
    }
  } else {
    if (kernel_name_ == prim::kPrimReduceMax->name()) {
      reduce_type_ = kReduceMax;
      reduce_func_ = [](const T *input, size_t pos, T *out) { *out = std::max(input[pos], *out); };
    } else if (kernel_name_ == prim::kPrimReduceMin->name()) {
      reduce_type_ = kReduceMin;
      reduce_func_ = [](const T *input, size_t pos, T *out) { *out = std::min(input[pos], *out); };
    } else if (kernel_name_ == prim::kPrimReduceSum->name()) {
      reduce_type_ = kReduceSum;
      reduce_func_ = [](const T *input, size_t pos, T *out) { *out += input[pos]; };
    } else if (kernel_name_ == prim::kPrimReduceMean->name()) {
      reduce_type_ = kReduceMean;
      reduce_func_ = [](const T *input, size_t pos, T *out) { *out += input[pos]; };
    } else {
      MS_LOG(EXCEPTION) << "Unsupported reduce operation:  " << kernel_name_;
    }
  }

  // special accelerate for axis = 1 and input has 2 dims
  if constexpr (std::is_same<T, float>::value) {
    if ((reduce_type_ == kReduceMean || reduce_type_ == kReduceSum) && axis_.size() == 1 && axis_[0] == 1 &&
        input_shape_.size() == 2) {
      simple_execute_ = true;
    }
  }
}

template <typename T>
bool ReduceCPUKernel<T>::Launch(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &,
                                const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kReduceInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kReduceOutputsNum, kernel_name_);
  size_t input_size = inputs[0]->size / sizeof(T);
  auto *input_addr = reinterpret_cast<T *>(inputs[0]->addr);
  auto *output_addr = reinterpret_cast<T *>(outputs[0]->addr);
  if (axis_.empty() || input_shape_.empty() || input_shape_.size() == 1) {
    if (input_size < kReduceSmallVectorSize) {
      // Get one ret
      *output_addr = input_addr[0];
      for (size_t i = 1; i < input_size; ++i) {
        reduce_func_(input_addr, i, output_addr);
      }
      if (reduce_type_ == kReduceMean) {
        *output_addr /= input_size;
      }
    } else {
      AccelerateLongVector(input_addr, output_addr, input_size);
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

    size_t output_size = outputs[0]->size / sizeof(T);
    if constexpr (std::is_same<T, float>::value) {
      if (simple_execute_) {
        auto task = [&](size_t start, size_t end) {
          for (size_t i = start; i < end; ++i) {
            (void)ReduceSumDim2Axis1(stride, input_addr + i * stride, output_addr + i);
            if (reduce_type_ == kReduceMean) {
              output_addr[i] /= stride;
            }
          }
        };
        ParallelLaunchAutoSearch(task, output_size, this, &parallel_search_info_);
        return true;
      }
    }
    // Calculate transpose shape
    std::vector<size_t> transpose_shape(input_shape_.size());
    for (int i = 0; i < dimension; ++i) {
      transpose_shape[i] = input_shape_[axes[i]];
    }
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
        if (reduce_type_ == kReduceMean) {
          output_addr[i] /= stride;
        }
      }
    };
    CPUKernelUtils::ParallelFor(task, output_size);
  }
  return true;
}

template <typename T>
void ReduceCPUKernel<T>::AccelerateLongVector(T *input_addr, T *output_addr, size_t input_size) {
  // init output_addr
  *output_addr = input_addr[0];
  std::mutex task_mutex;
  auto task = [this, input_addr, output_addr, &task_mutex](size_t start, size_t end) {
    if (start == 0) {
      ++start;
    }
    if (start == end) {
      return;
    }
    auto block_output = input_addr[start];
    size_t i = start + 1;
    while (i < end) {
      reduce_func_(input_addr, i, &block_output);
      ++i;
    }
    {
      std::lock_guard<std::mutex> task_lock(task_mutex);
      reduce_func_(&block_output, 0, output_addr);
    }
  };
  CPUKernelUtils::ParallelFor(task, input_size);
  if (reduce_type_ == kReduceMean) {
    *output_addr /= input_size;
  }
}
}  // namespace kernel
}  // namespace mindspore
