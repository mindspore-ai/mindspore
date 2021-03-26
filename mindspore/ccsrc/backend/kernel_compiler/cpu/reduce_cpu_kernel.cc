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
#include <unordered_set>

namespace mindspore {
namespace kernel {
namespace {
const size_t kMaxDim = 10;
}  // namespace
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
  auto kernel_name = AnfAlgo::GetCNodeName(kernel_node);
  if (kernel_name == "ReduceMax") {
    reduce_type_ = 1;
    reduce_func_ = [](const T *input, size_t pos, T *out) { *out = std::max(input[pos], *out); };
  } else if (kernel_name == "ReduceMin") {
    reduce_type_ = 2;
    reduce_func_ = [](const T *input, size_t pos, T *out) { *out = std::min(input[pos], *out); };
  } else if (kernel_name == "ReduceSum") {
    reduce_type_ = 3;
    reduce_func_ = [](const T *input, size_t pos, T *out) { *out += input[pos]; };
  } else if (kernel_name == "ReduceMean") {
    reduce_type_ = 4;
    reduce_func_ = [](const T *input, size_t pos, T *out) { *out += input[pos]; };
  } else {
    MS_LOG(EXCEPTION) << "unsupported reduce type: " << reduce_type_;
  }

  CheckParameter();
}

template <typename T>
bool ReduceCPUKernel<T>::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                const std::vector<kernel::AddressPtr> & /*workspaces*/,
                                const std::vector<kernel::AddressPtr> &outputs) {
  size_t input_size = inputs[0]->size / sizeof(T);
  auto input_addr = reinterpret_cast<T *>(inputs[0]->addr);
  auto output_addr = reinterpret_cast<T *>(outputs[0]->addr);
  if (axis_.empty()) {
    // Get one ret
    *output_addr = input_addr[0];
    for (size_t i = 1; i < input_size; ++i) {
      reduce_func_(input_addr, i, output_addr);
    }
    if (reduce_type_ == 4) {  // 4 is reduce mean
      *output_addr /= input_size;
    }
  } else {
    // transpose->calculate strides->calculate ret
    std::vector<size_t> out_shape;
    std::vector<size_t> strides;
    std::vector<size_t> back_strides;
    size_t stride;
    CalculateTransposeInfo(&out_shape, &strides, &back_strides, &stride);
    int dimension = input_shape_.size();
    std::vector<size_t> coordinates(dimension);
    auto get_next_pos = [&coordinates, &out_shape, &strides, &back_strides, &dimension](size_t &curr_pos) {
      for (int i = dimension - 1; i >= 0; --i) {
        if (coordinates[i] + 1 == out_shape[i]) {
          coordinates[i] = 0;
          curr_pos -= back_strides[i];
        } else {
          coordinates[i]++;
          curr_pos += strides[i];
          break;
        }
      }
    };
    size_t output_size = outputs[0]->size / sizeof(T);
    size_t pos = 0;
    for (size_t i = 0; i < output_size; ++i) {
      if (i != 0) {
        get_next_pos(pos);
      }
      output_addr[i] = input_addr[pos];
      for (size_t j = 1; j < stride; ++j) {
        get_next_pos(pos);
        reduce_func_(input_addr, pos, &output_addr[i]);
      }
      if (reduce_type_ == 4) {  // 4 is reduce mean
        output_addr[i] /= stride;
      }
    }
  }
  return true;
}

template <typename T>
void ReduceCPUKernel<T>::CalculateTransposeInfo(std::vector<size_t> *new_shape, std::vector<size_t> *strides,
                                                std::vector<size_t> *back_strides, size_t *stride) const {
  int dimension = input_shape_.size();
  std::vector<size_t> input_strides(dimension);
  input_strides[dimension - 1] = 1;
  for (int i = dimension - 2; i >= 0; --i) {
    input_strides[i] = input_shape_[i + 1] * input_strides[i + 1];
  }

  // Calculate transpose axes and stride
  std::vector<size_t> axes(dimension);
  int j = 0;
  int k = 0;
  *stride = 1;
  for (int i = 0; i < dimension; ++i) {
    if (i != axis_[j]) {
      axes[k] = i;
      ++k;
    } else {
      *stride *= input_shape_[i];
      ++j;
    }
  }
  for (auto &it : axis_) {
    axes[k] = it;
    ++k;
  }

  // Calculate strides, new_shape, back strides
  strides->resize(dimension);
  new_shape->resize(dimension);
  back_strides->resize(dimension);
  for (int i = dimension - 1; i >= 0; --i) {
    (*strides)[i] = input_strides[axes[i]];
    (*new_shape)[i] = input_shape_[axes[i]];
    (*back_strides)[i] = ((*new_shape)[i] - 1) * (*strides)[i];
  }
}

template <typename T>
void ReduceCPUKernel<T>::CheckParameter() const {
  if (input_shape_.empty() || input_shape_.size() > kMaxDim) {
    MS_LOG(EXCEPTION) << "Invalid input tensor of dimension: " << input_shape_.size();
  }

  if (axis_.empty()) {
    MS_LOG(INFO) << "axis is empty";
    return;
  }

  std::unordered_set<int> checker(axis_.begin(), axis_.end());
  if (checker.size() != axis_.size()) {
    MS_LOG(EXCEPTION) << "Duplicate value in axis";
  }

  int maxDimension = input_shape_.size();
  for (auto &axis : axis_) {
    if (axis >= maxDimension) {
      MS_LOG(EXCEPTION) << "Invalid value in axis: " << axis;
    }
  }
}
}  // namespace kernel
}  // namespace mindspore
