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
#include "backend/kernel_compiler/cpu/rolling_cpu_kernel.h"
#include <math.h>
#include <map>
#include <limits>
#include <algorithm>
#include <type_traits>
#include "common/thread_pool.h"

namespace mindspore {
namespace kernel {
using rolling::Method;
template <typename T, typename S>
void RollingCpuKernel<T, S>::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);

  static const std::map<std::string, Method> kValidMethods = {
    {"max", Method::Max}, {"min", Method::Min}, {"mean", Method::Mean},
    {"sum", Method::Sum}, {"std", Method::Std}, {"var", Method::Var},
  };
  auto method = AnfAlgo::GetNodeAttr<std::string>(kernel_node, METHOD);
  if (kValidMethods.find(method) == kValidMethods.end()) {
    MS_LOG(EXCEPTION) << "[" << method << "] not supported";
  }
  method_ = kValidMethods.at(method);
  auto window = AnfAlgo::GetNodeAttr<int64_t>(kernel_node, WINDOW);
  if (window <= 0) {
    MS_LOG(EXCEPTION) << "window size should not less than 0, but got " << window;
  }
  window_ = LongToInt(window);
  min_periods_ = AnfAlgo::GetNodeAttr<int64_t>(kernel_node, MIN_PERIODS);
  if (min_periods_ <= 0) {
    MS_LOG(EXCEPTION) << "min_periods should not less than 0, but got " << min_periods_;
  }
  center_ = AnfAlgo::GetNodeAttr<bool>(kernel_node, CENTER);
  auto axis = AnfAlgo::GetNodeAttr<int64_t>(kernel_node, AXIS);
  size_t axis_t = axis < 0 ? LongToSize(axis + SizeToLong(input_shape.size())) : LongToSize(axis);
  closed_ = AnfAlgo::GetNodeAttr<std::string>(kernel_node, CLOSED);
  if (axis_t >= input_shape.size()) {
    MS_LOG(EXCEPTION) << "axis should be smaller than the dimension of input tensor " << input_shape.size()
                      << "D, but got " << axis_t;
  }
  axisIterator_.Init(input_shape, axis_t);
  RollingBoundsCalculate();
  MethodSwitch();
}

template <typename T, typename S>
void RollingCpuKernel<T, S>::InitInputOutputSize(const CNodePtr &kernel_node) {
  CPUKernel::InitInputOutputSize(kernel_node);
  size_t element_size = axisIterator_.OuterSize() * axisIterator_.InnerSize() * axisIterator_.AxisSize();
  // input values
  (void)workspace_size_list_.emplace_back((sizeof(size_t) * element_size));
}

template <typename T, typename S>
void RollingCpuKernel<T, S>::RollingBoundsCalculate() {
  int offset = 0;
  if (center_) {
    offset = (window_ - 1) / 2;
  }

  starts_.resize(axisIterator_.AxisSize());
  ends_.resize(axisIterator_.AxisSize());
  int start_offset = 1;
  int end_offset = 1;
  if (closed_ == "left") {
    start_offset -= 1;
    end_offset -= 1;
  } else if (closed_ == "both") {
    start_offset -= 1;
  } else if (closed_ == "neither") {
    end_offset -= 1;
  }
  int axis_size = SizeToInt(axisIterator_.AxisSize());
  for (int i = 0; i < axis_size; ++i) {
    int end = offset + i + end_offset;
    int start = offset + i - window_ + start_offset;
    ends_[i] = IntToSize(std::max(0, std::min(end, axis_size)));
    starts_[i] = IntToSize(std::max(0, std::min(start, axis_size)));
  }
}

template <typename T, typename S>
void RollingCpuKernel<T, S>::MethodSwitch() {
  switch (method_) {
    case Method::Max:
      reduceMethod_ = [](const T *input_addr, const size_t *ids, size_t start, size_t end) {
        T max_value = std::numeric_limits<T>::min();
        for (size_t x = start; x < end; ++x) {
          if (max_value < input_addr[ids[x]]) {
            max_value = input_addr[ids[x]];
          }
        }
        return max_value;
      };
      break;
    case Method::Min:
      reduceMethod_ = [](const T *input_addr, const size_t *ids, size_t start, size_t end) {
        T min_value = std::numeric_limits<T>::max();
        for (size_t x = start; x < end; ++x) {
          if (min_value > input_addr[ids[x]]) {
            min_value = input_addr[ids[x]];
          }
        }
        return min_value;
      };
      break;
    case Method::Sum:
      reduceMethod_ = [](const T *input_addr, const size_t *ids, size_t start, size_t end) {
        T sum = 0;
        for (size_t x = start; x < end; ++x) {
          sum += input_addr[ids[x]];
        }
        return sum;
      };
      break;
    case Method::Mean:
      reduceMethod_ = [](const T *input_addr, const size_t *ids, size_t start, size_t end) {
        T sum = 0;
        for (size_t x = start; x < end; ++x) {
          sum += input_addr[ids[x]];
        }
        return sum / SizeToFloat(end - start);
      };
      break;
    case Method::Var:
      reduceMethod_ = [this](const T *input_addr, const size_t *ids, size_t start, size_t end) {
        return Var(input_addr, ids, start, end);
      };
      break;
    case Method::Std:
      reduceMethod_ = [this](const T *input_addr, const size_t *ids, size_t start, size_t end) {
        return std::sqrt(Var(input_addr, ids, start, end));
      };
      break;
    default:
      MS_LOG(EXCEPTION) << "reduce method is not yet supported: " << method_;
  }
}

template <typename T, typename S>
bool RollingCpuKernel<T, S>::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                    const std::vector<AddressPtr> &outputs) {
  auto input_addr = reinterpret_cast<T *>(inputs[0]->addr);
  auto workspace_addr = reinterpret_cast<size_t *>(workspace[0]->addr);
  auto output_addr = reinterpret_cast<S *>(outputs[0]->addr);

  T nan_value;
  if constexpr (std::is_same_v<T, float>) {
    nan_value = std::nanf("");
  } else if constexpr (std::is_same_v<T, double>) {
    nan_value = std::nan("");
  } else {
    // integer values not support nan
    nan_value = 0;
  }

  std::vector<common::Task> tasks;
  tasks.reserve(axisIterator_.OuterSize() * axisIterator_.InnerSize());
  for (size_t i = 0; i < axisIterator_.OuterSize(); ++i) {
    for (size_t j = 0; j < axisIterator_.InnerSize(); ++j) {
      size_t offset = (i * axisIterator_.InnerSize() + j) * axisIterator_.AxisSize();
      size_t *ids = workspace_addr + offset;
      (void)tasks.emplace_back([this, i, j, nan_value, input_addr, ids, output_addr] {
        AxisIterator iter(axisIterator_);
        iter.SetOffset(i, j);

        // set indexes to avoid duplicate calculation
        for (size_t k = 0; k < iter.AxisSize(); ++k) {
          ids[k] = iter.GetPos(k);
        }

        for (size_t w = 0; w < iter.AxisSize(); ++w) {
          if (ends_[w] - starts_[w] < static_cast<size_t>(min_periods_)) {
            output_addr[iter.GetPos(w)] = nan_value;
          } else {
            output_addr[iter.GetPos(w)] = reduceMethod_(input_addr, ids, starts_[w], ends_[w]);
          }
        }
        return common::SUCCESS;
      });
    }
  }
  (void)common::ThreadPool::GetInstance().SyncRun(tasks);
  return true;
}
}  // namespace kernel
}  // namespace mindspore
