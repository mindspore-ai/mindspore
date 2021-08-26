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
#include <cmath>
#include <algorithm>
#include <map>
#include <limits>
#include <functional>
#include "common/thread_pool.h"

namespace mindspore {
namespace kernel {

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
  window_ = AnfAlgo::GetNodeAttr<int64_t>(kernel_node, WINDOW);
  if (window_ <= 0) {
    MS_LOG(EXCEPTION) << "window size should not less than 0, but got " << window_;
  }
  min_periods_ = AnfAlgo::GetNodeAttr<int64_t>(kernel_node, MIN_PERIODS);
  if (min_periods_ <= 0) {
    MS_LOG(EXCEPTION) << "min_periods should not less than 0, but got " << min_periods_;
  }
  center_ = AnfAlgo::GetNodeAttr<bool>(kernel_node, CENTER);
  axis_ = AnfAlgo::GetNodeAttr<int64_t>(kernel_node, AXIS);
  closed_ = AnfAlgo::GetNodeAttr<std::string>(kernel_node, CLOSED);
  if (axis_ < 0) {
    axis_ += input_shape.size();
  }
  if ((axis_ < 0) || (axis_ >= static_cast<int64_t>(input_shape.size()))) {
    MS_LOG(EXCEPTION) << "axis should be smaller than the dimension of input tensor " << input_shape.size()
                      << "D, but got " << axis_;
  }
  AxisCalculate(input_shape);
  RollingBoundsCalculate();
  MethodSwitch();
}

template <typename T, typename S>
void RollingCpuKernel<T, S>::AxisCalculate(const std::vector<size_t> &input_shape) {
  outer_size_ = 1;
  for (int i = 0; i < axis_; i++) {
    outer_size_ *= input_shape[i];
  }

  axis_size_ = input_shape[axis_];

  inner_size_ = 1;
  for (int i = axis_ + 1; i < static_cast<int>(input_shape.size()); ++i) {
    inner_size_ *= input_shape[i];
  }
}

template <typename T, typename S>
void RollingCpuKernel<T, S>::RollingBoundsCalculate() {
  int offset = 0;
  if (center_) {
    offset = (window_ - 1) / 2;
  }
  starts_.resize(axis_size_);
  ends_.resize(axis_size_);
  int start_offset = 0;
  int end_offset = 0;
  if (closed_ == "left") {
    start_offset -= 1;
    end_offset -= 1;
  } else if (closed_ == "both") {
    start_offset -= 1;
  } else if (closed_ == "neither") {
    end_offset -= 1;
  }
  for (int i = 0; i < axis_size_; ++i) {
    int end = offset + i + 1;
    int start = end - window_;
    ends_[i] = std::max(0, std::min(end + end_offset, axis_size_));
    starts_[i] = std::max(0, std::min(start + start_offset, axis_size_));
  }
}

template <typename T, typename S>
void RollingCpuKernel<T, S>::MethodSwitch() {
  switch (method_) {
    case Method::Max:
      reduceMethod_ = [this](const T *input_addr, int outer_offset, int w, int col) {
        T max_value = std::numeric_limits<T>::min();
        for (int x = starts_[w]; x < ends_[w]; ++x) {
          int index = outer_offset + x * inner_size_ + col;
          if (max_value < input_addr[index]) {
            max_value = input_addr[index];
          }
        }
        return max_value;
      };
      break;
    case Method::Min:
      reduceMethod_ = [this](const T *input_addr, int outer_offset, int w, int col) {
        T min_value = std::numeric_limits<T>::max();
        for (int x = starts_[w]; x < ends_[w]; ++x) {
          int index = outer_offset + x * inner_size_ + col;
          if (min_value > input_addr[index]) {
            min_value = input_addr[index];
          }
        }
        return min_value;
      };
      break;
    case Method::Sum:
      reduceMethod_ = [this](const T *input_addr, int outer_offset, int w, int col) {
        T sum = 0;
        for (int x = starts_[w]; x < ends_[w]; ++x) {
          int index = outer_offset + x * inner_size_ + col;
          sum += input_addr[index];
        }
        return sum;
      };
      break;
    case Method::Mean:
      reduceMethod_ = [this](const T *input_addr, int outer_offset, int w, int col) {
        T sum = 0;
        for (int x = starts_[w]; x < ends_[w]; ++x) {
          int index = outer_offset + x * inner_size_ + col;
          sum += input_addr[index];
        }
        return sum * 1.0 / (ends_[w] - starts_[w]);
      };
      break;
    case Method::Var:
      reduceMethod_ = [this](const T *input_addr, int outer_offset, int w, int col) {
        // float for division
        float n = ends_[w] - starts_[w];
        T sum1 = 0;
        for (int x = starts_[w]; x < ends_[w]; ++x) {
          int index = outer_offset + x * inner_size_ + col;
          sum1 += input_addr[index];
        }
        double mean = sum1 / n;
        double sum2 = 0;
        for (int x = starts_[w]; x < ends_[w]; ++x) {
          int index = outer_offset + x * inner_size_ + col;
          sum2 += (input_addr[index] - mean) * (input_addr[index] - mean);
        }
        // ddof = 1
        return sum2 / (n - 1);
      };
      break;
    case Method::Std:
      reduceMethod_ = [this](const T *input_addr, int outer_offset, int w, int col) {
        // float for division
        float n = ends_[w] - starts_[w];
        T sum1 = 0;
        for (int x = starts_[w]; x < ends_[w]; ++x) {
          int index = outer_offset + x * inner_size_ + col;
          sum1 += input_addr[index];
        }
        double mean = sum1 / n;
        double sum2 = 0;
        for (int x = starts_[w]; x < ends_[w]; ++x) {
          int index = outer_offset + x * inner_size_ + col;
          sum2 += (input_addr[index] - mean) * (input_addr[index] - mean);
        }
        // ddof = 1
        return std::sqrt(sum2 / (n - 1));
      };
      break;
    default:
      MS_LOG(EXCEPTION) << "reduce method is not yet supported: " << method_;
  }
}

template <typename T, typename S>
bool RollingCpuKernel<T, S>::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                    const std::vector<AddressPtr> &outputs) {
  size_t input_size = inputs[0]->size / sizeof(T);
  if (input_size == 0) {
    MS_LOG(EXCEPTION) << "Input data size is 0.";
  }

  auto input_addr = reinterpret_cast<T *>(inputs[0]->addr);
  auto output_addr = reinterpret_cast<S *>(outputs[0]->addr);

  for (int i = 0; i < outer_size_; ++i) {
    int outer_offset = i * axis_size_ * inner_size_;
    for (int col = 0; col < inner_size_; ++col) {
      for (int w = 0; w < axis_size_; ++w) {
        int result_offset = outer_offset + w * inner_size_ + col;
        if (ends_[w] - starts_[w] < min_periods_) {
          if constexpr (std::is_same_v<T, float>) {
            output_addr[result_offset] = std::nanf("");
          } else if constexpr (std::is_same_v<T, double>) {
            output_addr[result_offset] = std::nan("");
          } else {
            // integer values not support nan
            output_addr[result_offset] = 0;
          }
        } else {
          output_addr[result_offset] = reduceMethod_(input_addr, outer_offset, w, col);
        }
      }
    }
  }

  return true;
}

}  // namespace kernel
}  // namespace mindspore
