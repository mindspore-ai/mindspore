/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#include "backend/kernel_compiler/cpu/rank_cpu_kernel.h"
#include <cmath>
#include <functional>
#include <map>
#include <type_traits>
#include "common/thread_pool.h"

namespace mindspore {
namespace kernel {
using rank::Method;
using rank::NaOption;
template <typename T>
void RankCpuKernelMod<T>::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);

  static const std::map<std::string, Method> kValidMethods = {
    {"max", Method::Max},     {"min", Method::Min},     {"average", Method::Average},
    {"first", Method::First}, {"dense", Method::Dense},
  };
  auto method = AnfAlgo::GetNodeAttr<std::string>(kernel_node, METHOD);
  if (kValidMethods.find(method) == kValidMethods.end()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the method should be in " << Map2Str(kValidMethods)
                      << ", but got " << method;
  }
  method_ = kValidMethods.at(method);

  static const std::map<std::string, NaOption> kValidOptions = {
    {"keep", NaOption::Keep},
    {"top", NaOption::Top},
    {"bottom", NaOption::Bottom},
  };
  auto option = AnfAlgo::GetNodeAttr<std::string>(kernel_node, NA_OPTION);
  if (kValidOptions.find(option) == kValidOptions.end()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the option should be in " << Map2Str(kValidOptions)
                      << ", but got " << option;
  }
  option_ = kValidOptions.at(option);

  ascending_ = AnfAlgo::GetNodeAttr<bool>(kernel_node, ASCENDING);
  pct_ = AnfAlgo::GetNodeAttr<bool>(kernel_node, PCT);
  auto axis = AnfAlgo::GetNodeAttr<int64_t>(kernel_node, AXIS);
  axis_ = axis < 0 ? LongToSize(axis + SizeToLong(input_shape.size())) : LongToSize(axis);
  if (axis_ >= input_shape.size()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the evaluated axis should be smaller than the dimension of input tensor "
                      << input_shape.size() << "D, but got " << axis_;
  }

  axisIterator_.Init(input_shape, axis_);
  SetFunc();
}

template <typename T>
void RankCpuKernelMod<T>::InitInputOutputSize(const CNodePtr &kernel_node) {
  NativeCpuKernelMod::InitInputOutputSize(kernel_node);
  size_t element_size = axisIterator_.OuterSize() * axisIterator_.InnerSize() * axisIterator_.AxisSize();
  // id
  (void)workspace_size_list_.emplace_back((sizeof(size_t) * element_size));
  // copy element
  (void)workspace_size_list_.emplace_back((sizeof(T) * element_size));
  if constexpr (!std::is_integral_v<T>) {
    // nan flags
    (void)workspace_size_list_.emplace_back((sizeof(bool) * element_size));
  }
}

template <typename T>
void RankCpuKernelMod<T>::SetFunc() {
  switch (method_) {
    case Method::Max: {
      func_ = [](size_t i, size_t duplicate_count, int culmutive_rank, const AxisIterator &axisIterator,
                 const size_t *const sort_idx, float *const output_addr) {
        for (size_t j = i - duplicate_count + 1; j < i + 1; ++j) {
          output_addr[axisIterator.GetPos(sort_idx[j])] = i + 1;
        }
      };
    } break;
    case Method::Min: {
      func_ = [](size_t i, size_t duplicate_count, int culmutive_rank, const AxisIterator &axisIterator,
                 const size_t *const sort_idx, float *const output_addr) {
        for (size_t j = i - duplicate_count + 1; j < i + 1; ++j) {
          output_addr[axisIterator.GetPos(sort_idx[j])] = i - duplicate_count + 2;
        }
      };
    } break;
    case Method::Average: {
      // how avg is computed directly:
      // sum = (i - duplicate_count + 1) + (i - duplicate_count + 2) +... + i
      //     = duplicate_count * (2 * i - duplicate_count + 1) / 2
      // rank_sum = sum + duplicate_count = duplicate_count * (2 * i -
      // duplicate_count + 3) / 2 avg = rank_sum / duplicate_count = (2 * i -
      // duplicate_count + 3) / 2
      func_ = [](size_t i, size_t duplicate_count, int culmutive_rank, const AxisIterator &axisIterator,
                 const size_t *const sort_idx, float *const output_addr) {
        float avg = (2 * i - duplicate_count + 3) / 2.0;
        for (size_t j = i - duplicate_count + 1; j < i + 1; ++j) {
          output_addr[axisIterator.GetPos(sort_idx[j])] = avg;
        }
      };
    } break;
    case Method::First: {
      func_ = [](size_t i, size_t duplicate_count, int culmutive_rank, const AxisIterator &axisIterator,
                 const size_t *const sort_idx, float *const output_addr) {
        for (size_t j = i - duplicate_count + 1; j < i + 1; ++j) {
          output_addr[axisIterator.GetPos(sort_idx[j])] = j + 1;
        }
      };
    } break;
    case Method::Dense: {
      func_ = [](size_t i, size_t duplicate_count, int culmutive_rank, const AxisIterator &axisIterator,
                 const size_t *const sort_idx, float *const output_addr) {
        for (size_t j = i - duplicate_count + 1; j < i + 1; ++j) {
          output_addr[axisIterator.GetPos(sort_idx[j])] = culmutive_rank;
        }
      };
    } break;
    case Method::MethodNotDefined:
    default:
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << "', method not init. The method should be 'max', 'min', 'average', 'first', or 'dense', "
                        << ", but got " << method_;
  }
}

template <typename T>
void RankCpuKernelMod<T>::Launch1D(const T *input_addr, size_t *sort_idx, T *values, const AxisIterator &iter,
                                   float *output_addr) const {
  const size_t n = axisIterator_.AxisSize();
  for (size_t i = 0; i < n; ++i) {
    values[i] = input_addr[iter.GetPos(i)];
  }

  SortIndex(sort_idx, values, iter);

  int culmutive_rank = 1;
  size_t duplicate_count = 0;

  for (size_t i = 0; i < n; ++i) {
    duplicate_count++;
    if ((i == n - 1) || (values[sort_idx[i]] != values[sort_idx[i + 1]])) {
      func_(i, duplicate_count, culmutive_rank, iter, sort_idx, output_addr);
      culmutive_rank++;
      duplicate_count = 0;
    }
  }
  PctConvert(output_addr, iter, culmutive_rank);
}

template <typename T>
void RankCpuKernelMod<T>::PctConvert(float *output_addr, const AxisIterator &iter, int culmutive_rank) const {
  const size_t n = iter.AxisSize();
  if (pct_) {
    // pct calculation
    if (method_ == Method::Dense) {
      auto size = static_cast<float>(culmutive_rank - 1);
      for (size_t i = 0; i < n; ++i) {
        output_addr[iter.GetPos(i)] = output_addr[iter.GetPos(i)] / size;
      }
    } else {
      auto size = static_cast<float>(n);
      for (size_t i = 0; i < n; ++i) {
        output_addr[iter.GetPos(i)] = output_addr[iter.GetPos(i)] / size;
      }
    }
  }
}

template <typename T>
void RankCpuKernelMod<T>::Launch1D(const T *input_addr, size_t *sort_idx, T *values, bool *is_nan,
                                   const AxisIterator &iter, float *output_addr) const {
  const size_t n = iter.AxisSize();
  T nan_padding_value = GetPaddingValue();

  for (size_t i = 0; i < n; ++i) {
    const T value = input_addr[iter.GetPos(i)];
    if (std::isnan(value)) {
      values[i] = nan_padding_value;
      is_nan[i] = true;
    } else {
      values[i] = value;
      is_nan[i] = false;
    }
  }

  SortIndex(sort_idx, values, iter);

  int culmutive_rank = 1;
  size_t duplicate_count = 0;
  int nans_count = 0;

  for (size_t i = 0; i < n; ++i) {
    duplicate_count++;
    if ((i == n - 1) || std::not_equal_to<T>()(values[sort_idx[i]], values[sort_idx[i + 1]]) ||
        (is_nan[sort_idx[i]] != is_nan[sort_idx[i + 1]])) {
      if ((option_ == NaOption::Keep) && is_nan[sort_idx[i]]) {
        for (size_t j = i - duplicate_count + 1; j < i + 1; ++j) {
          output_addr[iter.GetPos(sort_idx[j])] = NAN;
        }
      } else {
        func_(i, duplicate_count, culmutive_rank, iter, sort_idx, output_addr);
      }
      if (is_nan[sort_idx[i]]) {
        nans_count = duplicate_count;
      }
      culmutive_rank++;
      duplicate_count = 0;
    }
  }
  PctConvert(output_addr, iter, culmutive_rank, nans_count);
}

template <typename T>
void RankCpuKernelMod<T>::PctConvert(float *output_addr, const AxisIterator &iter, int culmutive_rank,
                                     int nans_count) const {
  const size_t n = iter.AxisSize();
  if (pct_) {
    // pct calculation
    if (method_ == Method::Dense) {
      auto size = static_cast<float>(culmutive_rank - 1);
      if ((option_ == NaOption::Keep && (nans_count > 0))) {
        size--;
      }
      for (size_t i = 0; i < n; ++i) {
        output_addr[iter.GetPos(i)] = output_addr[iter.GetPos(i)] / size;
      }
    } else {
      auto size = static_cast<float>(n);
      if (option_ == NaOption::Keep) {
        size -= static_cast<float>(nans_count);
      }
      for (size_t i = 0; i < n; ++i) {
        output_addr[iter.GetPos(i)] = output_addr[iter.GetPos(i)] / size;
      }
    }
  }
}

template <typename T>
bool RankCpuKernelMod<T>::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                 const std::vector<AddressPtr> &outputs) {
  if (inputs.size() != 1) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs should be 1, but got " << inputs.size();
  }
  if (outputs.size() != 1) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of outputs should be 1, but got " << outputs.size();
  }
  if constexpr (std::is_integral_v<T>) {
    if (workspace.size() != 2) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of workspaces should be 2, but got "
                        << workspace.size();
    }
  } else {
    if (workspace.size() != 3) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of workspaces should be 3, but got "
                        << workspace.size();
    }
  }
  auto input_addr = reinterpret_cast<T *>(inputs[0]->addr);
  auto ids_addr = reinterpret_cast<size_t *>(workspace[0]->addr);
  auto values_addr = reinterpret_cast<T *>(workspace[1]->addr);
  auto output_addr = reinterpret_cast<float *>(outputs[0]->addr);

  std::vector<common::Task> tasks;
  tasks.reserve(axisIterator_.OuterSize() * axisIterator_.InnerSize());
  for (size_t i = 0; i < axisIterator_.OuterSize(); ++i) {
    for (size_t j = 0; j < axisIterator_.InnerSize(); ++j) {
      (void)tasks.emplace_back([this, i, j, input_addr, ids_addr, values_addr, workspace, output_addr]() {
        AxisIterator iter(axisIterator_);
        iter.SetOffset(i, j);

        size_t offset = (i * iter.InnerSize() + j) * iter.AxisSize();
        size_t *sort_idx = ids_addr + offset;
        T *values = values_addr + offset;

        if constexpr (std::is_integral_v<T>) {
          Launch1D(input_addr, sort_idx, values, iter, output_addr);
        } else {
          auto flags_addr = reinterpret_cast<bool *>(workspace[2]->addr);
          bool *is_nan = flags_addr + offset;
          Launch1D(input_addr, sort_idx, values, is_nan, iter, output_addr);
        }
        return common::SUCCESS;
      });
    }
  }
  ParallelLaunch(tasks);
  return true;
}
}  // namespace kernel
}  // namespace mindspore
