/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/scatter_nd_arithmetic_cpu_kernel.h"
#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <limits>
#include "include/common/thread_pool.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "kernel/common_utils.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kMinIndiceRank = 2;
template <typename T>
inline T RealDiv(const T &a, const T &b) {
  T zero = static_cast<T>(0);
  if (b == zero) {
    if (a == zero) {
      return std::numeric_limits<T>::quiet_NaN();
    }
    if constexpr (std::numeric_limits<T>::has_infinity) {
      return a > zero ? std::numeric_limits<T>::infinity() : -std::numeric_limits<T>::infinity();
    } else {
      return a > zero ? std::numeric_limits<T>::max() : std::numeric_limits<T>::min();
    }
  }
  return static_cast<T>(a / b);
}
}  // namespace

bool ScatterNdArithmeticCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                           const std::vector<KernelTensorPtr> &inputs,
                                           const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  return MatchKernelFunc(base_operator, inputs, outputs);
}

int ScatterNdArithmeticCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                            const std::vector<KernelTensorPtr> &inputs,
                                            const std::vector<KernelTensorPtr> &outputs,
                                            const std::map<uint32_t, tensor::TensorPtr> &) {
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  input_shape_.clear();
  auto input_shape = inputs.at(kIndex0)->GetShapeVector();
  (void)std::transform(input_shape.begin(), input_shape.end(), std::back_inserter(input_shape_), LongToSize);
  auto indices_shape = inputs.at(kIndex1)->GetShapeVector();
  auto updates_shape = inputs.at(kIndex2)->GetShapeVector();
  const auto indices_rank = indices_shape.size();
  const auto last_indices_value = LongToSize(indices_shape.back());
  const auto update_rank = updates_shape.size();
  constexpr size_t min_indices_rank = 2;
  slice_size_ = last_indices_value;
  batch_size_ = 1;
  inner_size_ = 1;

  if (indices_shape.size() < kMinIndiceRank) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', the dimension of 'indices' must be at least 2, but got "
                             << indices_shape.size();
  }

  for (size_t i = 0; i < update_rank; ++i) {
    if (i <= indices_rank - min_indices_rank) {
      batch_size_ *= LongToSize(indices_shape[i]);
    } else {
      inner_size_ *= LongToSize(updates_shape[i]);
    }
  }

  batch_strides_.resize(last_indices_value);
  // Since the quit condition(i >= 0) is about negative integer,
  // we convert iterated index from unsigned integer to signed integer.
  for (auto i = SizeToLong(last_indices_value) - 1; i >= 0; i--) {
    auto idx = LongToSize(i);
    if (idx == last_indices_value - 1) {
      batch_strides_[idx] = 1;
    } else {
      batch_strides_[idx] = batch_strides_[idx + 1] * input_shape_[idx + 1];
    }
  }
  return KRET_OK;
}

template <typename T>
std::pair<bool, ScatterNdArithmeticCpuKernelMod::ComputeFunc<T>> ScatterNdArithmeticCpuKernelMod::InitComputeFunc() {
  std::pair<bool, ComputeFunc<T>> init_result;
  ComputeFunc<T> compute_func;
  static const mindspore::HashMap<std::string, std::function<T(const T &a, const T &b)>> scatter_nd_arithmetic_func_map{
    {prim::kPrimScatterNdMul->name(), [](const T &a, const T &b) { return a * b; }},
    {prim::kPrimScatterNdDiv->name(), [](const T &a, const T &b) { return RealDiv(a, b); }},
    {prim::kPrimScatterNdAdd->name(), [](const T &a, const T &b) { return a + b; }},
    {prim::kPrimScatterNdSub->name(), [](const T &a, const T &b) { return a - b; }},
    {prim::kPrimScatterNdMax->name(), [](const T &a, const T &b) { return a > b ? a : b; }},
    {prim::kPrimScatterNdMin->name(), [](const T &a, const T &b) { return a > b ? b : a; }},
  };
  auto func_iter = scatter_nd_arithmetic_func_map.find(kernel_name_);
  if (func_iter == scatter_nd_arithmetic_func_map.end()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the current operator does not support this operation.";
    init_result.first = false;
    return init_result;
  }
  auto &binary_func = func_iter->second;
  compute_func = [&binary_func](T *a, size_t a_idx, const T *b, size_t b_idx) {
    auto &atomic_ = reinterpret_cast<std::atomic<T> *>(a)[a_idx];
    T expect = atomic_.load();
    T result;
    do {
      result = binary_func(expect, b[b_idx]);
    } while (!atomic_.compare_exchange_weak(expect, result));
  };
  init_result.first = true;
  init_result.second = compute_func;
  return init_result;
}

template <typename T, typename S>
bool ScatterNdArithmeticCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                                   const std::vector<kernel::AddressPtr> &,
                                                   const std::vector<kernel::AddressPtr> &) {
  auto init_compute_func_result = InitComputeFunc<T>();
  if (!init_compute_func_result.first) {
    return false;
  }
  auto compute_func = init_compute_func_result.second;
  auto input = GetDeviceAddress<T>(inputs, kIndex0);
  auto indices = GetDeviceAddress<S>(inputs, kIndex1);
  auto updates = GetDeviceAddress<T>(inputs, kIndex2);
  int64_t invalid_index_pos = -1;
  auto task = [this, &compute_func, &input, &indices, &updates, &invalid_index_pos](size_t start, size_t end) {
    int pre_batch_idx = -1;
    for (size_t upd_idx = start, out_idx = 0; upd_idx < end; ++upd_idx, ++out_idx) {
      size_t batch_idx = upd_idx / inner_size_;
      // If current position in the same batch, we can same some duplicate computation,
      // otherwise, recompute the out_idx and check if index is valid.
      if (SizeToInt(batch_idx) != pre_batch_idx) {
        pre_batch_idx = SizeToInt(batch_idx);
        out_idx = upd_idx % inner_size_;
        size_t index_idx = batch_idx * slice_size_;
        for (size_t i = 0; i < slice_size_; i++) {
          auto index = indices[index_idx + i];
          if (index < 0 || index >= static_cast<S>(input_shape_[i])) {
            invalid_index_pos = SizeToLong(index_idx);
            break;
          }
          out_idx += batch_strides_[i] * LongToSize(index) * inner_size_;
        }
        if (invalid_index_pos != -1) {
          break;
        }
      }
      compute_func(input, out_idx, updates, upd_idx);
    }
    return common::SUCCESS;
  };

  auto element_size = batch_size_ * inner_size_;
  ParallelLaunch(task, element_size, block_size_, this, pool_);
  if (invalid_index_pos != -1) {
    std::stringstream indices_ss;
    std::stringstream input_shape_ss;
    auto pos = LongToSize(invalid_index_pos);
    for (size_t i = 0; i < slice_size_; i++) {
      if (i > 0) {
        indices_ss << ", ";
        input_shape_ss << ", ";
      }
      indices_ss << std::to_string(indices[pos + i]);
      input_shape_ss << std::to_string(input_shape_[i]);
    }
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the " << pos << "-th value of 'indices'[" << indices_ss.str()
                  << "] is out of range[" + input_shape_ss.str() + "].";
    return false;
  }
  return true;
}

#define SCATTER_ND_ARITHMETIC_CPU_REGISTER(IN_DT0, IN_DT1, IN_DT2, OUT_DT0, T, S)                                    \
  KernelAttr().AddInputAttr(IN_DT0).AddInputAttr(IN_DT1).AddInputAttr(IN_DT2).AddOutputAttr(OUT_DT0).AddOutInRef(0,  \
                                                                                                                 0), \
    &ScatterNdArithmeticCpuKernelMod::LaunchKernel<T, S>

const ScatterNdArithmeticCpuKernelMod::ScatterNdSupportListType &ScatterNdArithmeticCpuKernelMod::GetFuncList() const {
  static const ScatterNdArithmeticCpuKernelMod::ScatterNdSupportListType func_list = {
    {SCATTER_ND_ARITHMETIC_CPU_REGISTER(kNumberTypeFloat64, kNumberTypeInt64, kNumberTypeFloat64, kNumberTypeFloat64,
                                        double, int64_t)},
    {SCATTER_ND_ARITHMETIC_CPU_REGISTER(kNumberTypeFloat32, kNumberTypeInt64, kNumberTypeFloat32, kNumberTypeFloat32,
                                        float, int64_t)},
    {SCATTER_ND_ARITHMETIC_CPU_REGISTER(kNumberTypeFloat16, kNumberTypeInt64, kNumberTypeFloat16, kNumberTypeFloat16,
                                        float16, int64_t)},
    {SCATTER_ND_ARITHMETIC_CPU_REGISTER(kNumberTypeInt64, kNumberTypeInt64, kNumberTypeInt64, kNumberTypeInt64, int64_t,
                                        int64_t)},
    {SCATTER_ND_ARITHMETIC_CPU_REGISTER(kNumberTypeInt32, kNumberTypeInt64, kNumberTypeInt32, kNumberTypeInt32, int32_t,
                                        int64_t)},
    {SCATTER_ND_ARITHMETIC_CPU_REGISTER(kNumberTypeInt16, kNumberTypeInt64, kNumberTypeInt16, kNumberTypeInt16, int16_t,
                                        int64_t)},
    {SCATTER_ND_ARITHMETIC_CPU_REGISTER(kNumberTypeInt8, kNumberTypeInt64, kNumberTypeInt8, kNumberTypeInt8, int8_t,
                                        int64_t)},
    {SCATTER_ND_ARITHMETIC_CPU_REGISTER(kNumberTypeUInt64, kNumberTypeInt64, kNumberTypeUInt64, kNumberTypeUInt64,
                                        uint64_t, int64_t)},
    {SCATTER_ND_ARITHMETIC_CPU_REGISTER(kNumberTypeUInt32, kNumberTypeInt64, kNumberTypeUInt32, kNumberTypeUInt32,
                                        uint32_t, int64_t)},
    {SCATTER_ND_ARITHMETIC_CPU_REGISTER(kNumberTypeUInt16, kNumberTypeInt64, kNumberTypeUInt16, kNumberTypeUInt16,
                                        uint16_t, int64_t)},
    {SCATTER_ND_ARITHMETIC_CPU_REGISTER(kNumberTypeUInt8, kNumberTypeInt64, kNumberTypeUInt8, kNumberTypeUInt8, uint8_t,
                                        int64_t)},
    {SCATTER_ND_ARITHMETIC_CPU_REGISTER(kNumberTypeFloat64, kNumberTypeInt32, kNumberTypeFloat64, kNumberTypeFloat64,
                                        double, int32_t)},
    {SCATTER_ND_ARITHMETIC_CPU_REGISTER(kNumberTypeFloat32, kNumberTypeInt32, kNumberTypeFloat32, kNumberTypeFloat32,
                                        float, int32_t)},
    {SCATTER_ND_ARITHMETIC_CPU_REGISTER(kNumberTypeFloat16, kNumberTypeInt32, kNumberTypeFloat16, kNumberTypeFloat16,
                                        float16, int32_t)},
    {SCATTER_ND_ARITHMETIC_CPU_REGISTER(kNumberTypeInt64, kNumberTypeInt32, kNumberTypeInt64, kNumberTypeInt64, int64_t,
                                        int32_t)},
    {SCATTER_ND_ARITHMETIC_CPU_REGISTER(kNumberTypeInt32, kNumberTypeInt32, kNumberTypeInt32, kNumberTypeInt32, int32_t,
                                        int32_t)},
    {SCATTER_ND_ARITHMETIC_CPU_REGISTER(kNumberTypeInt16, kNumberTypeInt32, kNumberTypeInt16, kNumberTypeInt16, int16_t,
                                        int32_t)},
    {SCATTER_ND_ARITHMETIC_CPU_REGISTER(kNumberTypeInt8, kNumberTypeInt32, kNumberTypeInt8, kNumberTypeInt8, int8_t,
                                        int32_t)},
    {SCATTER_ND_ARITHMETIC_CPU_REGISTER(kNumberTypeUInt64, kNumberTypeInt32, kNumberTypeUInt64, kNumberTypeUInt64,
                                        uint64_t, int32_t)},
    {SCATTER_ND_ARITHMETIC_CPU_REGISTER(kNumberTypeUInt32, kNumberTypeInt32, kNumberTypeUInt32, kNumberTypeUInt32,
                                        uint32_t, int32_t)},
    {SCATTER_ND_ARITHMETIC_CPU_REGISTER(kNumberTypeUInt16, kNumberTypeInt32, kNumberTypeUInt16, kNumberTypeUInt16,
                                        uint16_t, int32_t)},
    {SCATTER_ND_ARITHMETIC_CPU_REGISTER(kNumberTypeUInt8, kNumberTypeInt32, kNumberTypeUInt8, kNumberTypeUInt8, uint8_t,
                                        int32_t)},
  };
  return func_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, ScatterNdAdd, ScatterNdArithmeticCpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, ScatterNdSub, ScatterNdArithmeticCpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, ScatterNdMul, ScatterNdArithmeticCpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, ScatterNdDiv, ScatterNdArithmeticCpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, ScatterNdMax, ScatterNdArithmeticCpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, ScatterNdMin, ScatterNdArithmeticCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
