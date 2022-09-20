/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/unsorted_segment_arithmetic_cpu_kernel.h"
#include <complex>

namespace mindspore {
namespace kernel {
namespace {
using complex64 = std::complex<float>;
using complex128 = std::complex<double>;
using KernelRunFunc = UnsortedSegmentArithmeticCpuKernelMod::KernelRunFunc;
}  // namespace
#define UNSORTED_SEGMENT_ARITH_CPU_REGISTER(T_DT, S_DT, T, S)             \
  KernelAttr().AddInputAttr(T_DT).AddInputAttr(S_DT).AddOutputAttr(T_DT), \
    &UnsortedSegmentArithmeticCpuKernelMod::LaunchKernel<T, S>
#define UNSORTED_SEGMENT_ARITH_CPU_DY_REGISTER(T_DT, S_DT, DT, T, S)                       \
  KernelAttr().AddInputAttr(T_DT).AddInputAttr(S_DT).AddInputAttr(DT).AddOutputAttr(T_DT), \
    &UnsortedSegmentArithmeticCpuKernelMod::LaunchKernel<T, S>

template <typename T>
T GetInitValue(std::string kernel_name) {
  static const std::map<std::string, T> UnsortedSegmentArithmeticInitValueMap{
    {prim::kPrimUnsortedSegmentMax->name(), std::numeric_limits<T>::lowest()},
    {prim::kPrimUnsortedSegmentMin->name(), std::numeric_limits<T>::max()},
    {prim::kPrimUnsortedSegmentSum->name(), static_cast<T>(0.0)},
    {prim::kPrimUnsortedSegmentProd->name(), static_cast<T>(1.0)}};

  if (UnsortedSegmentArithmeticInitValueMap.find(kernel_name) == UnsortedSegmentArithmeticInitValueMap.end()) {
    MS_LOG(ERROR) << "For '" << kernel_name << "', the current operator does not support this operation.";
  }

  T init_value = UnsortedSegmentArithmeticInitValueMap.at(kernel_name);
  return init_value;
}

template <typename T, typename S>
bool UnsortedSegmentArithmeticCpuKernelMod::ComputeFunc(T *input_addr, S *ids_addr, T *output_addr) {
  for (size_t loop = 0; loop < loop_size_; loop++) {
    auto output_index = ids_addr[loop];
    if (output_index < 0) {
      /* segment_ids is less than 0, drop it */
      continue;
    }
    if (kernel_name_ == prim::kPrimUnsortedSegmentMax->name() ||
        kernel_name_ == prim::kPrimUnsortedSegmentMin->name()) {
      if constexpr (std::is_same_v<T, complex64>) {
        MS_LOG(ERROR) << "For '" << kernel_name_ << "', input_x types can not be complex64.";
      } else if constexpr (std::is_same_v<T, complex128>) {
        MS_LOG(ERROR) << "For '" << kernel_name_ << "', input_x types can not be complex128.";
      } else {
        T *cur_input = input_addr + loop * comp_size_;
        T *cur_output = output_addr + output_index * comp_size_;
        for (size_t comp = 0; comp < comp_size_; comp++) {
          if (kernel_name_ == prim::kPrimUnsortedSegmentMax->name()) {
            cur_output[comp] = cur_input[comp] > cur_output[comp] ? cur_input[comp] : cur_output[comp];
          } else if (kernel_name_ == prim::kPrimUnsortedSegmentMin->name()) {
            cur_output[comp] = cur_input[comp] < cur_output[comp] ? cur_input[comp] : cur_output[comp];
          }
        }
      }
    } else if (kernel_name_ == prim::kPrimUnsortedSegmentSum->name()) {
      T *cur_input = input_addr + loop * comp_size_;
      T *cur_output = output_addr + output_index * comp_size_;
      for (size_t comp = 0; comp < comp_size_; comp++) {
        cur_output[comp] += cur_input[comp];
      }
    } else if (kernel_name_ == prim::kPrimUnsortedSegmentProd->name()) {
      T *cur_input = input_addr + loop * comp_size_;
      T *cur_output = output_addr + output_index * comp_size_;
      for (size_t comp = 0; comp < comp_size_; comp++) {
        cur_output[comp] *= cur_input[comp];
      }
    }
  }
  return true;
}

template <typename T, typename S>
bool UnsortedSegmentArithmeticCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                                         const std::vector<kernel::AddressPtr> &,
                                                         const std::vector<kernel::AddressPtr> &outputs) {
  T init_value = GetInitValue<T>(kernel_name_);

  T *input_src_addr = reinterpret_cast<T *>(inputs[kIndex0]->addr);
  S *ids_src_addr = reinterpret_cast<S *>(inputs[kIndex1]->addr);
  T *output_src_addr = reinterpret_cast<T *>(outputs[kIndex0]->addr);

  for (int64_t i = 0; i < batch_size_; i++) {
    T *input_addr = input_src_addr + i * in_stride_;
    S *ids_addr = ids_src_addr + i * ids_stride_;
    T *output_addr = output_src_addr + i * out_stride_;

    auto task = [output_addr, init_value](size_t start, size_t end) {
      for (size_t i = start; i < end; i++) {
        output_addr[i] = init_value;
      }
    };
    ParallelLaunchAutoSearch(task, out_size_, this, &parallel_search_info_);
    for (size_t loop = 0; loop < loop_size_; loop++) {
      if (ids_addr[loop] >= num_segments_) {
        MS_LOG(ERROR) << "For '" << kernel_name_ << "', segment_ids value should be [0, " << num_segments_ << ")";
        return false;
      }
    }
    bool result = UnsortedSegmentArithmeticCpuKernelMod::ComputeFunc(input_addr, ids_addr, output_addr);
    if (!result) {
      return false;
    }
  }
  return true;
}

bool UnsortedSegmentArithmeticCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                                 const std::vector<KernelTensorPtr> &inputs,
                                                 const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  batch_rank_ = base_operator->get_batch_rank();

  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
    return false;
  }
  return true;
}

int UnsortedSegmentArithmeticCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                                  const std::vector<KernelTensorPtr> &inputs,
                                                  const std::vector<KernelTensorPtr> &outputs,
                                                  const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (auto ret = NativeCpuKernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }

  auto in_shape = inputs[kIndex0]->GetShapeVector();
  auto ids_shapes = inputs[kIndex1]->GetShapeVector();
  auto out_shape = outputs[kIndex0]->GetShapeVector();

  batch_size_ = 1;
  for (int64_t i = 0; i < batch_rank_; i++) {
    batch_size_ *= in_shape[i];
  }
  in_stride_ = 1;
  for (size_t i = batch_rank_; i < in_shape.size(); i++) {
    in_stride_ *= in_shape[i];
  }
  ids_stride_ = 1;
  for (size_t i = batch_rank_; i < ids_shapes.size(); i++) {
    ids_stride_ *= ids_shapes[i];
  }
  out_stride_ = 1;
  for (size_t i = batch_rank_; i < out_shape.size(); i++) {
    out_stride_ *= out_shape[i];
  }

  comp_size_ = 1;
  out_size_ = out_shape[batch_rank_];
  num_segments_ = out_shape[batch_rank_];
  for (size_t i = batch_rank_ + 1; i < out_shape.size(); i++) {
    comp_size_ *= out_shape[i];
    out_size_ *= out_shape[i];
  }
  loop_size_ = 1;
  for (size_t i = batch_rank_; i < in_shape.size(); i++) {
    loop_size_ *= in_shape[i];
  }
  loop_size_ /= comp_size_;
  return KRET_OK;
}

const std::vector<std::pair<KernelAttr, KernelRunFunc>> &UnsortedSegmentArithmeticCpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, KernelRunFunc>> func_list = {
    {UNSORTED_SEGMENT_ARITH_CPU_REGISTER(kNumberTypeFloat64, kNumberTypeInt32, double, int)},
    {UNSORTED_SEGMENT_ARITH_CPU_REGISTER(kNumberTypeFloat64, kNumberTypeInt64, double, int64_t)},
    {UNSORTED_SEGMENT_ARITH_CPU_REGISTER(kNumberTypeFloat32, kNumberTypeInt32, float, int)},
    {UNSORTED_SEGMENT_ARITH_CPU_REGISTER(kNumberTypeFloat32, kNumberTypeInt64, float, int64_t)},
    {UNSORTED_SEGMENT_ARITH_CPU_REGISTER(kNumberTypeFloat16, kNumberTypeInt32, float16, int)},
    {UNSORTED_SEGMENT_ARITH_CPU_REGISTER(kNumberTypeFloat16, kNumberTypeInt64, float16, int64_t)},
    {UNSORTED_SEGMENT_ARITH_CPU_REGISTER(kNumberTypeInt32, kNumberTypeInt32, int, int)},
    {UNSORTED_SEGMENT_ARITH_CPU_REGISTER(kNumberTypeInt32, kNumberTypeInt64, int, int64_t)},
    {UNSORTED_SEGMENT_ARITH_CPU_REGISTER(kNumberTypeUInt8, kNumberTypeInt32, uint8_t, int)},
    {UNSORTED_SEGMENT_ARITH_CPU_REGISTER(kNumberTypeUInt8, kNumberTypeInt64, uint8_t, int64_t)},
    {UNSORTED_SEGMENT_ARITH_CPU_REGISTER(kNumberTypeInt16, kNumberTypeInt32, int16_t, int)},
    {UNSORTED_SEGMENT_ARITH_CPU_REGISTER(kNumberTypeInt16, kNumberTypeInt64, int16_t, int64_t)},
    {UNSORTED_SEGMENT_ARITH_CPU_REGISTER(kNumberTypeInt8, kNumberTypeInt32, int8_t, int)},
    {UNSORTED_SEGMENT_ARITH_CPU_REGISTER(kNumberTypeInt8, kNumberTypeInt64, int8_t, int64_t)},
    {UNSORTED_SEGMENT_ARITH_CPU_REGISTER(kNumberTypeInt64, kNumberTypeInt32, int64_t, int)},
    {UNSORTED_SEGMENT_ARITH_CPU_REGISTER(kNumberTypeInt64, kNumberTypeInt64, int64_t, int64_t)},
    {UNSORTED_SEGMENT_ARITH_CPU_REGISTER(kNumberTypeUInt16, kNumberTypeInt32, uint16_t, int)},
    {UNSORTED_SEGMENT_ARITH_CPU_REGISTER(kNumberTypeUInt16, kNumberTypeInt64, uint16_t, int64_t)},
    {UNSORTED_SEGMENT_ARITH_CPU_REGISTER(kNumberTypeUInt32, kNumberTypeInt32, uint32_t, int)},
    {UNSORTED_SEGMENT_ARITH_CPU_REGISTER(kNumberTypeUInt32, kNumberTypeInt64, uint32_t, int64_t)},
    {UNSORTED_SEGMENT_ARITH_CPU_REGISTER(kNumberTypeUInt64, kNumberTypeInt32, uint64_t, int)},
    {UNSORTED_SEGMENT_ARITH_CPU_REGISTER(kNumberTypeUInt64, kNumberTypeInt64, uint64_t, int64_t)},
    {UNSORTED_SEGMENT_ARITH_CPU_REGISTER(kNumberTypeComplex64, kNumberTypeInt32, complex64, int)},
    {UNSORTED_SEGMENT_ARITH_CPU_REGISTER(kNumberTypeComplex64, kNumberTypeInt64, complex64, int64_t)},
    {UNSORTED_SEGMENT_ARITH_CPU_REGISTER(kNumberTypeComplex128, kNumberTypeInt32, complex128, int)},
    {UNSORTED_SEGMENT_ARITH_CPU_REGISTER(kNumberTypeComplex128, kNumberTypeInt64, complex128, int64_t)},
    {UNSORTED_SEGMENT_ARITH_CPU_DY_REGISTER(kNumberTypeFloat64, kNumberTypeInt32, kNumberTypeInt32, double, int)},
    {UNSORTED_SEGMENT_ARITH_CPU_DY_REGISTER(kNumberTypeFloat64, kNumberTypeInt64, kNumberTypeInt32, double, int64_t)},
    {UNSORTED_SEGMENT_ARITH_CPU_DY_REGISTER(kNumberTypeFloat32, kNumberTypeInt32, kNumberTypeInt32, float, int)},
    {UNSORTED_SEGMENT_ARITH_CPU_DY_REGISTER(kNumberTypeFloat32, kNumberTypeInt64, kNumberTypeInt32, float, int64_t)},
    {UNSORTED_SEGMENT_ARITH_CPU_DY_REGISTER(kNumberTypeInt32, kNumberTypeInt32, kNumberTypeInt32, int, int)},
    {UNSORTED_SEGMENT_ARITH_CPU_DY_REGISTER(kNumberTypeInt32, kNumberTypeInt64, kNumberTypeInt32, int, int64_t)},
    {UNSORTED_SEGMENT_ARITH_CPU_DY_REGISTER(kNumberTypeUInt8, kNumberTypeInt32, kNumberTypeInt32, uint8_t, int)},
    {UNSORTED_SEGMENT_ARITH_CPU_DY_REGISTER(kNumberTypeUInt8, kNumberTypeInt64, kNumberTypeInt32, uint8_t, int64_t)},
    {UNSORTED_SEGMENT_ARITH_CPU_DY_REGISTER(kNumberTypeInt16, kNumberTypeInt32, kNumberTypeInt32, int16_t, int)},
    {UNSORTED_SEGMENT_ARITH_CPU_DY_REGISTER(kNumberTypeInt16, kNumberTypeInt64, kNumberTypeInt32, int16_t, int64_t)},
    {UNSORTED_SEGMENT_ARITH_CPU_DY_REGISTER(kNumberTypeInt8, kNumberTypeInt32, kNumberTypeInt32, int8_t, int)},
    {UNSORTED_SEGMENT_ARITH_CPU_DY_REGISTER(kNumberTypeInt8, kNumberTypeInt64, kNumberTypeInt32, int8_t, int64_t)},
    {UNSORTED_SEGMENT_ARITH_CPU_DY_REGISTER(kNumberTypeInt64, kNumberTypeInt32, kNumberTypeInt32, int64_t, int)},
    {UNSORTED_SEGMENT_ARITH_CPU_DY_REGISTER(kNumberTypeInt64, kNumberTypeInt64, kNumberTypeInt32, int64_t, int64_t)},
    {UNSORTED_SEGMENT_ARITH_CPU_DY_REGISTER(kNumberTypeUInt16, kNumberTypeInt32, kNumberTypeInt32, uint16_t, int)},
    {UNSORTED_SEGMENT_ARITH_CPU_DY_REGISTER(kNumberTypeUInt16, kNumberTypeInt64, kNumberTypeInt32, uint16_t, int64_t)},
    {UNSORTED_SEGMENT_ARITH_CPU_DY_REGISTER(kNumberTypeUInt32, kNumberTypeInt32, kNumberTypeInt32, uint32_t, int)},
    {UNSORTED_SEGMENT_ARITH_CPU_DY_REGISTER(kNumberTypeUInt32, kNumberTypeInt64, kNumberTypeInt32, uint32_t, int64_t)},
    {UNSORTED_SEGMENT_ARITH_CPU_DY_REGISTER(kNumberTypeUInt64, kNumberTypeInt32, kNumberTypeInt32, uint64_t, int)},
    {UNSORTED_SEGMENT_ARITH_CPU_DY_REGISTER(kNumberTypeUInt64, kNumberTypeInt64, kNumberTypeInt32, uint64_t, int64_t)},
    {UNSORTED_SEGMENT_ARITH_CPU_DY_REGISTER(kNumberTypeFloat64, kNumberTypeInt32, kNumberTypeInt64, double, int)},
    {UNSORTED_SEGMENT_ARITH_CPU_DY_REGISTER(kNumberTypeFloat64, kNumberTypeInt64, kNumberTypeInt64, double, int64_t)},
    {UNSORTED_SEGMENT_ARITH_CPU_DY_REGISTER(kNumberTypeFloat32, kNumberTypeInt32, kNumberTypeInt64, float, int)},
    {UNSORTED_SEGMENT_ARITH_CPU_DY_REGISTER(kNumberTypeFloat32, kNumberTypeInt64, kNumberTypeInt64, float, int64_t)},
    {UNSORTED_SEGMENT_ARITH_CPU_DY_REGISTER(kNumberTypeInt32, kNumberTypeInt32, kNumberTypeInt64, int, int)},
    {UNSORTED_SEGMENT_ARITH_CPU_DY_REGISTER(kNumberTypeInt32, kNumberTypeInt64, kNumberTypeInt64, int, int64_t)},
    {UNSORTED_SEGMENT_ARITH_CPU_DY_REGISTER(kNumberTypeUInt8, kNumberTypeInt32, kNumberTypeInt64, uint8_t, int)},
    {UNSORTED_SEGMENT_ARITH_CPU_DY_REGISTER(kNumberTypeUInt8, kNumberTypeInt64, kNumberTypeInt64, uint8_t, int64_t)},
    {UNSORTED_SEGMENT_ARITH_CPU_DY_REGISTER(kNumberTypeInt16, kNumberTypeInt32, kNumberTypeInt64, int16_t, int)},
    {UNSORTED_SEGMENT_ARITH_CPU_DY_REGISTER(kNumberTypeInt16, kNumberTypeInt64, kNumberTypeInt64, int16_t, int64_t)},
    {UNSORTED_SEGMENT_ARITH_CPU_DY_REGISTER(kNumberTypeInt8, kNumberTypeInt32, kNumberTypeInt64, int8_t, int)},
    {UNSORTED_SEGMENT_ARITH_CPU_DY_REGISTER(kNumberTypeInt8, kNumberTypeInt64, kNumberTypeInt64, int8_t, int64_t)},
    {UNSORTED_SEGMENT_ARITH_CPU_DY_REGISTER(kNumberTypeInt64, kNumberTypeInt32, kNumberTypeInt64, int64_t, int)},
    {UNSORTED_SEGMENT_ARITH_CPU_DY_REGISTER(kNumberTypeInt64, kNumberTypeInt64, kNumberTypeInt64, int64_t, int64_t)},
    {UNSORTED_SEGMENT_ARITH_CPU_DY_REGISTER(kNumberTypeUInt16, kNumberTypeInt32, kNumberTypeInt64, uint16_t, int)},
    {UNSORTED_SEGMENT_ARITH_CPU_DY_REGISTER(kNumberTypeUInt16, kNumberTypeInt64, kNumberTypeInt64, uint16_t, int64_t)},
    {UNSORTED_SEGMENT_ARITH_CPU_DY_REGISTER(kNumberTypeUInt32, kNumberTypeInt32, kNumberTypeInt64, uint32_t, int)},
    {UNSORTED_SEGMENT_ARITH_CPU_DY_REGISTER(kNumberTypeUInt32, kNumberTypeInt64, kNumberTypeInt64, uint32_t, int64_t)},
    {UNSORTED_SEGMENT_ARITH_CPU_DY_REGISTER(kNumberTypeUInt64, kNumberTypeInt32, kNumberTypeInt64, uint64_t, int)},
    {UNSORTED_SEGMENT_ARITH_CPU_DY_REGISTER(kNumberTypeUInt64, kNumberTypeInt64, kNumberTypeInt64, uint64_t, int64_t)}};
  return func_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, UnsortedSegmentMin, UnsortedSegmentArithmeticCpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, UnsortedSegmentMax, UnsortedSegmentArithmeticCpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, UnsortedSegmentSum, UnsortedSegmentArithmeticCpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, UnsortedSegmentProd, UnsortedSegmentArithmeticCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
