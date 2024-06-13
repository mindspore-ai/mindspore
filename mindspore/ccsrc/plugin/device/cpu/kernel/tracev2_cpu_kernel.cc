/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/tracev2_cpu_kernel.h"
#include <functional>
#include <algorithm>
#include <utility>
#include <memory>
#include <complex>
#include "utils/linalg_utils.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kInputNum = 5;
constexpr size_t kOutputNum = 1;
constexpr size_t kIndexX = 0;
constexpr size_t kIndexOffset = 1;
constexpr size_t kIndexAxis1 = 2;
constexpr size_t kIndexAxis2 = 3;
constexpr size_t kIndexDtype = 4;
constexpr size_t kIndexOut = 0;
constexpr size_t kIndexTransX = 0;
}  // namespace

bool TraceV2CpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputNum, kernel_name_);
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << kernel_name_ << " valid cpu kernel does not support this kernel data type: " << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;
  data_unit_size_ = abstract::TypeIdSize(kernel_attr.GetOutputAttr(kIndexOut).dtype);
  return true;
}

int TraceV2CpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  std::vector<int64_t> x_shape = inputs[kIndex0]->GetShapeVector();
  int64_t x_rank = static_cast<int64_t>(x_shape.size());
  if (x_rank < 2) {
    MS_LOG(WARNING) << "For '" << kernel_name_
                    << "', the dim of input 'x' should greateer or equal to 2, but got 'x' at" << x_rank
                    << "-dimention";
    return KRET_RESIZE_FAILED;
  }
  x_size_ = SizeOf(x_shape);
  offset_ = inputs[kIndexOffset]->GetValueWithCheck<int64_t>();
  int64_t axis1 = inputs[kIndexAxis1]->GetValueWithCheck<int64_t>();
  if (axis1 < -x_rank || axis1 >= x_rank) {
    MS_LOG(WARNING) << "For '" << kernel_name_ << "', the 'axis1' must be in [" - x_rank << ", " << x_rank << "), "
                    << "but got " << axis1;
    return KRET_RESIZE_FAILED;
  }
  int64_t axis2 = inputs[kIndexAxis2]->GetValueWithCheck<int64_t>();
  if (axis2 < -x_rank || axis2 >= x_rank) {
    MS_LOG(WARNING) << "For '" << kernel_name_ << "', the 'axis2' must be in [" - x_rank << ", " << x_rank << "), "
                    << "but got " << axis2;
    return KRET_RESIZE_FAILED;
  }
  axis1 = axis1 < 0 ? axis1 + x_rank : axis1;
  axis2 = axis2 < 0 ? axis2 + x_rank : axis2;
  if (axis1 == axis2) {
    MS_LOG(WARNING) << "For '" << kernel_name_
                    << "', the value of 'axis1' and 'axis2' must be different, but got 'axis1': " << axis1
                    << "and 'axis2: " << axis2;
    return KRET_RESIZE_FAILED;
  }
  mat_size_ = x_shape[axis1] * x_shape[axis2];
  mat_row_size_ = x_shape[axis2];
  mat_col_size_ = x_shape[axis1];

  std::vector<int64_t> trans_x_shape;
  std::vector<int64_t> perm_vec;
  batch_size_ = 1;

  for (int64_t i = 0; i < x_rank; i++) {
    if (i != axis1 && i != axis2) {
      trans_x_shape.emplace_back(x_shape[i]);
      perm_vec.emplace_back(i);
      batch_size_ *= x_shape[i];
    }
  }
  trans_x_shape.emplace_back(x_shape[axis1]);
  trans_x_shape.emplace_back(x_shape[axis2]);
  perm_vec.emplace_back(axis1);
  perm_vec.emplace_back(axis2);
  TransposeIterator iter(trans_x_shape, LongVecToSizeVec(perm_vec), x_shape);
  tanspose_index_.clear();
  iter.SetPos(0);
  for (size_t i = 0; i < x_size_; i++) {
    tanspose_index_.emplace_back(iter.GetPos());
    iter.GenNextPos();
  }
  (void)workspace_size_list_.emplace_back(x_size_ * data_unit_size_);
  return KRET_OK;
}

template <typename T_in, typename T_out>
bool TraceV2CpuKernelMod::LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                                       const std::vector<kernel::KernelTensor *> &workspace,
                                       const std::vector<kernel::KernelTensor *> &outputs) {
  T_in *x_array = reinterpret_cast<T_in *>(inputs[kIndexX]->device_ptr());
  T_out *out_array = reinterpret_cast<T_out *>(outputs[kIndexOut]->device_ptr());
  T_out *trans_x_array = reinterpret_cast<T_out *>(workspace[kIndexTransX]->device_ptr());

  for (size_t i = 0; i < x_size_; i++) {
    Cast(x_array + tanspose_index_[i], trans_x_array + i);
  }

  for (int64_t i = 0; i < batch_size_; i++) {
    T_out *trans_x = trans_x_array + i * mat_size_;
    out_array[i] = static_cast<T_out>(0);
    int64_t row_idx;
    int64_t col_idx;
    if (offset_ > 0) {
      row_idx = 0;
      col_idx = offset_;
    } else {
      col_idx = 0;
      row_idx = -offset_;
    }
    while (row_idx < mat_col_size_ && col_idx < mat_row_size_) {
      int64_t idx = row_idx * mat_row_size_ + col_idx;
      out_array[i] += trans_x[idx];
      row_idx++;
      col_idx++;
    }
  }
  return true;
}

#define TRACEV2_CPU_REG(T1, T2, T3, T4)                  \
  KernelAttr()                                           \
    .AddInputAttr(T1)                       /* x */      \
    .AddInputAttr(kNumberTypeInt64)         /* offset */ \
    .AddInputAttr(kNumberTypeInt64)         /* axis1 */  \
    .AddInputAttr(kNumberTypeInt64)         /* axis2 */  \
    .AddOptionalInputAttr(kNumberTypeInt64) /* dtype */  \
    .AddOutputAttr(T2),                                  \
    &TraceV2CpuKernelMod::LaunchKernel<T3, T4>

std::vector<std::pair<KernelAttr, TraceV2CpuKernelMod::TraceV2Func>> TraceV2CpuKernelMod::func_list_ = {
  {TRACEV2_CPU_REG(kNumberTypeUInt8, kNumberTypeUInt8, uint8_t, uint8_t)},
  {TRACEV2_CPU_REG(kNumberTypeUInt8, kNumberTypeUInt16, uint8_t, uint16_t)},
  {TRACEV2_CPU_REG(kNumberTypeUInt8, kNumberTypeUInt32, uint8_t, uint32_t)},
  {TRACEV2_CPU_REG(kNumberTypeUInt8, kNumberTypeUInt64, uint8_t, uint64_t)},
  {TRACEV2_CPU_REG(kNumberTypeUInt8, kNumberTypeInt8, uint8_t, int8_t)},
  {TRACEV2_CPU_REG(kNumberTypeUInt8, kNumberTypeInt16, uint8_t, int16_t)},
  {TRACEV2_CPU_REG(kNumberTypeUInt8, kNumberTypeInt32, uint8_t, int32_t)},
  {TRACEV2_CPU_REG(kNumberTypeUInt8, kNumberTypeInt64, uint8_t, int64_t)},
  {TRACEV2_CPU_REG(kNumberTypeUInt8, kNumberTypeFloat16, uint8_t, float16)},
  {TRACEV2_CPU_REG(kNumberTypeUInt8, kNumberTypeFloat32, uint8_t, float)},
  {TRACEV2_CPU_REG(kNumberTypeUInt8, kNumberTypeFloat64, uint8_t, double)},
  {TRACEV2_CPU_REG(kNumberTypeUInt8, kNumberTypeComplex64, uint8_t, complex64)},
  {TRACEV2_CPU_REG(kNumberTypeUInt8, kNumberTypeComplex128, uint8_t, complex128)},
  {TRACEV2_CPU_REG(kNumberTypeUInt8, kNumberTypeBool, uint8_t, bool)},
  {TRACEV2_CPU_REG(kNumberTypeUInt16, kNumberTypeUInt8, uint16_t, uint8_t)},
  {TRACEV2_CPU_REG(kNumberTypeUInt16, kNumberTypeUInt16, uint16_t, uint16_t)},
  {TRACEV2_CPU_REG(kNumberTypeUInt16, kNumberTypeUInt32, uint16_t, uint32_t)},
  {TRACEV2_CPU_REG(kNumberTypeUInt16, kNumberTypeUInt64, uint16_t, uint64_t)},
  {TRACEV2_CPU_REG(kNumberTypeUInt16, kNumberTypeInt8, uint16_t, int8_t)},
  {TRACEV2_CPU_REG(kNumberTypeUInt16, kNumberTypeInt16, uint16_t, int16_t)},
  {TRACEV2_CPU_REG(kNumberTypeUInt16, kNumberTypeInt32, uint16_t, int32_t)},
  {TRACEV2_CPU_REG(kNumberTypeUInt16, kNumberTypeInt64, uint16_t, int64_t)},
  {TRACEV2_CPU_REG(kNumberTypeUInt16, kNumberTypeFloat16, uint16_t, float16)},
  {TRACEV2_CPU_REG(kNumberTypeUInt16, kNumberTypeFloat32, uint16_t, float)},
  {TRACEV2_CPU_REG(kNumberTypeUInt16, kNumberTypeFloat64, uint16_t, double)},
  {TRACEV2_CPU_REG(kNumberTypeUInt16, kNumberTypeComplex64, uint16_t, complex64)},
  {TRACEV2_CPU_REG(kNumberTypeUInt16, kNumberTypeComplex128, uint16_t, complex128)},
  {TRACEV2_CPU_REG(kNumberTypeUInt16, kNumberTypeBool, uint16_t, bool)},
  {TRACEV2_CPU_REG(kNumberTypeUInt32, kNumberTypeUInt8, uint32_t, uint8_t)},
  {TRACEV2_CPU_REG(kNumberTypeUInt32, kNumberTypeUInt16, uint32_t, uint16_t)},
  {TRACEV2_CPU_REG(kNumberTypeUInt32, kNumberTypeUInt32, uint32_t, uint32_t)},
  {TRACEV2_CPU_REG(kNumberTypeUInt32, kNumberTypeUInt64, uint32_t, uint64_t)},
  {TRACEV2_CPU_REG(kNumberTypeUInt32, kNumberTypeInt8, uint32_t, int8_t)},
  {TRACEV2_CPU_REG(kNumberTypeUInt32, kNumberTypeInt16, uint32_t, int16_t)},
  {TRACEV2_CPU_REG(kNumberTypeUInt32, kNumberTypeInt32, uint32_t, int32_t)},
  {TRACEV2_CPU_REG(kNumberTypeUInt32, kNumberTypeInt64, uint32_t, int64_t)},
  {TRACEV2_CPU_REG(kNumberTypeUInt32, kNumberTypeFloat16, uint32_t, float16)},
  {TRACEV2_CPU_REG(kNumberTypeUInt32, kNumberTypeFloat32, uint32_t, float)},
  {TRACEV2_CPU_REG(kNumberTypeUInt32, kNumberTypeFloat64, uint32_t, double)},
  {TRACEV2_CPU_REG(kNumberTypeUInt32, kNumberTypeComplex64, uint32_t, complex64)},
  {TRACEV2_CPU_REG(kNumberTypeUInt32, kNumberTypeComplex128, uint32_t, complex128)},
  {TRACEV2_CPU_REG(kNumberTypeUInt32, kNumberTypeBool, uint32_t, bool)},
  {TRACEV2_CPU_REG(kNumberTypeUInt64, kNumberTypeUInt8, uint64_t, uint8_t)},
  {TRACEV2_CPU_REG(kNumberTypeUInt64, kNumberTypeUInt16, uint64_t, uint16_t)},
  {TRACEV2_CPU_REG(kNumberTypeUInt64, kNumberTypeUInt32, uint64_t, uint32_t)},
  {TRACEV2_CPU_REG(kNumberTypeUInt64, kNumberTypeUInt64, uint64_t, uint64_t)},
  {TRACEV2_CPU_REG(kNumberTypeUInt64, kNumberTypeInt8, uint64_t, int8_t)},
  {TRACEV2_CPU_REG(kNumberTypeUInt64, kNumberTypeInt16, uint64_t, int16_t)},
  {TRACEV2_CPU_REG(kNumberTypeUInt64, kNumberTypeInt32, uint64_t, int32_t)},
  {TRACEV2_CPU_REG(kNumberTypeUInt64, kNumberTypeInt64, uint64_t, int64_t)},
  {TRACEV2_CPU_REG(kNumberTypeUInt64, kNumberTypeFloat16, uint64_t, float16)},
  {TRACEV2_CPU_REG(kNumberTypeUInt64, kNumberTypeFloat32, uint64_t, float)},
  {TRACEV2_CPU_REG(kNumberTypeUInt64, kNumberTypeFloat64, uint64_t, double)},
  {TRACEV2_CPU_REG(kNumberTypeUInt64, kNumberTypeComplex64, uint64_t, complex64)},
  {TRACEV2_CPU_REG(kNumberTypeUInt64, kNumberTypeComplex128, uint64_t, complex128)},
  {TRACEV2_CPU_REG(kNumberTypeUInt64, kNumberTypeBool, uint64_t, bool)},
  {TRACEV2_CPU_REG(kNumberTypeInt8, kNumberTypeUInt8, int8_t, uint8_t)},
  {TRACEV2_CPU_REG(kNumberTypeInt8, kNumberTypeUInt16, int8_t, uint16_t)},
  {TRACEV2_CPU_REG(kNumberTypeInt8, kNumberTypeUInt32, int8_t, uint32_t)},
  {TRACEV2_CPU_REG(kNumberTypeInt8, kNumberTypeUInt64, int8_t, uint64_t)},
  {TRACEV2_CPU_REG(kNumberTypeInt8, kNumberTypeInt8, int8_t, int8_t)},
  {TRACEV2_CPU_REG(kNumberTypeInt8, kNumberTypeInt16, int8_t, int16_t)},
  {TRACEV2_CPU_REG(kNumberTypeInt8, kNumberTypeInt32, int8_t, int32_t)},
  {TRACEV2_CPU_REG(kNumberTypeInt8, kNumberTypeInt64, int8_t, int64_t)},
  {TRACEV2_CPU_REG(kNumberTypeInt8, kNumberTypeFloat16, int8_t, float16)},
  {TRACEV2_CPU_REG(kNumberTypeInt8, kNumberTypeFloat32, int8_t, float)},
  {TRACEV2_CPU_REG(kNumberTypeInt8, kNumberTypeFloat64, int8_t, double)},
  {TRACEV2_CPU_REG(kNumberTypeInt8, kNumberTypeComplex64, int8_t, complex64)},
  {TRACEV2_CPU_REG(kNumberTypeInt8, kNumberTypeComplex128, int8_t, complex128)},
  {TRACEV2_CPU_REG(kNumberTypeInt8, kNumberTypeBool, int8_t, bool)},
  {TRACEV2_CPU_REG(kNumberTypeInt16, kNumberTypeUInt8, int16_t, uint8_t)},
  {TRACEV2_CPU_REG(kNumberTypeInt16, kNumberTypeUInt16, int16_t, uint16_t)},
  {TRACEV2_CPU_REG(kNumberTypeInt16, kNumberTypeUInt32, int16_t, uint32_t)},
  {TRACEV2_CPU_REG(kNumberTypeInt16, kNumberTypeUInt64, int16_t, uint64_t)},
  {TRACEV2_CPU_REG(kNumberTypeInt16, kNumberTypeInt8, int16_t, int8_t)},
  {TRACEV2_CPU_REG(kNumberTypeInt16, kNumberTypeInt16, int16_t, int16_t)},
  {TRACEV2_CPU_REG(kNumberTypeInt16, kNumberTypeInt32, int16_t, int32_t)},
  {TRACEV2_CPU_REG(kNumberTypeInt16, kNumberTypeInt64, int16_t, int64_t)},
  {TRACEV2_CPU_REG(kNumberTypeInt16, kNumberTypeFloat16, int16_t, float16)},
  {TRACEV2_CPU_REG(kNumberTypeInt16, kNumberTypeFloat32, int16_t, float)},
  {TRACEV2_CPU_REG(kNumberTypeInt16, kNumberTypeFloat64, int16_t, double)},
  {TRACEV2_CPU_REG(kNumberTypeInt16, kNumberTypeComplex64, int16_t, complex64)},
  {TRACEV2_CPU_REG(kNumberTypeInt16, kNumberTypeComplex128, int16_t, complex128)},
  {TRACEV2_CPU_REG(kNumberTypeInt16, kNumberTypeBool, int16_t, bool)},
  {TRACEV2_CPU_REG(kNumberTypeInt32, kNumberTypeUInt8, int32_t, uint8_t)},
  {TRACEV2_CPU_REG(kNumberTypeInt32, kNumberTypeUInt16, int32_t, uint16_t)},
  {TRACEV2_CPU_REG(kNumberTypeInt32, kNumberTypeUInt32, int32_t, uint32_t)},
  {TRACEV2_CPU_REG(kNumberTypeInt32, kNumberTypeUInt64, int32_t, uint64_t)},
  {TRACEV2_CPU_REG(kNumberTypeInt32, kNumberTypeInt8, int32_t, int8_t)},
  {TRACEV2_CPU_REG(kNumberTypeInt32, kNumberTypeInt16, int32_t, int16_t)},
  {TRACEV2_CPU_REG(kNumberTypeInt32, kNumberTypeInt32, int32_t, int32_t)},
  {TRACEV2_CPU_REG(kNumberTypeInt32, kNumberTypeInt64, int32_t, int64_t)},
  {TRACEV2_CPU_REG(kNumberTypeInt32, kNumberTypeFloat16, int32_t, float16)},
  {TRACEV2_CPU_REG(kNumberTypeInt32, kNumberTypeFloat32, int32_t, float)},
  {TRACEV2_CPU_REG(kNumberTypeInt32, kNumberTypeFloat64, int32_t, double)},
  {TRACEV2_CPU_REG(kNumberTypeInt32, kNumberTypeComplex64, int32_t, complex64)},
  {TRACEV2_CPU_REG(kNumberTypeInt32, kNumberTypeComplex128, int32_t, complex128)},
  {TRACEV2_CPU_REG(kNumberTypeInt32, kNumberTypeBool, int32_t, bool)},
  {TRACEV2_CPU_REG(kNumberTypeInt64, kNumberTypeUInt8, int64_t, uint8_t)},
  {TRACEV2_CPU_REG(kNumberTypeInt64, kNumberTypeUInt16, int64_t, uint16_t)},
  {TRACEV2_CPU_REG(kNumberTypeInt64, kNumberTypeUInt32, int64_t, uint32_t)},
  {TRACEV2_CPU_REG(kNumberTypeInt64, kNumberTypeUInt64, int64_t, uint64_t)},
  {TRACEV2_CPU_REG(kNumberTypeInt64, kNumberTypeInt8, int64_t, int8_t)},
  {TRACEV2_CPU_REG(kNumberTypeInt64, kNumberTypeInt16, int64_t, int16_t)},
  {TRACEV2_CPU_REG(kNumberTypeInt64, kNumberTypeInt32, int64_t, int32_t)},
  {TRACEV2_CPU_REG(kNumberTypeInt64, kNumberTypeInt64, int64_t, int64_t)},
  {TRACEV2_CPU_REG(kNumberTypeInt64, kNumberTypeFloat16, int64_t, float16)},
  {TRACEV2_CPU_REG(kNumberTypeInt64, kNumberTypeFloat32, int64_t, float)},
  {TRACEV2_CPU_REG(kNumberTypeInt64, kNumberTypeFloat64, int64_t, double)},
  {TRACEV2_CPU_REG(kNumberTypeInt64, kNumberTypeComplex64, int64_t, complex64)},
  {TRACEV2_CPU_REG(kNumberTypeInt64, kNumberTypeComplex128, int64_t, complex128)},
  {TRACEV2_CPU_REG(kNumberTypeInt64, kNumberTypeBool, int64_t, bool)},
  {TRACEV2_CPU_REG(kNumberTypeFloat16, kNumberTypeUInt8, float16, uint8_t)},
  {TRACEV2_CPU_REG(kNumberTypeFloat16, kNumberTypeUInt16, float16, uint16_t)},
  {TRACEV2_CPU_REG(kNumberTypeFloat16, kNumberTypeUInt32, float16, uint32_t)},
  {TRACEV2_CPU_REG(kNumberTypeFloat16, kNumberTypeUInt64, float16, uint64_t)},
  {TRACEV2_CPU_REG(kNumberTypeFloat16, kNumberTypeInt8, float16, int8_t)},
  {TRACEV2_CPU_REG(kNumberTypeFloat16, kNumberTypeInt16, float16, int16_t)},
  {TRACEV2_CPU_REG(kNumberTypeFloat16, kNumberTypeInt32, float16, int32_t)},
  {TRACEV2_CPU_REG(kNumberTypeFloat16, kNumberTypeInt64, float16, int64_t)},
  {TRACEV2_CPU_REG(kNumberTypeFloat16, kNumberTypeFloat16, float16, float16)},
  {TRACEV2_CPU_REG(kNumberTypeFloat16, kNumberTypeFloat32, float16, float)},
  {TRACEV2_CPU_REG(kNumberTypeFloat16, kNumberTypeFloat64, float16, double)},
  {TRACEV2_CPU_REG(kNumberTypeFloat16, kNumberTypeComplex64, float16, complex64)},
  {TRACEV2_CPU_REG(kNumberTypeFloat16, kNumberTypeComplex128, float16, complex128)},
  {TRACEV2_CPU_REG(kNumberTypeFloat16, kNumberTypeBool, float16, bool)},
  {TRACEV2_CPU_REG(kNumberTypeFloat32, kNumberTypeUInt8, float, uint8_t)},
  {TRACEV2_CPU_REG(kNumberTypeFloat32, kNumberTypeUInt16, float, uint16_t)},
  {TRACEV2_CPU_REG(kNumberTypeFloat32, kNumberTypeUInt32, float, uint32_t)},
  {TRACEV2_CPU_REG(kNumberTypeFloat32, kNumberTypeUInt64, float, uint64_t)},
  {TRACEV2_CPU_REG(kNumberTypeFloat32, kNumberTypeInt8, float, int8_t)},
  {TRACEV2_CPU_REG(kNumberTypeFloat32, kNumberTypeInt16, float, int16_t)},
  {TRACEV2_CPU_REG(kNumberTypeFloat32, kNumberTypeInt32, float, int32_t)},
  {TRACEV2_CPU_REG(kNumberTypeFloat32, kNumberTypeInt64, float, int64_t)},
  {TRACEV2_CPU_REG(kNumberTypeFloat32, kNumberTypeFloat16, float, float16)},
  {TRACEV2_CPU_REG(kNumberTypeFloat32, kNumberTypeFloat32, float, float)},
  {TRACEV2_CPU_REG(kNumberTypeFloat32, kNumberTypeFloat64, float, double)},
  {TRACEV2_CPU_REG(kNumberTypeFloat32, kNumberTypeComplex64, float, complex64)},
  {TRACEV2_CPU_REG(kNumberTypeFloat32, kNumberTypeComplex128, float, complex128)},
  {TRACEV2_CPU_REG(kNumberTypeFloat32, kNumberTypeBool, float, bool)},
  {TRACEV2_CPU_REG(kNumberTypeFloat64, kNumberTypeUInt8, double, uint8_t)},
  {TRACEV2_CPU_REG(kNumberTypeFloat64, kNumberTypeUInt16, double, uint16_t)},
  {TRACEV2_CPU_REG(kNumberTypeFloat64, kNumberTypeUInt32, double, uint32_t)},
  {TRACEV2_CPU_REG(kNumberTypeFloat64, kNumberTypeUInt64, double, uint64_t)},
  {TRACEV2_CPU_REG(kNumberTypeFloat64, kNumberTypeInt8, double, int8_t)},
  {TRACEV2_CPU_REG(kNumberTypeFloat64, kNumberTypeInt16, double, int16_t)},
  {TRACEV2_CPU_REG(kNumberTypeFloat64, kNumberTypeInt32, double, int32_t)},
  {TRACEV2_CPU_REG(kNumberTypeFloat64, kNumberTypeInt64, double, int64_t)},
  {TRACEV2_CPU_REG(kNumberTypeFloat64, kNumberTypeFloat16, double, float16)},
  {TRACEV2_CPU_REG(kNumberTypeFloat64, kNumberTypeFloat32, double, float)},
  {TRACEV2_CPU_REG(kNumberTypeFloat64, kNumberTypeFloat64, double, double)},
  {TRACEV2_CPU_REG(kNumberTypeFloat64, kNumberTypeComplex64, double, complex64)},
  {TRACEV2_CPU_REG(kNumberTypeFloat64, kNumberTypeComplex128, double, complex128)},
  {TRACEV2_CPU_REG(kNumberTypeFloat64, kNumberTypeBool, double, bool)},
  {TRACEV2_CPU_REG(kNumberTypeComplex64, kNumberTypeUInt8, complex64, uint8_t)},
  {TRACEV2_CPU_REG(kNumberTypeComplex64, kNumberTypeUInt16, complex64, uint16_t)},
  {TRACEV2_CPU_REG(kNumberTypeComplex64, kNumberTypeUInt32, complex64, uint32_t)},
  {TRACEV2_CPU_REG(kNumberTypeComplex64, kNumberTypeUInt64, complex64, uint64_t)},
  {TRACEV2_CPU_REG(kNumberTypeComplex64, kNumberTypeInt8, complex64, int8_t)},
  {TRACEV2_CPU_REG(kNumberTypeComplex64, kNumberTypeInt16, complex64, int16_t)},
  {TRACEV2_CPU_REG(kNumberTypeComplex64, kNumberTypeInt32, complex64, int32_t)},
  {TRACEV2_CPU_REG(kNumberTypeComplex64, kNumberTypeInt64, complex64, int64_t)},
  {TRACEV2_CPU_REG(kNumberTypeComplex64, kNumberTypeFloat16, complex64, float16)},
  {TRACEV2_CPU_REG(kNumberTypeComplex64, kNumberTypeFloat32, complex64, float)},
  {TRACEV2_CPU_REG(kNumberTypeComplex64, kNumberTypeFloat64, complex64, double)},
  {TRACEV2_CPU_REG(kNumberTypeComplex64, kNumberTypeComplex64, complex64, complex64)},
  {TRACEV2_CPU_REG(kNumberTypeComplex64, kNumberTypeComplex128, complex64, complex128)},
  {TRACEV2_CPU_REG(kNumberTypeComplex64, kNumberTypeBool, complex64, bool)},
  {TRACEV2_CPU_REG(kNumberTypeComplex128, kNumberTypeUInt8, complex128, uint8_t)},
  {TRACEV2_CPU_REG(kNumberTypeComplex128, kNumberTypeUInt16, complex128, uint16_t)},
  {TRACEV2_CPU_REG(kNumberTypeComplex128, kNumberTypeUInt32, complex128, uint32_t)},
  {TRACEV2_CPU_REG(kNumberTypeComplex128, kNumberTypeUInt64, complex128, uint64_t)},
  {TRACEV2_CPU_REG(kNumberTypeComplex128, kNumberTypeInt8, complex128, int8_t)},
  {TRACEV2_CPU_REG(kNumberTypeComplex128, kNumberTypeInt16, complex128, int16_t)},
  {TRACEV2_CPU_REG(kNumberTypeComplex128, kNumberTypeInt32, complex128, int32_t)},
  {TRACEV2_CPU_REG(kNumberTypeComplex128, kNumberTypeInt64, complex128, int64_t)},
  {TRACEV2_CPU_REG(kNumberTypeComplex128, kNumberTypeFloat16, complex128, float16)},
  {TRACEV2_CPU_REG(kNumberTypeComplex128, kNumberTypeFloat32, complex128, float)},
  {TRACEV2_CPU_REG(kNumberTypeComplex128, kNumberTypeFloat64, complex128, double)},
  {TRACEV2_CPU_REG(kNumberTypeComplex128, kNumberTypeComplex64, complex128, complex64)},
  {TRACEV2_CPU_REG(kNumberTypeComplex128, kNumberTypeComplex128, complex128, complex128)},
  {TRACEV2_CPU_REG(kNumberTypeComplex128, kNumberTypeBool, complex128, bool)},
  {TRACEV2_CPU_REG(kNumberTypeBool, kNumberTypeUInt8, bool, uint8_t)},
  {TRACEV2_CPU_REG(kNumberTypeBool, kNumberTypeUInt16, bool, uint16_t)},
  {TRACEV2_CPU_REG(kNumberTypeBool, kNumberTypeUInt32, bool, uint32_t)},
  {TRACEV2_CPU_REG(kNumberTypeBool, kNumberTypeUInt64, bool, uint64_t)},
  {TRACEV2_CPU_REG(kNumberTypeBool, kNumberTypeInt8, bool, int8_t)},
  {TRACEV2_CPU_REG(kNumberTypeBool, kNumberTypeInt16, bool, int16_t)},
  {TRACEV2_CPU_REG(kNumberTypeBool, kNumberTypeInt32, bool, int32_t)},
  {TRACEV2_CPU_REG(kNumberTypeBool, kNumberTypeInt64, bool, int64_t)},
  {TRACEV2_CPU_REG(kNumberTypeBool, kNumberTypeFloat16, bool, float16)},
  {TRACEV2_CPU_REG(kNumberTypeBool, kNumberTypeFloat32, bool, float)},
  {TRACEV2_CPU_REG(kNumberTypeBool, kNumberTypeFloat64, bool, double)},
  {TRACEV2_CPU_REG(kNumberTypeBool, kNumberTypeComplex64, bool, complex64)},
  {TRACEV2_CPU_REG(kNumberTypeBool, kNumberTypeComplex128, bool, complex128)},
  {TRACEV2_CPU_REG(kNumberTypeBool, kNumberTypeBool, bool, bool)},
};

std::vector<KernelAttr> TraceV2CpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, TraceV2Func> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, TraceV2, TraceV2CpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
