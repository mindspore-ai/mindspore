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

#include "plugin/device/cpu/kernel/matrix_band_part_cpu_kernel.h"
#include <algorithm>
#include <memory>
#include <functional>
#include <complex>
#include "mindspore/core/ops/matrix_band_part.h"

namespace mindspore {
namespace kernel {
namespace {
using complex64 = std::complex<float>;
using complex128 = std::complex<double>;
constexpr size_t kMaxDims = 8;
constexpr size_t kXMinShapeSize = 2;
}  // namespace

bool MatrixBandPartCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                      const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it got empty inputs or outputs, which is invalid.";
    return false;
  }
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For 'MatrixBandPart', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

void MatrixBandPartCpuKernelMod::BroadcastShape(const ShapeVector &x_shape, const ShapeVector &lower_shape,
                                                const ShapeVector &upper_shape, const ShapeVector &output_shape) {
  broadcast_x_shape_.clear();
  broadcast_lower_shape_.clear();
  broadcast_upper_shape_.clear();
  broadcast_output_shape_.clear();
  broadcast_x_shape_.resize(kMaxDims, 1);
  broadcast_lower_shape_.resize(kMaxDims, 1);
  broadcast_upper_shape_.resize(kMaxDims, 1);
  broadcast_output_shape_.resize(kMaxDims, 1);
  auto expanded_lower_shape = ops::GetExpandedShape<int64_t>(lower_shape, output_shape.size());
  auto expanded_upper_shape = ops::GetExpandedShape<int64_t>(upper_shape, output_shape.size());

  for (size_t i = 0; i < output_shape.size(); i++) {
    broadcast_output_shape_[i] = output_shape[i];
  }

  for (size_t i = 0; i < x_shape.size() - kXMinShapeSize; i++) {
    broadcast_x_shape_[i] = x_shape[i];
  }
  broadcast_x_shape_[output_shape.size() - 2] = x_shape[x_shape.size() - 2];
  broadcast_x_shape_[output_shape.size() - 1] = x_shape[x_shape.size() - 1];

  for (size_t i = 0; i < expanded_lower_shape.size() - kXMinShapeSize; i++) {
    broadcast_lower_shape_[i] = expanded_lower_shape[i];
  }
  broadcast_lower_shape_[output_shape.size() - 2] = expanded_lower_shape[expanded_lower_shape.size() - 2];
  broadcast_lower_shape_[output_shape.size() - 1] = expanded_lower_shape[expanded_lower_shape.size() - 1];

  for (size_t i = 0; i < expanded_upper_shape.size() - kXMinShapeSize; i++) {
    broadcast_upper_shape_[i] = expanded_upper_shape[i];
  }
  broadcast_upper_shape_[output_shape.size() - 2] = expanded_upper_shape[expanded_upper_shape.size() - 2];
  broadcast_upper_shape_[output_shape.size() - 1] = expanded_upper_shape[expanded_upper_shape.size() - 1];
}

int MatrixBandPartCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                       const std::vector<KernelTensorPtr> &outputs,
                                       const std::map<uint32_t, tensor::TensorPtr> &) {
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }

  auto x_shape = inputs.at(kIndex0)->GetShapeVector();
  auto lower_shape = inputs.at(kIndex1)->GetShapeVector();
  auto upper_shape = inputs.at(kIndex2)->GetShapeVector();
  auto output_shape = outputs.at(kIndex0)->GetShapeVector();

  size_t input_element_num = SizeOf(x_shape);
  is_null_input_ = (input_element_num == 0);
  if (is_null_input_) {
    return KRET_OK;
  }

  dim_size_ = x_shape.size();
  if (x_shape.size() < kDim2) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dims of input x must be greater than or equal to 2D, "
                  << "but got " << x_shape.size() << "D.";
    return KRET_RESIZE_FAILED;
  }
  m_ = LongToSize(x_shape[dim_size_ - kDim2]);
  n_ = LongToSize(x_shape[dim_size_ - kDim1]);
  if (m_ == 0 || n_ == 0) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the size of -2 axis or -1 axis can not be 0, "
                  << "but got m_=" << m_ << ", n_=" << n_;
    return KRET_RESIZE_FAILED;
  }
  output_outer_size_ = 1;
  for (size_t i = 0; i < output_shape.size() - kDim2; i++) {
    output_outer_size_ *= LongToSize(output_shape[i]);
  }
  output_element_num_ = output_outer_size_ * m_ * n_;

  need_broadcast_ = lower_shape.size() > 0 || upper_shape.size() > 0;
  if (need_broadcast_) {
    if (output_shape.size() > kMaxDims) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of broadcast output cannot be greater than "
                        << kMaxDims << ", but got the shape of broadcast output: " << output_shape;
    }
    BroadcastShape(x_shape, lower_shape, upper_shape, output_shape);
  }
  return KRET_OK;
}

template <typename T, typename LU>
bool MatrixBandPartCpuKernelMod::LaunchKernelNotBroadcast(const T *x_ptr, const LU *lower_ptr, const LU *upper_ptr,
                                                          T *output_ptr) {
  const auto lower = lower_ptr[0];
  const auto upper = upper_ptr[0];
  lower_ = (lower < 0 || lower > SizeToLong(m_)) ? m_ : LongToSize(lower);
  upper_ = (upper < 0 || upper > SizeToLong(n_)) ? n_ : LongToSize(upper);
  if (lower_ >= m_ && upper_ >= n_) {
    auto ret_s2 = memcpy_s(output_ptr, output_element_num_ * sizeof(T), x_ptr, output_element_num_ * sizeof(T));
    if (ret_s2 != EOK) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', it's memcpy failed. Error no: " << ret_s2;
    }
    return true;
  }
  auto task = [this, x_ptr, output_ptr](size_t start, size_t end) {
    const int start_int = static_cast<int>(start);
    const int end_int = static_cast<int>(end);
    const int m_int = static_cast<int>(m_);
    const int n_int = static_cast<int>(n_);
    const int lower_int = static_cast<int>(lower_);
    const int upper_int = static_cast<int>(upper_);
    for (int t = start_int; t < end_int; t++) {
      const int i = t / m_int;
      const int j = t % m_int;
      const int offset = i * m_int * n_int + j * n_int;
      const int s = (j < lower_int ? 0 : j - lower_int);
      // When j + upper_ >= n_, the e is n - 1.
      const int e = (j >= n_int - upper_int ? n_int - 1 : j + upper_int);
      const int zero_end = std::min(s, n_int);
      for (int zero_i = offset; zero_i < offset + zero_end; ++zero_i) {
        output_ptr[zero_i] = static_cast<T>(0.0);
      }
      for (int cpy_i = offset + s; cpy_i < offset + e + 1; ++cpy_i) {
        output_ptr[cpy_i] = x_ptr[cpy_i];
      }
      for (int zero_i = offset + e + 1; zero_i < offset + n_int; ++zero_i) {
        output_ptr[zero_i] = static_cast<T>(0.0);
      }
    }
  };
  constexpr float min_column_size = 8;
  ParallelLaunch(task, output_outer_size_ * m_, min_column_size, this, pool_);
  return true;
}

template <typename T, typename LU>
bool MatrixBandPartCpuKernelMod::LaunchKernelBroadcast(const T *x_ptr, const LU *lower_ptr, const LU *upper_ptr,
                                                       T *output_ptr) {
  MultipleBroadcastIterator multi_broadcast_iterator(
    {broadcast_x_shape_, broadcast_lower_shape_, broadcast_upper_shape_}, broadcast_output_shape_);
  auto task = [this, x_ptr, lower_ptr, upper_ptr, output_ptr, &multi_broadcast_iterator](size_t start, size_t end) {
    auto iter = multi_broadcast_iterator;
    iter.SetPos(start);
    for (size_t i = start; i < end; i++) {
      const size_t last_two_dim_offset = i % (m_ * n_);
      int64_t ii = static_cast<int64_t>(last_two_dim_offset / n_);
      int64_t jj = static_cast<int64_t>(last_two_dim_offset % n_);
      T x_value = x_ptr[iter.GetInputPos(kIndex0)];
      LU lower = lower_ptr[iter.GetInputPos(kIndex1)];
      LU upper = upper_ptr[iter.GetInputPos(kIndex2)];
      // Note: the type of ii or jj can not be size_t.
      if ((lower < 0 || (ii - jj) <= lower) && (upper < 0 || (jj - ii) <= upper)) {
        output_ptr[i] = x_value;
      } else {
        output_ptr[i] = static_cast<T>(0.0);
      }
      iter.GenNextPos();
    }
  };
  ParallelLaunchAutoSearch(task, output_element_num_, this, &parallel_search_info_, pool_);
  return true;
}

template <typename T, typename LU>
bool MatrixBandPartCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                              const std::vector<kernel::AddressPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(inputs[kIndex0]);
  MS_EXCEPTION_IF_NULL(inputs[kIndex1]);
  MS_EXCEPTION_IF_NULL(inputs[kIndex2]);
  MS_EXCEPTION_IF_NULL(outputs[kIndex0]);
  const auto x_ptr = reinterpret_cast<T *>(inputs[kIndex0]->addr);
  // Both the lower and upper have done the type check in C++ primitive.
  const auto lower_ptr = reinterpret_cast<LU *>(inputs[kIndex1]->addr);
  const auto upper_ptr = reinterpret_cast<LU *>(inputs[kIndex2]->addr);
  auto output_ptr = reinterpret_cast<T *>(outputs[kIndex0]->addr);

  if (need_broadcast_) {
    return LaunchKernelBroadcast(x_ptr, lower_ptr, upper_ptr, output_ptr);
  } else {
    return LaunchKernelNotBroadcast(x_ptr, lower_ptr, upper_ptr, output_ptr);
  }
}

std::vector<std::pair<KernelAttr, MatrixBandPartCpuKernelMod::MatrixBandPartFunc>>
  MatrixBandPartCpuKernelMod::func_list_ = {{KernelAttr()
                                               .AddInputAttr(kNumberTypeInt8)
                                               .AddInputAttr(kNumberTypeInt32)
                                               .AddInputAttr(kNumberTypeInt32)
                                               .AddOutputAttr(kNumberTypeInt8),
                                             &MatrixBandPartCpuKernelMod::LaunchKernel<int8_t, int32_t>},
                                            {KernelAttr()
                                               .AddInputAttr(kNumberTypeInt16)
                                               .AddInputAttr(kNumberTypeInt32)
                                               .AddInputAttr(kNumberTypeInt32)
                                               .AddOutputAttr(kNumberTypeInt16),
                                             &MatrixBandPartCpuKernelMod::LaunchKernel<int16_t, int32_t>},
                                            {KernelAttr()
                                               .AddInputAttr(kNumberTypeInt32)
                                               .AddInputAttr(kNumberTypeInt32)
                                               .AddInputAttr(kNumberTypeInt32)
                                               .AddOutputAttr(kNumberTypeInt32),
                                             &MatrixBandPartCpuKernelMod::LaunchKernel<int32_t, int32_t>},
                                            {KernelAttr()
                                               .AddInputAttr(kNumberTypeInt64)
                                               .AddInputAttr(kNumberTypeInt32)
                                               .AddInputAttr(kNumberTypeInt32)
                                               .AddOutputAttr(kNumberTypeInt64),
                                             &MatrixBandPartCpuKernelMod::LaunchKernel<int64_t, int32_t>},
                                            {KernelAttr()
                                               .AddInputAttr(kNumberTypeUInt8)
                                               .AddInputAttr(kNumberTypeInt32)
                                               .AddInputAttr(kNumberTypeInt32)
                                               .AddOutputAttr(kNumberTypeUInt8),
                                             &MatrixBandPartCpuKernelMod::LaunchKernel<uint8_t, int32_t>},
                                            {KernelAttr()
                                               .AddInputAttr(kNumberTypeUInt16)
                                               .AddInputAttr(kNumberTypeInt32)
                                               .AddInputAttr(kNumberTypeInt32)
                                               .AddOutputAttr(kNumberTypeUInt16),
                                             &MatrixBandPartCpuKernelMod::LaunchKernel<uint16_t, int32_t>},
                                            {KernelAttr()
                                               .AddInputAttr(kNumberTypeUInt32)
                                               .AddInputAttr(kNumberTypeInt32)
                                               .AddInputAttr(kNumberTypeInt32)
                                               .AddOutputAttr(kNumberTypeUInt32),
                                             &MatrixBandPartCpuKernelMod::LaunchKernel<uint32_t, int32_t>},
                                            {KernelAttr()
                                               .AddInputAttr(kNumberTypeUInt64)
                                               .AddInputAttr(kNumberTypeInt32)
                                               .AddInputAttr(kNumberTypeInt32)
                                               .AddOutputAttr(kNumberTypeUInt64),
                                             &MatrixBandPartCpuKernelMod::LaunchKernel<uint64_t, int32_t>},
                                            {KernelAttr()
                                               .AddInputAttr(kNumberTypeFloat32)
                                               .AddInputAttr(kNumberTypeInt32)
                                               .AddInputAttr(kNumberTypeInt32)
                                               .AddOutputAttr(kNumberTypeFloat32),
                                             &MatrixBandPartCpuKernelMod::LaunchKernel<float, int32_t>},
                                            {KernelAttr()
                                               .AddInputAttr(kNumberTypeFloat64)
                                               .AddInputAttr(kNumberTypeInt32)
                                               .AddInputAttr(kNumberTypeInt32)
                                               .AddOutputAttr(kNumberTypeFloat64),
                                             &MatrixBandPartCpuKernelMod::LaunchKernel<double, int32_t>},
                                            {KernelAttr()
                                               .AddInputAttr(kNumberTypeComplex64)
                                               .AddInputAttr(kNumberTypeInt32)
                                               .AddInputAttr(kNumberTypeInt32)
                                               .AddOutputAttr(kNumberTypeComplex64),
                                             &MatrixBandPartCpuKernelMod::LaunchKernel<complex64, int32_t>},
                                            {KernelAttr()
                                               .AddInputAttr(kNumberTypeComplex128)
                                               .AddInputAttr(kNumberTypeInt32)
                                               .AddInputAttr(kNumberTypeInt32)
                                               .AddOutputAttr(kNumberTypeComplex128),
                                             &MatrixBandPartCpuKernelMod::LaunchKernel<complex128, int32_t>},
                                            {KernelAttr()
                                               .AddInputAttr(kNumberTypeBool)
                                               .AddInputAttr(kNumberTypeInt32)
                                               .AddInputAttr(kNumberTypeInt32)
                                               .AddOutputAttr(kNumberTypeBool),
                                             &MatrixBandPartCpuKernelMod::LaunchKernel<bool, int32_t>},
                                            {KernelAttr()
                                               .AddInputAttr(kNumberTypeInt8)
                                               .AddInputAttr(kNumberTypeInt64)
                                               .AddInputAttr(kNumberTypeInt64)
                                               .AddOutputAttr(kNumberTypeInt8),
                                             &MatrixBandPartCpuKernelMod::LaunchKernel<int8_t, int64_t>},
                                            {KernelAttr()
                                               .AddInputAttr(kNumberTypeInt16)
                                               .AddInputAttr(kNumberTypeInt64)
                                               .AddInputAttr(kNumberTypeInt64)
                                               .AddOutputAttr(kNumberTypeInt16),
                                             &MatrixBandPartCpuKernelMod::LaunchKernel<int16_t, int64_t>},
                                            {KernelAttr()
                                               .AddInputAttr(kNumberTypeInt32)
                                               .AddInputAttr(kNumberTypeInt64)
                                               .AddInputAttr(kNumberTypeInt64)
                                               .AddOutputAttr(kNumberTypeInt32),
                                             &MatrixBandPartCpuKernelMod::LaunchKernel<int32_t, int64_t>},
                                            {KernelAttr()
                                               .AddInputAttr(kNumberTypeInt64)
                                               .AddInputAttr(kNumberTypeInt64)
                                               .AddInputAttr(kNumberTypeInt64)
                                               .AddOutputAttr(kNumberTypeInt64),
                                             &MatrixBandPartCpuKernelMod::LaunchKernel<int64_t, int64_t>},
                                            {KernelAttr()
                                               .AddInputAttr(kNumberTypeUInt8)
                                               .AddInputAttr(kNumberTypeInt64)
                                               .AddInputAttr(kNumberTypeInt64)
                                               .AddOutputAttr(kNumberTypeUInt8),
                                             &MatrixBandPartCpuKernelMod::LaunchKernel<uint8_t, int64_t>},
                                            {KernelAttr()
                                               .AddInputAttr(kNumberTypeUInt16)
                                               .AddInputAttr(kNumberTypeInt64)
                                               .AddInputAttr(kNumberTypeInt64)
                                               .AddOutputAttr(kNumberTypeUInt16),
                                             &MatrixBandPartCpuKernelMod::LaunchKernel<uint16_t, int64_t>},
                                            {KernelAttr()
                                               .AddInputAttr(kNumberTypeUInt32)
                                               .AddInputAttr(kNumberTypeInt64)
                                               .AddInputAttr(kNumberTypeInt64)
                                               .AddOutputAttr(kNumberTypeUInt32),
                                             &MatrixBandPartCpuKernelMod::LaunchKernel<uint32_t, int64_t>},
                                            {KernelAttr()
                                               .AddInputAttr(kNumberTypeUInt64)
                                               .AddInputAttr(kNumberTypeInt64)
                                               .AddInputAttr(kNumberTypeInt64)
                                               .AddOutputAttr(kNumberTypeUInt64),
                                             &MatrixBandPartCpuKernelMod::LaunchKernel<uint64_t, int64_t>},
                                            {KernelAttr()
                                               .AddInputAttr(kNumberTypeFloat32)
                                               .AddInputAttr(kNumberTypeInt64)
                                               .AddInputAttr(kNumberTypeInt64)
                                               .AddOutputAttr(kNumberTypeFloat32),
                                             &MatrixBandPartCpuKernelMod::LaunchKernel<float, int64_t>},
                                            {KernelAttr()
                                               .AddInputAttr(kNumberTypeFloat64)
                                               .AddInputAttr(kNumberTypeInt64)
                                               .AddInputAttr(kNumberTypeInt64)
                                               .AddOutputAttr(kNumberTypeFloat64),
                                             &MatrixBandPartCpuKernelMod::LaunchKernel<double, int64_t>},
                                            {KernelAttr()
                                               .AddInputAttr(kNumberTypeComplex64)
                                               .AddInputAttr(kNumberTypeInt64)
                                               .AddInputAttr(kNumberTypeInt64)
                                               .AddOutputAttr(kNumberTypeComplex64),
                                             &MatrixBandPartCpuKernelMod::LaunchKernel<complex64, int64_t>},
                                            {KernelAttr()
                                               .AddInputAttr(kNumberTypeComplex128)
                                               .AddInputAttr(kNumberTypeInt64)
                                               .AddInputAttr(kNumberTypeInt64)
                                               .AddOutputAttr(kNumberTypeComplex128),
                                             &MatrixBandPartCpuKernelMod::LaunchKernel<complex128, int64_t>},
                                            {KernelAttr()
                                               .AddInputAttr(kNumberTypeBool)
                                               .AddInputAttr(kNumberTypeInt64)
                                               .AddInputAttr(kNumberTypeInt64)
                                               .AddOutputAttr(kNumberTypeBool),
                                             &MatrixBandPartCpuKernelMod::LaunchKernel<bool, int64_t>}};

std::vector<KernelAttr> MatrixBandPartCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, MatrixBandPartFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, MatrixBandPart, MatrixBandPartCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
