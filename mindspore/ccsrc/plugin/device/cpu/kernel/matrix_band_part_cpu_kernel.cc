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
#include "mindspore/core/ops/matrix_band_part.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kMaxDims = 8;
constexpr size_t kXMinShapeSize = 2;
using KernelRunFunc = MatrixBandPartCpuKernelMod::KernelRunFunc;
}  // namespace
bool MatrixBandPartCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                      const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it got empty inputs or outputs, which is invalid.";
    return false;
  }
  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
    return false;
  }
  return true;
}

void MatrixBandPartCpuKernelMod::BroadcastShape(const std::vector<size_t> &x_shape,
                                                const std::vector<size_t> &lower_shape,
                                                const std::vector<size_t> &upper_shape,
                                                const std::vector<size_t> &output_shape) {
  broadcast_x_shape_.clear();
  broadcast_lower_shape_.clear();
  broadcast_upper_shape_.clear();
  broadcast_output_shape_.clear();
  broadcast_x_shape_.resize(kMaxDims, 1);
  broadcast_lower_shape_.resize(kMaxDims, 1);
  broadcast_upper_shape_.resize(kMaxDims, 1);
  broadcast_output_shape_.resize(kMaxDims, 1);
  auto expanded_lower_shape = ops::GetExpandedShape<size_t>(lower_shape);
  auto expanded_upper_shape = ops::GetExpandedShape<size_t>(upper_shape);

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
  auto x_shape_temp = inputs.at(kIndex0)->GetShapeVector();
  auto lower_shape_temp = inputs.at(kIndex1)->GetShapeVector();
  auto upper_shape_temp = inputs.at(kIndex2)->GetShapeVector();
  auto output_shape_temp = outputs.at(kIndex0)->GetShapeVector();
  std::vector<size_t> x_shape{};
  std::vector<size_t> lower_shape{};
  std::vector<size_t> upper_shape{};
  std::vector<size_t> output_shape{};
  (void)std::transform(x_shape_temp.begin(), x_shape_temp.end(), std::back_inserter(x_shape), LongToSize);
  (void)std::transform(lower_shape_temp.begin(), lower_shape_temp.end(), std::back_inserter(lower_shape), LongToSize);
  (void)std::transform(upper_shape_temp.begin(), upper_shape_temp.end(), std::back_inserter(upper_shape), LongToSize);
  (void)std::transform(output_shape_temp.begin(), output_shape_temp.end(), std::back_inserter(output_shape),
                       LongToSize);
  size_t input_element_num = std::accumulate(x_shape.begin(), x_shape.end(), 1, std::multiplies<size_t>());
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
  m_ = x_shape[dim_size_ - kDim2];
  n_ = x_shape[dim_size_ - kDim1];
  if (m_ == 0 || n_ == 0) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the size of -2 axis or -1 axis can not be 0, "
                  << "but got m_=" << m_ << ", n_=" << n_;
    return KRET_RESIZE_FAILED;
  }
  output_outer_size_ = 1;
  for (size_t i = 0; i < output_shape.size() - kDim2; i++) {
    output_outer_size_ *= output_shape[i];
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
  auto ret_s1 = memset_s(output_ptr, output_element_num_ * sizeof(T), 0, output_element_num_ * sizeof(T));
  if (ret_s1 != EOK) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', it's memset failed. Error no: " << ret_s1;
  }
  bool is_diagonal = (lower_ == 0 && upper_ == 0);
  // The non_zero_len is the length of the non zero element along the -2 axis, so it can skip the position with 0.
  size_t non_zero_len = std::min(m_, lower_ + n_);
  int errno_t = EOK;
  auto task = [this, &errno_t, is_diagonal, non_zero_len, x_ptr, output_ptr](size_t start, size_t end) {
    for (size_t t = start; t < end; t++) {
      // The non_zero_len can not be 0.
      const auto i = t / non_zero_len;
      const auto j = t % non_zero_len;
      const auto offset = i * m_ * n_ + j * n_;
      if (is_diagonal) {
        output_ptr[offset + j] = x_ptr[offset + j];
      } else {
        const auto s = (j < lower_ ? 0 : j - lower_);
        // When j + upper_ >= n_, the e is n - 1.
        const auto e = (j >= n_ - upper_ ? n_ - 1 : j + upper_);
        auto temp_errno_t = memcpy_s(output_ptr + offset + s, output_element_num_ * sizeof(T), x_ptr + offset + s,
                                     (e - s + 1) * sizeof(T));
        if (temp_errno_t != EOK) {
          // In multi-thread, it can not throw exception.
          errno_t = temp_errno_t;
          break;
        }
      }
    }
  };
  ParallelLaunchAutoSearch(task, output_outer_size_ * non_zero_len, this, &parallel_search_info_, pool_);
  if (errno_t != EOK) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', it's memcpy failed. Error no: " << errno_t;
  }
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
        output_ptr[i] = 0;
      }
      iter.GenNextPos();
    }
  };
  ParallelLaunchAutoSearch(task, output_element_num_, this, &parallel_search_info_, pool_);
  return true;
}

template <typename T, typename LU>
bool MatrixBandPartCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                              const std::vector<AddressPtr> &,
                                              const std::vector<kernel::AddressPtr> &outputs) {
  const auto x_ptr = reinterpret_cast<T *>(inputs[0]->addr);
  // Both the lower and upper have done the type check in C++ primitive.
  const auto lower_ptr = reinterpret_cast<LU *>(inputs[1]->addr);
  const auto upper_ptr = reinterpret_cast<LU *>(inputs[2]->addr);
  auto output_ptr = reinterpret_cast<T *>(outputs[0]->addr);

  if (need_broadcast_) {
    LaunchKernelBroadcast(x_ptr, lower_ptr, upper_ptr, output_ptr);
    return true;
  } else {
    return LaunchKernelNotBroadcast(x_ptr, lower_ptr, upper_ptr, output_ptr);
  }
}

const std::vector<std::pair<KernelAttr, KernelRunFunc>> &MatrixBandPartCpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, KernelRunFunc>> func_list = {
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
  };
  return func_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, MatrixBandPart, MatrixBandPartCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
