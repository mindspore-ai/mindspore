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
#include "utils/ms_utils.h"

namespace mindspore {
namespace kernel {
bool MatrixBandPartCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                      const std::vector<KernelTensorPtr> &outputs) {
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

int MatrixBandPartCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                       const std::vector<KernelTensorPtr> &outputs,
                                       const std::map<uint32_t, tensor::TensorPtr> &) {
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  shapes_.clear();
  auto input_shape = inputs.at(kIndex0)->GetShapeVector();
  (void)std::transform(input_shape.begin(), input_shape.end(), std::back_inserter(shapes_), LongToSize);
  size_t input_element_num = std::accumulate(shapes_.begin(), shapes_.end(), 1, std::multiplies<size_t>());
  is_null_input_ = (input_element_num == 0);
  if (is_null_input_) {
    return KRET_OK;
  }

  dim_size_ = shapes_.size();
  if (shapes_.size() < kDim2) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it's input dims must be a matrix greater than or equal to 2D, "
                  << "but got " << shapes_.size() << "D.";
    return KRET_RESIZE_FAILED;
  }
  m_ = shapes_[dim_size_ - kDim2];
  n_ = shapes_[dim_size_ - kDim1];
  if (m_ == 0 || n_ == 0) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the size of -2 axis or -1 axis can not be 0, "
                  << "but got m_=" << m_ << ", n_=" << n_;
    return KRET_RESIZE_FAILED;
  }
  output_outer_size_ = 1;
  for (size_t i = 0; i < shapes_.size() - kDim2; i++) {
    output_outer_size_ *= shapes_[i];
  }
  output_element_num_ = output_outer_size_ * m_ * n_;
  return KRET_OK;
}

template <typename T, typename LU>
bool MatrixBandPartCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                              const std::vector<kernel::AddressPtr> &outputs) {
  T *input_ptr = reinterpret_cast<T *>(inputs[0]->addr);
  // Both the lower and upper have done the type check in C++ primitive.
  const auto lower = reinterpret_cast<LU *>(inputs[1]->addr)[0];
  const auto upper = reinterpret_cast<LU *>(inputs[2]->addr)[0];
  T *output_ptr = reinterpret_cast<T *>(outputs[0]->addr);

  lower_ = (lower < 0 || lower > SizeToLong(m_)) ? m_ : LongToSize(lower);
  upper_ = (upper < 0 || upper > SizeToLong(n_)) ? n_ : LongToSize(upper);
  if (lower_ >= m_ && upper_ >= n_) {
    auto ret_s2 = memcpy_s(output_ptr, output_element_num_ * sizeof(T), input_ptr, output_element_num_ * sizeof(T));
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
  auto task = [this, &errno_t, is_diagonal, non_zero_len, input_ptr, output_ptr](size_t start, size_t end) {
    for (size_t t = start; t < end; t++) {
      // The non_zero_len can not be 0.
      const auto i = t / non_zero_len;
      const auto j = t % non_zero_len;
      const auto offset = i * m_ * n_ + j * n_;
      if (is_diagonal) {
        output_ptr[offset + j] = input_ptr[offset + j];
      } else {
        const auto s = (j < lower_ ? 0 : j - lower_);
        // When j + upper_ >= n_, the e is n - 1.
        const auto e = (j >= n_ - upper_ ? n_ - 1 : j + upper_);
        auto temp_errno_t = memcpy_s(output_ptr + offset + s, output_element_num_ * sizeof(T), input_ptr + offset + s,
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

std::vector<std::pair<KernelAttr, MatrixBandPartCpuKernelMod::MatrixBandPartFunc>>
  MatrixBandPartCpuKernelMod::func_list_ = {{KernelAttr()
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
                                             &MatrixBandPartCpuKernelMod::LaunchKernel<double, int64_t>}};

std::vector<KernelAttr> MatrixBandPartCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, MatrixBandPartFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, MatrixBandPart, MatrixBandPartCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
