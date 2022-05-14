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

#include "plugin/device/gpu/kernel/arrays/matrix_band_part_gpu_kernel.h"
#include <functional>

namespace mindspore {
namespace kernel {
bool MatrixBandPartGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
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

int MatrixBandPartGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                       const std::vector<KernelTensorPtr> &outputs,
                                       const std::map<uint32_t, tensor::TensorPtr> &) {
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs) != KRET_OK) {
    return ret;
  }

  auto input_shape = inputs.at(kIndex0)->GetShapeVector();
  shapes_.clear();
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
bool MatrixBandPartGpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                              const std::vector<kernel::AddressPtr> &outputs) {
  auto input_ptr = reinterpret_cast<T *>(inputs.at(kIndex0)->addr);
  // Both the lower and upper have done the type check in C++ primitive.
  auto lower_ptr = reinterpret_cast<LU *>(inputs.at(kIndex1)->addr);
  auto upper_ptr = reinterpret_cast<LU *>(inputs.at(kIndex2)->addr);
  auto output_ptr = reinterpret_cast<T *>(outputs.at(kIndex0)->addr);
  LU lower = 0;
  LU upper = 0;
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMemcpyAsync(&lower, lower_ptr, sizeof(LU), cudaMemcpyDeviceToHost,
                                                     reinterpret_cast<cudaStream_t>(cuda_stream_)),
                                     "For 'MatrixBandPart', copying input lower to host failed.");
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMemcpyAsync(&upper, upper_ptr, sizeof(LU), cudaMemcpyDeviceToHost,
                                                     reinterpret_cast<cudaStream_t>(cuda_stream_)),
                                     "For 'MatrixBandPart', copying input upper to host failed.");

  lower_ = static_cast<int64_t>(lower);
  upper_ = static_cast<int64_t>(upper);
  lower_ = (lower_ < 0 || lower_ > SizeToLong(m_)) ? SizeToLong(m_) : lower_;
  upper_ = (upper_ < 0 || upper_ > SizeToLong(n_)) ? SizeToLong(n_) : upper_;
  if (lower_ >= SizeToLong(m_) && upper_ >= SizeToLong(n_)) {
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(output_ptr, input_ptr, output_element_num_ * sizeof(T), cudaMemcpyDeviceToDevice,
                      reinterpret_cast<cudaStream_t>(cuda_stream_)),
      "For 'MatrixBandPart', it's cudaMemcpyAsync failed.");
    return true;
  }
  if (lower_ == 0 && upper_ == 0) {
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemsetAsync(output_ptr, 0, output_element_num_ * sizeof(T), reinterpret_cast<cudaStream_t>(cuda_stream_)),
      "For 'MatrixBandPart', it's cudaMemsetAsync failed.");
  }
  MatrixBandPart(output_outer_size_, input_ptr, m_, n_, lower_, upper_, output_ptr, device_id_,
                 reinterpret_cast<cudaStream_t>(cuda_stream_));
  return true;
}

std::vector<std::pair<KernelAttr, MatrixBandPartGpuKernelMod::MatrixBandPartFunc>>
  MatrixBandPartGpuKernelMod::func_list_ = {{KernelAttr()
                                               .AddInputAttr(kNumberTypeInt32)
                                               .AddInputAttr(kNumberTypeInt32)
                                               .AddInputAttr(kNumberTypeInt32)
                                               .AddOutputAttr(kNumberTypeInt32),
                                             &MatrixBandPartGpuKernelMod::LaunchKernel<int32_t, int32_t>},
                                            {KernelAttr()
                                               .AddInputAttr(kNumberTypeInt64)
                                               .AddInputAttr(kNumberTypeInt32)
                                               .AddInputAttr(kNumberTypeInt32)
                                               .AddOutputAttr(kNumberTypeInt64),
                                             &MatrixBandPartGpuKernelMod::LaunchKernel<int64_t, int32_t>},
                                            {KernelAttr()
                                               .AddInputAttr(kNumberTypeFloat16)
                                               .AddInputAttr(kNumberTypeInt32)
                                               .AddInputAttr(kNumberTypeInt32)
                                               .AddOutputAttr(kNumberTypeFloat16),
                                             &MatrixBandPartGpuKernelMod::LaunchKernel<half, int32_t>},
                                            {KernelAttr()
                                               .AddInputAttr(kNumberTypeFloat32)
                                               .AddInputAttr(kNumberTypeInt32)
                                               .AddInputAttr(kNumberTypeInt32)
                                               .AddOutputAttr(kNumberTypeFloat32),
                                             &MatrixBandPartGpuKernelMod::LaunchKernel<float, int32_t>},
                                            {KernelAttr()
                                               .AddInputAttr(kNumberTypeFloat64)
                                               .AddInputAttr(kNumberTypeInt32)
                                               .AddInputAttr(kNumberTypeInt32)
                                               .AddOutputAttr(kNumberTypeFloat64),
                                             &MatrixBandPartGpuKernelMod::LaunchKernel<double, int32_t>},
                                            {KernelAttr()
                                               .AddInputAttr(kNumberTypeInt32)
                                               .AddInputAttr(kNumberTypeInt64)
                                               .AddInputAttr(kNumberTypeInt64)
                                               .AddOutputAttr(kNumberTypeInt32),
                                             &MatrixBandPartGpuKernelMod::LaunchKernel<int32_t, int64_t>},
                                            {KernelAttr()
                                               .AddInputAttr(kNumberTypeInt64)
                                               .AddInputAttr(kNumberTypeInt64)
                                               .AddInputAttr(kNumberTypeInt64)
                                               .AddOutputAttr(kNumberTypeInt64),
                                             &MatrixBandPartGpuKernelMod::LaunchKernel<int64_t, int64_t>},
                                            {KernelAttr()
                                               .AddInputAttr(kNumberTypeFloat16)
                                               .AddInputAttr(kNumberTypeInt64)
                                               .AddInputAttr(kNumberTypeInt64)
                                               .AddOutputAttr(kNumberTypeFloat16),
                                             &MatrixBandPartGpuKernelMod::LaunchKernel<half, int64_t>},
                                            {KernelAttr()
                                               .AddInputAttr(kNumberTypeFloat32)
                                               .AddInputAttr(kNumberTypeInt64)
                                               .AddInputAttr(kNumberTypeInt64)
                                               .AddOutputAttr(kNumberTypeFloat32),
                                             &MatrixBandPartGpuKernelMod::LaunchKernel<float, int64_t>},
                                            {KernelAttr()
                                               .AddInputAttr(kNumberTypeFloat64)
                                               .AddInputAttr(kNumberTypeInt64)
                                               .AddInputAttr(kNumberTypeInt64)
                                               .AddOutputAttr(kNumberTypeFloat64),
                                             &MatrixBandPartGpuKernelMod::LaunchKernel<double, int64_t>}};

std::vector<KernelAttr> MatrixBandPartGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, MatrixBandPartFunc> &pair) { return pair.first; });
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, MatrixBandPart, MatrixBandPartGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
