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

#include "kernel/gpu/cuda/arrays/matrix_band_part_gpu_kernel.h"
#include <functional>
#include "mindspore/ops/infer/matrix_band_part.h"
#include "kernel/gpu/cuda_impl/cuda_ops/complex.h"

namespace mindspore {
namespace kernel {
template <typename T>
using Complex = mindspore::utils::Complex<T>;
constexpr size_t kMaxDims = 8;
constexpr size_t kXMinShapeSize = 2;

bool MatrixBandPartGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                      const std::vector<KernelTensor *> &outputs) {
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

void MatrixBandPartGpuKernelMod::BroadcastShape(const std::vector<int64_t> &x_shape,
                                                const std::vector<int64_t> &lower_shape,
                                                const std::vector<int64_t> &upper_shape,
                                                const std::vector<int64_t> &output_shape) {
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

int MatrixBandPartGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &outputs) {
  if (int ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }

  auto x_shape_temp = inputs.at(kIndex0)->GetShapeVector();
  auto lower_shape_temp = inputs.at(kIndex1)->GetShapeVector();
  auto upper_shape_temp = inputs.at(kIndex2)->GetShapeVector();
  auto output_shape_temp = outputs.at(kIndex0)->GetShapeVector();
  std::vector<int64_t> x_shape{};
  std::vector<int64_t> lower_shape{};
  std::vector<int64_t> upper_shape{};
  std::vector<int64_t> output_shape{};
  (void)std::transform(x_shape_temp.begin(), x_shape_temp.end(), std::back_inserter(x_shape), LongToSize);
  (void)std::transform(lower_shape_temp.begin(), lower_shape_temp.end(), std::back_inserter(lower_shape), LongToSize);
  (void)std::transform(upper_shape_temp.begin(), upper_shape_temp.end(), std::back_inserter(upper_shape), LongToSize);
  (void)std::transform(output_shape_temp.begin(), output_shape_temp.end(), std::back_inserter(output_shape),
                       LongToSize);
  size_t input_element_num = std::accumulate(x_shape.begin(), x_shape.end(), size_t(1), std::multiplies<size_t>());
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
  for (size_t i = 0; i < output_shape.size() - kXMinShapeSize; i++) {
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
bool MatrixBandPartGpuKernelMod::LaunchKernelNotBroadcast(const T *x_ptr, const LU *lower_ptr, const LU *upper_ptr,
                                                          T *output_ptr) {
  LU lower = 0;
  LU upper = 0;
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMemcpyAsync(&lower, lower_ptr, sizeof(LU), cudaMemcpyDeviceToHost,
                                                     reinterpret_cast<cudaStream_t>(cuda_stream_)),
                                     "For 'MatrixBandPart', copying input lower to host failed.");
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMemcpyAsync(&upper, upper_ptr, sizeof(LU), cudaMemcpyDeviceToHost,
                                                     reinterpret_cast<cudaStream_t>(cuda_stream_)),
                                     "For 'MatrixBandPart', copying input upper to host failed.");
  if (cudaStreamQuery(reinterpret_cast<cudaStream_t>(cuda_stream_)) != cudaSuccess) {
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(cuda_stream_)),
                                       "For 'MatrixBandPart', cuda Stream Sync Failed.");
  }

  lower_ = static_cast<int64_t>(lower);
  upper_ = static_cast<int64_t>(upper);
  lower_ = (lower_ < 0 || lower_ > SizeToLong(m_)) ? SizeToLong(m_) : lower_;
  upper_ = (upper_ < 0 || upper_ > SizeToLong(n_)) ? SizeToLong(n_) : upper_;

  if (lower == 0 && upper == 0) {
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemsetAsync(output_ptr, 0, output_element_num_ * sizeof(T), reinterpret_cast<cudaStream_t>(cuda_stream_)),
      "For 'MatrixBandPart', it's cudaMemsetAsync failed.");
  } else {
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(output_ptr, x_ptr, output_element_num_ * sizeof(T), cudaMemcpyDeviceToDevice,
                      reinterpret_cast<cudaStream_t>(cuda_stream_)),
      "For 'MatrixBandPart', it's cudaMemcpyAsync failed.");
  }

  if (lower_ >= SizeToLong(m_) && upper_ >= SizeToLong(n_)) {
    return true;
  }
  auto status = MatrixBandPart(output_outer_size_, x_ptr, m_, n_, lower_, upper_, output_ptr, device_id_,
                               reinterpret_cast<cudaStream_t>(cuda_stream_));
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

template <typename T, typename LU>
bool MatrixBandPartGpuKernelMod::LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                                              const std::vector<kernel::KernelTensor *> &outputs) {
  const auto x_ptr = reinterpret_cast<T *>(inputs.at(kIndex0)->device_ptr());
  // Both the lower and upper have done the type check in C++ primitive.
  const auto lower_ptr = reinterpret_cast<LU *>(inputs.at(kIndex1)->device_ptr());
  const auto upper_ptr = reinterpret_cast<LU *>(inputs.at(kIndex2)->device_ptr());
  auto output_ptr = reinterpret_cast<T *>(outputs.at(kIndex0)->device_ptr());
  if (need_broadcast_) {
    auto status = MatrixBandPartBroadcast(
      output_element_num_, broadcast_x_shape_, broadcast_lower_shape_, broadcast_upper_shape_, broadcast_output_shape_,
      x_ptr, m_, n_, lower_ptr, upper_ptr, output_ptr, device_id_, reinterpret_cast<cudaStream_t>(cuda_stream_));
    CHECK_CUDA_STATUS(status, kernel_name_);
    return true;
  } else {
    return LaunchKernelNotBroadcast(x_ptr, lower_ptr, upper_ptr, output_ptr);
  }
}

std::vector<std::pair<KernelAttr, MatrixBandPartGpuKernelMod::MatrixBandPartFunc>>
  MatrixBandPartGpuKernelMod::func_list_ = {{KernelAttr()
                                               .AddInputAttr(kNumberTypeInt8)
                                               .AddInputAttr(kNumberTypeInt32)
                                               .AddInputAttr(kNumberTypeInt32)
                                               .AddOutputAttr(kNumberTypeInt8),
                                             &MatrixBandPartGpuKernelMod::LaunchKernel<char, int32_t>},
                                            {KernelAttr()
                                               .AddInputAttr(kNumberTypeInt16)
                                               .AddInputAttr(kNumberTypeInt32)
                                               .AddInputAttr(kNumberTypeInt32)
                                               .AddOutputAttr(kNumberTypeInt16),
                                             &MatrixBandPartGpuKernelMod::LaunchKernel<int16_t, int32_t>},
                                            {KernelAttr()
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
                                               .AddInputAttr(kNumberTypeUInt8)
                                               .AddInputAttr(kNumberTypeInt32)
                                               .AddInputAttr(kNumberTypeInt32)
                                               .AddOutputAttr(kNumberTypeUInt8),
                                             &MatrixBandPartGpuKernelMod::LaunchKernel<uchar, int32_t>},
                                            {KernelAttr()
                                               .AddInputAttr(kNumberTypeUInt16)
                                               .AddInputAttr(kNumberTypeInt32)
                                               .AddInputAttr(kNumberTypeInt32)
                                               .AddOutputAttr(kNumberTypeUInt16),
                                             &MatrixBandPartGpuKernelMod::LaunchKernel<uint16_t, int32_t>},
                                            {KernelAttr()
                                               .AddInputAttr(kNumberTypeUInt32)
                                               .AddInputAttr(kNumberTypeInt32)
                                               .AddInputAttr(kNumberTypeInt32)
                                               .AddOutputAttr(kNumberTypeUInt32),
                                             &MatrixBandPartGpuKernelMod::LaunchKernel<uint32_t, int32_t>},
                                            {KernelAttr()
                                               .AddInputAttr(kNumberTypeUInt64)
                                               .AddInputAttr(kNumberTypeInt32)
                                               .AddInputAttr(kNumberTypeInt32)
                                               .AddOutputAttr(kNumberTypeUInt64),
                                             &MatrixBandPartGpuKernelMod::LaunchKernel<uint64_t, int32_t>},
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
                                               .AddInputAttr(kNumberTypeComplex64)
                                               .AddInputAttr(kNumberTypeInt32)
                                               .AddInputAttr(kNumberTypeInt32)
                                               .AddOutputAttr(kNumberTypeComplex64),
                                             &MatrixBandPartGpuKernelMod::LaunchKernel<Complex<float>, int32_t>},
                                            {KernelAttr()
                                               .AddInputAttr(kNumberTypeComplex128)
                                               .AddInputAttr(kNumberTypeInt32)
                                               .AddInputAttr(kNumberTypeInt32)
                                               .AddOutputAttr(kNumberTypeComplex128),
                                             &MatrixBandPartGpuKernelMod::LaunchKernel<Complex<double>, int32_t>},
                                            {KernelAttr()
                                               .AddInputAttr(kNumberTypeBool)
                                               .AddInputAttr(kNumberTypeInt32)
                                               .AddInputAttr(kNumberTypeInt32)
                                               .AddOutputAttr(kNumberTypeBool),
                                             &MatrixBandPartGpuKernelMod::LaunchKernel<bool, int32_t>},
                                            {KernelAttr()
                                               .AddInputAttr(kNumberTypeInt8)
                                               .AddInputAttr(kNumberTypeInt64)
                                               .AddInputAttr(kNumberTypeInt64)
                                               .AddOutputAttr(kNumberTypeInt8),
                                             &MatrixBandPartGpuKernelMod::LaunchKernel<char, int64_t>},
                                            {KernelAttr()
                                               .AddInputAttr(kNumberTypeInt16)
                                               .AddInputAttr(kNumberTypeInt64)
                                               .AddInputAttr(kNumberTypeInt64)
                                               .AddOutputAttr(kNumberTypeInt16),
                                             &MatrixBandPartGpuKernelMod::LaunchKernel<int16_t, int64_t>},
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
                                               .AddInputAttr(kNumberTypeUInt8)
                                               .AddInputAttr(kNumberTypeInt64)
                                               .AddInputAttr(kNumberTypeInt64)
                                               .AddOutputAttr(kNumberTypeUInt8),
                                             &MatrixBandPartGpuKernelMod::LaunchKernel<uchar, int64_t>},
                                            {KernelAttr()
                                               .AddInputAttr(kNumberTypeUInt16)
                                               .AddInputAttr(kNumberTypeInt64)
                                               .AddInputAttr(kNumberTypeInt64)
                                               .AddOutputAttr(kNumberTypeUInt16),
                                             &MatrixBandPartGpuKernelMod::LaunchKernel<uint16_t, int64_t>},
                                            {KernelAttr()
                                               .AddInputAttr(kNumberTypeUInt32)
                                               .AddInputAttr(kNumberTypeInt64)
                                               .AddInputAttr(kNumberTypeInt64)
                                               .AddOutputAttr(kNumberTypeUInt32),
                                             &MatrixBandPartGpuKernelMod::LaunchKernel<uint32_t, int64_t>},
                                            {KernelAttr()
                                               .AddInputAttr(kNumberTypeUInt64)
                                               .AddInputAttr(kNumberTypeInt64)
                                               .AddInputAttr(kNumberTypeInt64)
                                               .AddOutputAttr(kNumberTypeUInt64),
                                             &MatrixBandPartGpuKernelMod::LaunchKernel<uint64_t, int64_t>},
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
                                             &MatrixBandPartGpuKernelMod::LaunchKernel<double, int64_t>},
                                            {KernelAttr()
                                               .AddInputAttr(kNumberTypeComplex64)
                                               .AddInputAttr(kNumberTypeInt64)
                                               .AddInputAttr(kNumberTypeInt64)
                                               .AddOutputAttr(kNumberTypeComplex64),
                                             &MatrixBandPartGpuKernelMod::LaunchKernel<Complex<float>, int64_t>},
                                            {KernelAttr()
                                               .AddInputAttr(kNumberTypeComplex128)
                                               .AddInputAttr(kNumberTypeInt64)
                                               .AddInputAttr(kNumberTypeInt64)
                                               .AddOutputAttr(kNumberTypeComplex128),
                                             &MatrixBandPartGpuKernelMod::LaunchKernel<Complex<double>, int64_t>},
                                            {KernelAttr()
                                               .AddInputAttr(kNumberTypeBool)
                                               .AddInputAttr(kNumberTypeInt64)
                                               .AddInputAttr(kNumberTypeInt64)
                                               .AddOutputAttr(kNumberTypeBool),
                                             &MatrixBandPartGpuKernelMod::LaunchKernel<bool, int64_t>}};

std::vector<KernelAttr> MatrixBandPartGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, MatrixBandPartFunc> &pair) { return pair.first; });
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, MatrixBandPart, MatrixBandPartGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
