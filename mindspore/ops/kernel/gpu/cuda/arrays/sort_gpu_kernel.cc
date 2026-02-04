/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "kernel/gpu/cuda/arrays/sort_gpu_kernel.h"
#include <map>

namespace mindspore {
namespace kernel {
constexpr double kMinValue = -65504.;

template <typename K, typename V>
bool SortGpuKernelMod<K, V>::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                          const std::vector<KernelTensor *> &workspace,
                                          const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_LOG(EXCEPTION) << "Only support input datatype in [float16, float32] for sort kernel";
  return false;
}

template <>
bool SortGpuKernelMod<int32_t, half>::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                                   const std::vector<KernelTensor *> &workspace,
                                                   const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  cuda_stream_ = reinterpret_cast<cudaStream_t>(stream_ptr);
  if (is_null_input_) {
    return true;
  }
  half *input_device = GetDeviceAddress<half>(inputs, kIndex0);
  half *output_device = GetDeviceAddress<half>(outputs, kIndex0);
  int32_t *indices_device = GetDeviceAddress<int32_t>(outputs, kIndex1);
  half *temp_output_device = GetDeviceAddress<half>(workspace, kIndex0);
  int32_t *temp_indices_device = GetDeviceAddress<int32_t>(workspace, kIndex1);

  TransposeInfo InInfo;
  TransposeInfo TrInfo;
  for (size_t i = 0; i < input_rank_; ++i) {
    InInfo.input_shape.push_back(static_cast<int64_t>(input_shape_[i]));
    InInfo.perm.push_back(static_cast<int32_t>(perm_[i]));
    TrInfo.input_shape.push_back(static_cast<int64_t>(transposed_shape_[i]));
    TrInfo.perm.push_back(static_cast<int32_t>(perm_[i]));
  }

  // Sort is implemented using a combination of Neg, Transpose, and TopK. It's
  // Not safe to treat Transpose and TopK as inplace operators, so we alternate
  // between using temp_output_device and output_device for intermediate calculations,
  // this way only a constant number of allocations is needed instead of needing to
  // allocate once for each intermediate calculation.
  half *intermediate_input_device = input_device;
  half *intermediate_output_device = output_device;

  cudaError_t status = cudaErrorNotReady;
  // if sort not in descending order, negate input and negate back after sorting
  if (!descending_) {
    status = UnaryOpsCudaFunc<ElwiseOpType::kNeg, half, half>(
      input_size_, intermediate_input_device, intermediate_output_device, reinterpret_cast<cudaStream_t>(stream_ptr));
    CHECK_CUDA_STATUS(status, kernel_name_);
    intermediate_input_device = output_device;
    intermediate_output_device = temp_output_device;
  }

  // transpose so that desired dimension to sort along becomes the last one
  status = CalTranspose<half, true>(input_size_, intermediate_input_device, InInfo, intermediate_output_device,
                                    reinterpret_cast<cudaStream_t>(stream_ptr));
  CHECK_CUDA_STATUS(status, "Transpose called by " + kernel_name_);
  intermediate_input_device = intermediate_output_device;
  intermediate_output_device = intermediate_input_device == output_device ? temp_output_device : output_device;

  // topk sorts the input along the last dimension
  half topk_init_ = static_cast<half>(kMinValue);
  status =
    FastTopK(outer_size_, inner_size_, intermediate_input_device, static_cast<int32_t>(input_shape_[axis_]),
             intermediate_output_device, temp_indices_device, topk_init_, reinterpret_cast<cudaStream_t>(stream_ptr));
  CHECK_CUDA_STATUS(status, kernel_name_);
  std::swap(intermediate_input_device, intermediate_output_device);

  // transpose the sorted output back to the original input shape
  status = CalTranspose<half, true>(input_size_, intermediate_input_device, TrInfo, intermediate_output_device,
                                    reinterpret_cast<cudaStream_t>(stream_ptr));
  CHECK_CUDA_STATUS(status, "Transpose called by " + kernel_name_);

  // transpose the indices back to the original input shape
  status = CalTranspose<int32_t, true>(input_size_, temp_indices_device, TrInfo, indices_device,
                                       reinterpret_cast<cudaStream_t>(stream_ptr));
  CHECK_CUDA_STATUS(status, "Transpose called by " + kernel_name_);

  // negate back the sorted values if we negated prior to sorting
  if (!descending_) {
    std::swap(intermediate_input_device, intermediate_output_device);
    status = UnaryOpsCudaFunc<ElwiseOpType::kNeg, half, half>(
      input_size_, intermediate_input_device, intermediate_output_device, reinterpret_cast<cudaStream_t>(stream_ptr));
    CHECK_CUDA_STATUS(status, kernel_name_);
  }
  return true;
}

template <>
bool SortGpuKernelMod<int32_t, float>::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                                    const std::vector<KernelTensor *> &workspace,
                                                    const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  cuda_stream_ = reinterpret_cast<cudaStream_t>(stream_ptr);
  if (is_null_input_) {
    return true;
  }
  float *input_device = GetDeviceAddress<float>(inputs, kIndex0);
  float *output_device = GetDeviceAddress<float>(outputs, kIndex0);
  int32_t *indices_device = GetDeviceAddress<int32_t>(outputs, kIndex1);
  int32_t *temp_indices_device = GetDeviceAddress<int32_t>(workspace, kIndex1);
  float *temp_output_device = GetDeviceAddress<float>(workspace, kIndex0);

  TransposeInfo InInfo;
  TransposeInfo TrInfo;
  for (size_t i = 0; i < input_rank_; ++i) {
    InInfo.input_shape.push_back(static_cast<int64_t>(input_shape_[i]));
    InInfo.perm.push_back(static_cast<int32_t>(perm_[i]));
    TrInfo.input_shape.push_back(static_cast<int64_t>(transposed_shape_[i]));
    TrInfo.perm.push_back(static_cast<int32_t>(perm_[i]));
  }

  // Sort is implemented using a combination of Neg, Transpose, and TopK. It's
  // Not safe to treat Transpose and TopK as inplace operators, so we alternate
  // between using temp_output_device and output_device for intermediate calculations,
  // this way only a constant number of allocations is needed instead of needing to
  // allocate once for each intermediate calculation.
  float *intermediate_input_device = input_device;
  float *intermediate_output_device = output_device;

  // if sort not in descending order, negate input and negate back after sorting
  cudaError_t status = cudaErrorNotReady;
  if (!descending_) {
    status = UnaryOpsCudaFunc<ElwiseOpType::kNeg, float, float>(
      input_size_, intermediate_input_device, intermediate_output_device, reinterpret_cast<cudaStream_t>(stream_ptr));
    CHECK_CUDA_STATUS(status, kernel_name_);
    intermediate_input_device = output_device;
    intermediate_output_device = temp_output_device;
  }

  // transpose so that desired dimension to sort along becomes the last one
  status = CalTranspose<float, true>(input_size_, intermediate_input_device, InInfo, intermediate_output_device,
                                     reinterpret_cast<cudaStream_t>(stream_ptr));
  CHECK_CUDA_STATUS(status, "Transpose called by " + kernel_name_);
  intermediate_input_device = intermediate_output_device;
  intermediate_output_device = intermediate_input_device == output_device ? temp_output_device : output_device;

  // topk sorts the input along the last dimension
  float topk_init_ = std::numeric_limits<float>::lowest();
  status =
    FastTopK(outer_size_, inner_size_, intermediate_input_device, static_cast<int32_t>(input_shape_[axis_]),
             intermediate_output_device, temp_indices_device, topk_init_, reinterpret_cast<cudaStream_t>(stream_ptr));
  CHECK_CUDA_STATUS(status, kernel_name_);
  std::swap(intermediate_input_device, intermediate_output_device);

  // transpose the sorted output back to the original input shape
  status = CalTranspose<float, true>(input_size_, intermediate_input_device, TrInfo, intermediate_output_device,
                                     reinterpret_cast<cudaStream_t>(stream_ptr));
  CHECK_CUDA_STATUS(status, "Transpose called by " + kernel_name_);

  // transpose the indices back to the original input shape
  status = CalTranspose<int32_t, true>(input_size_, temp_indices_device, TrInfo, indices_device,
                                       reinterpret_cast<cudaStream_t>(stream_ptr));
  CHECK_CUDA_STATUS(status, "Transpose called by " + kernel_name_);

  // negate back the sorted values if we negated prior to sorting
  if (!descending_) {
    std::swap(intermediate_input_device, intermediate_output_device);
    status = UnaryOpsCudaFunc<ElwiseOpType::kNeg, float, float>(
      input_size_, intermediate_input_device, intermediate_output_device, reinterpret_cast<cudaStream_t>(stream_ptr));
    CHECK_CUDA_STATUS(status, kernel_name_);
  }

  return true;
}

MS_REG_GPU_KERNEL_TWO(
  Sort, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeInt32),
  SortGpuKernelMod, int32_t, bool);
MS_REG_GPU_KERNEL_TWO(
  Sort, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt32),
  SortGpuKernelMod, int32_t, int8_t);
MS_REG_GPU_KERNEL_TWO(
  Sort, KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt32),
  SortGpuKernelMod, int32_t, int16_t);
MS_REG_GPU_KERNEL_TWO(
  Sort, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
  SortGpuKernelMod, int32_t, int32_t);
MS_REG_GPU_KERNEL_TWO(
  Sort, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt32),
  SortGpuKernelMod, int32_t, int64_t);
MS_REG_GPU_KERNEL_TWO(
  Sort, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeInt32),
  SortGpuKernelMod, int32_t, uint8_t);
MS_REG_GPU_KERNEL_TWO(
  Sort, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeInt32),
  SortGpuKernelMod, int32_t, half);
MS_REG_GPU_KERNEL_TWO(
  Sort, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt32),
  SortGpuKernelMod, int32_t, float);
MS_REG_GPU_KERNEL_TWO(
  Sort, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeInt32),
  SortGpuKernelMod, int32_t, double);
}  // namespace kernel
}  // namespace mindspore
