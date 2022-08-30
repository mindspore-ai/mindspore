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

#include "plugin/device/gpu/kernel/arrays/sort_gpu_kernel.h"
#include <map>

namespace mindspore {
namespace kernel {
constexpr double kMinValue = -65504.;
template <typename T>
bool SortGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                    const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  cuda_stream_ = reinterpret_cast<cudaStream_t>(stream_ptr);
  if (is_null_input_) {
    return true;
  }
  T *input_device = GetDeviceAddress<T>(inputs, kIndex0);

  T *output_device = GetDeviceAddress<T>(outputs, kIndex0);
  int32_t *indices_device = GetDeviceAddress<int32_t>(outputs, kIndex1);

  T *temp_output_device = GetDeviceAddress<T>(workspace, kIndex0);
  int32_t *temp_indices_device = GetDeviceAddress<int32_t>(workspace, kIndex1);
  size_t *input_shape_device = GetDeviceAddress<size_t>(workspace, kIndex2);
  size_t *perm_device = GetDeviceAddress<size_t>(workspace, kIndex3);
  size_t *transposed_shape_device = GetDeviceAddress<size_t>(workspace, kIndex4);

  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(input_shape_device, &input_shape_[0], workspace_size_list_[kIndex2], cudaMemcpyHostToDevice,
                    reinterpret_cast<cudaStream_t>(stream_ptr)),
    "cudaMemcpyAsync for input_shape_ failed");

  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(perm_device, &perm_[0], workspace_size_list_[kIndex3], cudaMemcpyHostToDevice,
                    reinterpret_cast<cudaStream_t>(stream_ptr)),
    "cudaMemcpyAsync for perm_ failed");

  // Sort is implemented using a combination of Neg, Transpose, and TopK. It's
  // Not safe to treat Transpose and TopK as inplace operators, so we alternate
  // between using temp_output_device and output_device for intermediate calculations,
  // this way only a constant number of allocations is needed instead of needing to
  // allocate once for each intermediate calculation.
  T *intermediate_input_device = input_device;
  T *intermediate_output_device = output_device;

  T topk_init_ = std::numeric_limits<T>::lowest();
  if (std::is_same<T, half>::value) {
    // min value representable by float16, std::numeric_limits doesn't support half
    topk_init_ = static_cast<half>(kMinValue);
  }

  // if sort not in descending order, negate input and negate back after sorting
  if (!descending_) {
    Negative(intermediate_input_device, intermediate_output_device, input_size_,
             reinterpret_cast<cudaStream_t>(stream_ptr));
    intermediate_input_device = output_device;
    intermediate_output_device = temp_output_device;
  }

  // transpose so that desired dimension to sort along becomes the last one
  CalTranspose(input_size_, intermediate_input_device, input_shape_device, perm_device, input_rank_,
               intermediate_output_device, reinterpret_cast<cudaStream_t>(stream_ptr));
  intermediate_input_device = intermediate_output_device;
  intermediate_output_device = intermediate_input_device == output_device ? temp_output_device : output_device;

  // topk sorts the input along the last dimension
  FastTopK(outer_size_, inner_size_, intermediate_input_device, static_cast<int32_t>(input_shape_[axis_]),
           intermediate_output_device, temp_indices_device, topk_init_, reinterpret_cast<cudaStream_t>(stream_ptr));
  std::swap(intermediate_input_device, intermediate_output_device);

  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(transposed_shape_device, &transposed_shape_[0], workspace_size_list_[kIndex4],
                    cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
    "cudaMemcpyAsync for transposed_shape_ failed");

  // transpose the sorted output back to the original input shape
  CalTranspose(input_size_, intermediate_input_device, transposed_shape_device, perm_device, input_rank_,
               intermediate_output_device, reinterpret_cast<cudaStream_t>(stream_ptr));

  // transpose the indices back to the original input shape
  CalTranspose(input_size_, temp_indices_device, transposed_shape_device, perm_device, input_rank_, indices_device,
               reinterpret_cast<cudaStream_t>(stream_ptr));

  // negate back the sorted values if we negated prior to sorting
  if (!descending_) {
    std::swap(intermediate_input_device, intermediate_output_device);
    Negative(intermediate_input_device, intermediate_output_device, input_size_,
             reinterpret_cast<cudaStream_t>(stream_ptr));
  }

  return true;
}

int SortGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                             const std::vector<KernelTensorPtr> &outputs,
                             const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  auto ret = KernelMod::Resize(base_operator, inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }

  input_shape_ = inputs[0]->GetShapeVector();
  auto kernel_name = base_operator->GetPrim()->name();
  is_null_input_ = CHECK_SHAPE_NULL(input_shape_, kernel_name, "input");
  if (is_null_input_) {
    return KRET_RESIZE_FAILED;
  }

  input_size_ = 1;
  for (size_t i = 0; i < input_rank_; i++) {
    input_size_ *= static_cast<size_t>(input_shape_[i]);
  }

  transposed_shape_ = input_shape_;
  std::swap(transposed_shape_[input_rank_ - 1], transposed_shape_[axis_]);
  inner_size_ = static_cast<size_t>(input_shape_[axis_]);
  outer_size_ = input_size_ / inner_size_;
  MS_LOG(DEBUG) << "In gpu kernel sort Resize, axis_=" << axis_ << " descending_=" << descending_
                << " input_rank_=" << input_rank_ << " input_size_=" << input_size_ << " inner_size_=" << inner_size_
                << " outer_size_=" << outer_size_;

  if (input_size_list_.size() > 0) {
    size_t input_bytes = input_size_list_.at(kIndex0);
    size_t indices_bytes = input_size_ * sizeof(int32_t);
    workspace_size_list_.push_back(input_bytes);
    workspace_size_list_.push_back(indices_bytes);
    workspace_size_list_.push_back(input_rank_ * sizeof(size_t));
    workspace_size_list_.push_back(input_rank_ * sizeof(size_t));
    workspace_size_list_.push_back(input_rank_ * sizeof(size_t));
  }
  return KRET_OK;
}

std::vector<std::pair<KernelAttr, SortGpuKernelMod::SortLaunchFunc>> SortGpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeInt32),
   &SortGpuKernelMod::LaunchKernel<half>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt32),
   &SortGpuKernelMod::LaunchKernel<float>}};

std::vector<KernelAttr> SortGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, SortGpuKernelMod::SortLaunchFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, Sort, SortGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
