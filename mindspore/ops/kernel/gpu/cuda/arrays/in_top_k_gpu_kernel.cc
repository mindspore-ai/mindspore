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

#include "kernel/gpu/cuda/arrays/in_top_k_gpu_kernel.h"
#include <cstdint>
#include <limits>
#include <algorithm>
#include "infer/in_top_k.h"
#include "kernel/gpu/cuda_impl/cuda_ops/cast_impl.cuh"
#include "kernel/gpu/cuda_impl/cuda_ops/in_top_k_impl.cuh"
#include "kernel/gpu/cuda_impl/cuda_ops/topk_impl.cuh"

namespace mindspore {
namespace kernel {
constexpr size_t kInTopKInputsNum = 2;
constexpr size_t kInTopKOutputsNum = 1;
constexpr size_t kInTopKShapeRank = 2;

bool InTopKGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInTopKInputsNum, kernel_name_);
  k_ = GetValue<int64_t>(primitive_->GetAttr("k"));
  dtype_ = inputs[0]->dtype_id();
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' does not support this kernel type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int InTopKGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInTopKInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kInTopKOutputsNum, kernel_name_);
  int ret = KernelMod::Resize(inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }

  input_shape_ = inputs[0]->GetShapeVector();
  if (input_shape_.size() < kInTopKShapeRank) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of input cannot be less than 2, but got "
                  << input_shape_.size();
    return KRET_RESIZE_FAILED;
  }

  is_null_input_ = CHECK_SHAPE_NULL(input_shape_, kernel_name_, "input");
  if (is_null_input_) {
    InitSizeLists();
    return ret;
  }

  input_size_ = 1;
  for (size_t i = 0; i < input_shape_.size(); i++) {
    input_size_ *= static_cast<size_t>(input_shape_[i]);
  }

  inner_size_ = static_cast<size_t>(input_shape_[1]);
  outer_size_ = static_cast<size_t>(input_shape_[0]);
  InitSizeLists();
  return ret;
}

void InTopKGpuKernelMod::InitSizeLists() {
  if (k_ > 0) {
    auto unit_size = GetTypeByte(TypeIdToType(dtype_));
    workspace_size_list_.push_back(static_cast<size_t>(input_shape_[0]) * k_ * unit_size);
    workspace_size_list_.push_back(static_cast<size_t>(input_shape_[0]) * k_ * sizeof(int32_t));
  }

  // remove later! urgent fix for bug: topk has incorrect output for float16
  if (dtype_ == kNumberTypeFloat16) {
    workspace_size_list_.push_back(input_size_ * sizeof(float));
    if (k_ > 0) {
      workspace_size_list_.push_back(static_cast<size_t>(input_shape_[0]) * k_ * sizeof(float));
    }
  }
}

template <typename T, typename S>
bool InTopKGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                      const std::vector<KernelTensor *> &workspace,
                                      const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  if (is_null_input_) {
    return true;
  }
  T *predictions_device = GetDeviceAddress<T>(inputs, kIndex0);
  S *targets_device = GetDeviceAddress<S>(inputs, kIndex1);

  bool *output_device = GetDeviceAddress<bool>(outputs, kIndex0);

  if (k_ <= 0) {
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemsetAsync(output_device, false, outer_size_, reinterpret_cast<cudaStream_t>(stream_ptr)),
      "cudaMemsetAsync failed.");
    return true;
  }

  k_ = std::min(k_, static_cast<int64_t>(inner_size_));

  cudaError_t status = cudaErrorNotReady;
  constexpr int64_t max_for_topk = 128;
  if (k_ > max_for_topk) {
    // ApplyInTopK have a better performance when k > 128.
    status = ApplyInTopK(predictions_device, targets_device, output_device, input_shape_[0], input_shape_[1], k_,
                         device_id_, reinterpret_cast<cudaStream_t>(stream_ptr));
    CHECK_CUDA_STATUS(status, kernel_name_);
    return true;
  }

  T *top_k_output_device = GetDeviceAddress<T>(workspace, kIndex0);
  int32_t *top_k_indices_device = GetDeviceAddress<int32_t>(workspace, kIndex1);

  if (std::is_same<T, half>::value) {
    // remove later! urgent fix for bug: topk has incorrect output for float16
    float top_k_init = std::numeric_limits<float>::lowest();

    // cast to float32
    float *casted_float32_input = GetDeviceAddress<float>(workspace, kIndex2);
    float *top_k_output_device_float32 = GetDeviceAddress<float>(workspace, kIndex3);

    status = Cast(input_size_, predictions_device, casted_float32_input, reinterpret_cast<cudaStream_t>(stream_ptr));
    CHECK_CUDA_STATUS(status, kernel_name_);
    status =
      FastTopK(outer_size_, inner_size_, casted_float32_input, static_cast<int32_t>(k_), top_k_output_device_float32,
               top_k_indices_device, top_k_init, reinterpret_cast<cudaStream_t>(stream_ptr));
    CHECK_CUDA_STATUS(status, kernel_name_);
    status = CalInTopK(casted_float32_input, targets_device, output_device, top_k_output_device_float32,
                       input_shape_[0], input_shape_[1], k_, reinterpret_cast<cudaStream_t>(stream_ptr));
    CHECK_CUDA_STATUS(status, kernel_name_);
  } else {
    // topk sorts the input along the last dimension
    T top_k_init;
    if (std::is_same<T, half>::value) {
      // min value representable by float16, std::numeric_limits doesn't support half
      float half_lowest = -65504;
      top_k_init = static_cast<half>(half_lowest);
    } else {
      top_k_init = std::numeric_limits<T>::lowest();
    }
    status = FastTopK(outer_size_, inner_size_, predictions_device, static_cast<int32_t>(k_), top_k_output_device,
                      top_k_indices_device, top_k_init, reinterpret_cast<cudaStream_t>(stream_ptr));
    CHECK_CUDA_STATUS(status, kernel_name_);

    status = CalInTopK(predictions_device, targets_device, output_device, top_k_output_device, input_shape_[0],
                       input_shape_[1], k_, reinterpret_cast<cudaStream_t>(stream_ptr));
    CHECK_CUDA_STATUS(status, kernel_name_);
  }

  return true;
}

std::vector<std::pair<KernelAttr, InTopKGpuKernelMod::InTopKFunc>> InTopKGpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeBool),
   &InTopKGpuKernelMod::LaunchKernel<half, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeBool),
   &InTopKGpuKernelMod::LaunchKernel<float, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeBool),
   &InTopKGpuKernelMod::LaunchKernel<half, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeBool),
   &InTopKGpuKernelMod::LaunchKernel<float, int64_t>},
};

std::vector<KernelAttr> InTopKGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, InTopKFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, InTopK, InTopKGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
