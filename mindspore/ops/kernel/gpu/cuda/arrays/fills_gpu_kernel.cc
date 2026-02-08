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

#include "kernel/gpu/cuda/arrays/fills_gpu_kernel.h"
#include <limits>
#include <memory>
#include <functional>
#include <algorithm>
#include "base/float16.h"
#include "mindspore/ops/infer/fills.h"
#include "kernel/gpu/cuda_impl/cuda_ops/fills_impl.cuh"
namespace mindspore {
namespace kernel {
template <typename T>
typename std::enable_if<std::is_same<T, half>::value, bool>::type overflows(float f) {
  using limit = std::numeric_limits<float16>;
  if (std::isinf(f) || (f != f)) {
    return false;
  }
  return f < static_cast<float>(limit::lowest()) || f > static_cast<float>(limit::max());
}

template <typename T>
typename std::enable_if<!std::is_same<T, half>::value, bool>::type overflows(float f) {
  using limit = std::numeric_limits<T>;
  if (std::isinf(f)) {
    return !limit::has_infinity;
  }
  if (!limit::has_quiet_NaN && (f != f)) {
    return true;
  }
  return f < limit::lowest() || f > limit::max();
}

#define FILLS_GPU_REG(MS_T, T)                                                            \
  {                                                                                       \
    KernelAttr().AddInputAttr(MS_T).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(MS_T), \
      &FillsGpuKernelMod::LaunchKernel<T>                                                 \
  }

std::vector<std::pair<KernelAttr, FillsGpuKernelMod::FillsFunc>> FillsGpuKernelMod::func_list_ = {
  FILLS_GPU_REG(kNumberTypeInt8, int8_t), FILLS_GPU_REG(kNumberTypeInt16, int16_t),
  FILLS_GPU_REG(kNumberTypeInt32, int32_t), FILLS_GPU_REG(kNumberTypeFloat16, half),
  FILLS_GPU_REG(kNumberTypeFloat32, float)};

bool FillsGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  auto tensor_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(tensor_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel type: " << tensor_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  auto x_type_id = tensor_attr.GetInputAttr(kIndex0).dtype;
  unit_size_ = abstract::TypeIdSize(x_type_id);
  x_type_str_ = TypeIdToString(x_type_id);
  return true;
}

int FillsGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  ResetResource();
  int ret = KRET_OK;
  if ((ret = KernelMod::Resize(inputs, outputs)) != KRET_OK) {
    return ret;
  }
  auto shape = inputs.at(kIndex0)->GetShapeVector();
  input_elements_ = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
  auto workspace_size = sizeof(float);
  workspace_size_list_.emplace_back(workspace_size);
  return KRET_OK;
}

std::vector<KernelAttr> FillsGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, FillsFunc> &item) { return item.first; });
  return support_list;
}

void FillsGpuKernelMod::ResetResource() noexcept {
  is_null_input_ = false;
  input_elements_ = 0;
  output_size_list_.clear();
  workspace_size_list_.clear();
}

template <typename T>
bool FillsGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                     const std::vector<KernelTensor *> &workspace,
                                     const std::vector<KernelTensor *> &outputs) {
  auto value_ptr = GetDeviceAddress<float>(inputs, kIndex1);
  auto y_ptr = GetDeviceAddress<T>(outputs, kIndex0);
  float value = 0;
  auto cuda_stream = reinterpret_cast<cudaStream_t>(cuda_stream_);
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(&value, value_ptr, sizeof(float), cudaMemcpyDeviceToHost, cuda_stream),
    "cudaMemcpy value variable failed.");
  if (overflows<T>(value)) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', value cannot be converted to type " << x_type_str_
                  << " without overflow: " << value;
    return false;
  }
  auto status = FillsForward(input_elements_, value_ptr, y_ptr, device_id_, cuda_stream);
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, Fills, FillsGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
