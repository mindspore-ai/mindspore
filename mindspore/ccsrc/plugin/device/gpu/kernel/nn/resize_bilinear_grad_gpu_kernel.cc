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

#include "plugin/device/gpu/kernel/nn/resize_bilinear_grad_gpu_kernel.h"
#include <map>
#include <functional>
#include <algorithm>
#include <utility>
#include "mindspore/core/ops/grad/resize_bilinear_grad.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/resize_bilinear_impl.cuh"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kInputsNum = 2;
constexpr size_t kOutputsNum = 1;
constexpr size_t kDyIndexForN = 0;
constexpr size_t kDyIndexForC = 1;
constexpr size_t kDyIndexForH = 2;
constexpr size_t kDyIndexForW = 3;
constexpr size_t kDxIndexForH = 2;
constexpr size_t kDxIndexForW = 3;
}  // namespace

using FuncVec = std::vector<std::pair<KernelAttr, ResizeBilinearGradGpuKernelMod::ResizeBilinearGradFunc>>;

bool ResizeBilinearGradGpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                          const std::vector<KernelTensorPtr> &inputs,
                                          const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::ResizeBilinearGrad>(base_operator);
  if (!kernel_ptr) {
    MS_LOG(ERROR) << "cast ResizeBilinearGrad ops failed!";
    return false;
  }
  kernel_name_ = kernel_ptr->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputsNum, kernel_name_);

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For 'ResizeBilinearGrad', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int ResizeBilinearGradGpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                           const std::vector<KernelTensorPtr> &inputs,
                                           const std::vector<KernelTensorPtr> &outputs,
                                           const std::map<uint32_t, tensor::TensorPtr> &) {
  ResetResource();
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  auto dy_shape = inputs[kIndex0]->GetShapeVector();
  auto dx_shape = outputs[kIndex0]->GetShapeVector();
  auto input_element_num = std::accumulate(dy_shape.begin(), dy_shape.end(), size_t(1), std::multiplies<size_t>());
  is_null_input_ = (input_element_num == 0);
  if (is_null_input_) {
    return static_cast<int>(KRET_OK);
  }

  auto align_corners = base_operator->GetAttr(kAttrAlignCorners);
  MS_EXCEPTION_IF_NULL(align_corners);
  align_corners_ = GetValue<bool>(align_corners);
  auto half_pixel_centers = base_operator->GetAttr(kAttrHalfPixelCenters);
  MS_EXCEPTION_IF_NULL(half_pixel_centers);
  half_pixel_centers_ = GetValue<bool>(half_pixel_centers);

  n_ = LongToInt(dy_shape[kDyIndexForN]);
  c_ = LongToInt(dy_shape[kDyIndexForC]);
  dy_h_ = LongToInt(dy_shape[kDyIndexForH]);
  dy_w_ = LongToInt(dy_shape[kDyIndexForW]);
  dx_h_ = LongToInt(dx_shape[kDxIndexForH]);
  dx_w_ = LongToInt(dx_shape[kDxIndexForW]);
  dy_size_ = abstract::TypeIdSize(inputs[0]->GetDtype()) * SizeOf(dy_shape);
  dx_size_ = abstract::TypeIdSize(inputs[1]->GetDtype()) * SizeOf(dx_shape);
  if (inputs[0]->GetDtype() == kNumberTypeFloat16) {
    workspace_size_ = SizeOf(dx_shape) * sizeof(float);
  } else {
    workspace_size_ = dx_size_;
  }
  InitSizeLists();
  return static_cast<int>(KRET_OK);
}

template <typename T>
bool ResizeBilinearGradGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                  const std::vector<AddressPtr> &workspace,
                                                  const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  T *dy = GetDeviceAddress<T>(inputs, 0);
  T *interim = GetDeviceAddress<T>(workspace, 0);
  T *dx = GetDeviceAddress<T>(outputs, 0);
  float h_scale = Scaling(dx_h_, dy_h_, align_corners_);
  float w_scale = Scaling(dx_w_, dy_w_, align_corners_);
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMemsetAsync(dx, 0, dx_size_, reinterpret_cast<cudaStream_t>(stream_ptr)),
                                     "cudaMemsetAsync dx failed");
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemsetAsync(interim, 0, workspace_size_, reinterpret_cast<cudaStream_t>(stream_ptr)),
    "cudaMemsetAsync dx_interim failed");
  CalResizeBilinearGrad(dy, n_, c_, dy_h_, dy_w_, dx_h_, dx_w_, h_scale, w_scale, half_pixel_centers_, dx, interim,
                        device_id_, reinterpret_cast<cudaStream_t>(stream_ptr));
  return true;
}

template <typename T>
bool ResizeBilinearGradGpuKernelMod::LaunchHalfKernel(const std::vector<AddressPtr> &inputs,
                                                      const std::vector<AddressPtr> &workspace,
                                                      const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  T *dy = GetDeviceAddress<T>(inputs, 0);
  float *interim = GetDeviceAddress<float>(workspace, 0);
  T *dx = GetDeviceAddress<T>(outputs, 0);
  float h_scale = Scaling(dx_h_, dy_h_, align_corners_);
  float w_scale = Scaling(dx_w_, dy_w_, align_corners_);
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMemsetAsync(dx, 0, dx_size_, reinterpret_cast<cudaStream_t>(stream_ptr)),
                                     "cudaMemsetAsync dx failed");
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemsetAsync(interim, 0, workspace_size_, reinterpret_cast<cudaStream_t>(stream_ptr)),
    "cudaMemsetAsync dx_interim failed");
  CalResizeBilinearGradHalf(dy, n_, c_, dy_h_, dy_w_, dx_h_, dx_w_, h_scale, w_scale, half_pixel_centers_, dx, interim,
                            device_id_, reinterpret_cast<cudaStream_t>(stream_ptr));
  return true;
}

FuncVec ResizeBilinearGradGpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
   &ResizeBilinearGradGpuKernelMod::LaunchHalfKernel<half>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
   &ResizeBilinearGradGpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
   &ResizeBilinearGradGpuKernelMod::LaunchKernel<double>}};

std::vector<KernelAttr> ResizeBilinearGradGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, ResizeBilinearGradFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, ResizeBilinearGrad, ResizeBilinearGradGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
