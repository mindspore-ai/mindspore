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

#include "plugin/device/gpu/kernel/nn/upsample_nearest_3d_gpu_grad.h"
#include <functional>
#include <utility>
#include <iostream>
#include <string>
#include <algorithm>
#include "mindspore/core/abstract/utils.h"
#include "mindspore/core/ops/grad/upsample_nearest_3d_grad.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/upsample_nearest_3d_grad_impl.cuh"
#include "kernel/common_utils.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr int kUpsampleNearest3DGpuGradInputsNum = 1;
constexpr int kUpsampleNearest3DGpuGradOutputsNum = 1;
}  // namespace

void UpsampleNearest3DGradGpuKernelMod::ResetResource() noexcept {
  n_ = 0;
  c_ = 0;
  dy_d_ = 0;
  dy_h_ = 0;
  dy_w_ = 0;
  dx_d_ = 0;
  dx_h_ = 0;
  dx_w_ = 0;
  scale_factors_.clear();
  input_size_list_.clear();
  output_size_list_.clear();
  workspace_size_list_.clear();
}

template <typename F>
void UpsampleNearest3DGradGpuKernelMod::CheckDims(string check_dim_name, int expected_size,
                                                  std::vector<F> check_vector) {
  int dim_size = check_vector.size();
  if (dim_size != expected_size) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', " << check_dim_name << " size is " << dim_size << " should be"
                      << expected_size << ".";
  }
  for (int i = 0; i < dim_size; i++) {
    if (check_vector[i] <= 0) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', " << check_dim_name << " dimension " << i
                        << " value is <= 0. ";
    }
  }
}

float UpsampleNearest3DGradGpuKernelMod::ScalingSizes(const size_t in_size, const size_t out_size) {
  if (out_size == 0) {
    MS_LOG(EXCEPTION) << kernel_name_ << "', output (dx) shape contains 0.";
  }
  return in_size / static_cast<float>(out_size);
}

float UpsampleNearest3DGradGpuKernelMod::ScalingScales(float scale_value, size_t idx) {
  if (scale_value <= 0.0f) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', "
                      << "scales "
                      << "dimension " << idx << " value is <= 0.";
  } else {
    return scale_value;
  }
}

bool UpsampleNearest3DGradGpuKernelMod::GetUpsampleNearest3DAttr(const BaseOperatorPtr &base_operator) {
  if (kernel_name_ != prim::kPrimUpsampleNearest3DGrad->name()) {
    MS_LOG(ERROR) << "For '" << prim::kPrimUpsampleNearest3DGrad->name()
                  << "' , it's kernel name must be equal to UpsampleNearest3D, but got " << kernel_name_;
    return false;
  }
  auto kernel_ptr = std::make_shared<ops::UpsampleNearest3DGrad>(base_operator->GetPrim());
  out_spatial_size_me_ = kernel_ptr->get_out_spatial_size();
  scale_factors_ = kernel_ptr->get_scale_factors();
  return true;
}

bool UpsampleNearest3DGradGpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                             const std::vector<KernelTensorPtr> &inputs,
                                             const std::vector<KernelTensorPtr> &outputs) {
  MS_ERROR_IF_NULL_W_RET_VAL(base_operator, false);
  kernel_name_ = base_operator->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kUpsampleNearest3DGpuGradInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kUpsampleNearest3DGpuGradOutputsNum, kernel_name_);
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  t_size_ = abstract::TypeIdSize(kernel_attr.GetOutputAttr(kIndex0).first);
  kernel_func_ = func_list_[index].second;
  return GetUpsampleNearest3DAttr(base_operator);
}

int UpsampleNearest3DGradGpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                              const std::vector<KernelTensorPtr> &inputs,
                                              const std::vector<KernelTensorPtr> &outputs,
                                              const std::map<uint32_t, tensor::TensorPtr> &) {
  ResetResource();
  std::vector<int64_t> dy_shape = inputs[kIndex0]->GetShapeVector();
  std::vector<int64_t> dx_shape = outputs[kIndex0]->GetShapeVector();
  size_t dy_size = std::accumulate(dy_shape.begin(), dy_shape.end(), t_size_, std::multiplies<size_t>());
  size_t dx_size = std::accumulate(dx_shape.begin(), dx_shape.end(), t_size_, std::multiplies<size_t>());
  n_ = dy_shape[kIndex0];
  c_ = dy_shape[kIndex1];
  // input
  dy_d_ = dy_shape[kIndex2];
  dy_h_ = dy_shape[kIndex3];
  dy_w_ = dy_shape[kIndex4];
  // output
  dx_d_ = dx_shape[kIndex2];
  dx_h_ = dx_shape[kIndex3];
  dx_w_ = dx_shape[kIndex4];
  if (!out_spatial_size_me_.empty()) {
    scale_factors_.emplace_back(ScalingSizes(dy_d_, dx_d_));
    scale_factors_.emplace_back(ScalingSizes(dy_h_, dx_h_));
    scale_factors_.emplace_back(ScalingSizes(dy_w_, dx_w_));
  } else {
    scale_factors_.emplace_back(ScalingScales(scale_factors_[kIndex0], kIndex0));
    scale_factors_.emplace_back(ScalingScales(scale_factors_[kIndex1], kIndex1));
    scale_factors_.emplace_back(ScalingScales(scale_factors_[kIndex2], kIndex2));
  }
  input_size_list_.emplace_back(dy_size);
  output_size_list_.emplace_back(dx_size);
  return KRET_OK;
}

template <typename T>
bool UpsampleNearest3DGradGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                     const std::vector<AddressPtr> &workspace,
                                                     const std::vector<AddressPtr> &outputs) {
  auto dy = reinterpret_cast<T *>(inputs.at(kIndex0)->addr);
  auto dx = reinterpret_cast<T *>(outputs.at(kIndex0)->addr);
  CalUpsampleNearest3DGrad(dy, n_, c_, dy_d_, dy_h_, dy_w_, dx_d_, dx_h_, dx_w_, scale_factors_[kIndex0],
                           scale_factors_[kIndex1], scale_factors_[kIndex2], dx,
                           reinterpret_cast<cudaStream_t>(cuda_stream_));
  return true;
}

std::vector<std::pair<KernelAttr, UpsampleNearest3DGradGpuKernelMod::UpsampleNearest3DGradFunc>>
  UpsampleNearest3DGradGpuKernelMod::func_list_ = {
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
     &UpsampleNearest3DGradGpuKernelMod::LaunchKernel<half>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &UpsampleNearest3DGradGpuKernelMod::LaunchKernel<float>}};
std::vector<KernelAttr> UpsampleNearest3DGradGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, UpsampleNearest3DGradFunc> &pair) { return pair.first; });
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, UpsampleNearest3DGrad, UpsampleNearest3DGradGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
