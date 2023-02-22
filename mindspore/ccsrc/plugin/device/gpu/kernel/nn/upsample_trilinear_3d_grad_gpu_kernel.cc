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

#include "plugin/device/gpu/kernel/nn/upsample_trilinear_3d_grad_gpu_kernel.h"
#include <algorithm>
#include <vector>
#include <memory>
#include <map>
#include <utility>
#include "abstract/utils.h"
#include "kernel/common_utils.h"
#include "mindspore/core/ops/grad/upsample_trilinear_3d_grad.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/upsample_trilinear_3d_grad_impl.cuh"
#include "plugin/device/gpu/kernel/nn/upsample_trilinear_3d_gpu_kernel.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr int kInputsNum = 1;
constexpr int kOutputsNum = 1;
}  // namespace

void UpsampleTrilinear3DGradGpuKernelMod::ResetResource() noexcept {
  n_ = 0;
  c_ = 0;
  grad_d_ = 0;
  grad_h_ = 0;
  grad_w_ = 0;
  dinput_d_ = 0;
  dinput_h_ = 0;
  dinput_w_ = 0;
  scale_factors_.clear();
  input_size_list_.clear();
  output_size_list_.clear();
  workspace_size_list_.clear();
}

float UpsampleTrilinear3DGradGpuKernelMod::ScalingD(const size_t in_size, const size_t out_size, bool align_corners) {
  // for input/output size
  if (out_size > 1) {
    return align_corners ? (in_size - 1) / static_cast<float>(out_size - 1) : in_size / static_cast<float>(out_size);
  } else {
    return static_cast<float>(0.0);
  }
}

float UpsampleTrilinear3DGradGpuKernelMod::ScalingS(float scale_value, int idx, const size_t out_size) {
  // for scale factors
  if (out_size > 1) {
    if (scale_value <= 0.0f) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', "
                        << "scales "
                        << "dimension " << idx << " value is <= 0.";
    } else {
      return static_cast<float>(1.0 / scale_value);
    }
  } else {
    return static_cast<float>(0.0);
  }
}

bool UpsampleTrilinear3DGradGpuKernelMod::GetUpsampleTrilinear3DGradAttr(const BaseOperatorPtr &base_operator) {
  if (kernel_name_ != prim::kPrimUpsampleTrilinear3DGrad->name()) {
    MS_LOG(ERROR) << "For '" << prim::kPrimUpsampleTrilinear3DGrad->name()
                  << "' , it's kernel name must be equal to UpsampleTrilinear3DGrad, but got " << kernel_name_;
    return false;
  }
  auto kernel_ptr = std::make_shared<ops::UpsampleTrilinear3DGrad>(base_operator->GetPrim());
  align_corners_ = kernel_ptr->get_align_corners();
  out_spatial_size_me_ = kernel_ptr->get_out_spatial_size();
  scale_factors_ = kernel_ptr->get_scale_factors();
  return true;
}

bool UpsampleTrilinear3DGradGpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                               const std::vector<KernelTensorPtr> &inputs,
                                               const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it got empty inputs or outputs, which is invalid.";
    return false;
  }
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputsNum, kernel_name_);
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel type: " << kernel_attr;
    return false;
  }
  t_size_ = abstract::TypeIdSize(kernel_attr.GetOutputAttr(kIndex0).dtype);
  kernel_func_ = func_list_[index].second;
  return GetUpsampleTrilinear3DGradAttr(base_operator);
}

int UpsampleTrilinear3DGradGpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                                const std::vector<KernelTensorPtr> &inputs,
                                                const std::vector<KernelTensorPtr> &outputs,
                                                const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  ResetResource();
  std::vector<int64_t> grad_shape = inputs[kIndex0]->GetShapeVector();
  size_t grad_size = std::accumulate(grad_shape.begin(), grad_shape.end(), t_size_, std::multiplies<size_t>());
  std::vector<int64_t> dinput_shape = outputs[kIndex0]->GetShapeVector();
  size_t dinput_size = std::accumulate(dinput_shape.begin(), dinput_shape.end(), t_size_, std::multiplies<size_t>());
  n_ = grad_shape[kIndex0];
  c_ = grad_shape[kIndex1];
  grad_d_ = grad_shape[kIndex2];
  grad_h_ = grad_shape[kIndex3];
  grad_w_ = grad_shape[kIndex4];
  dinput_d_ = dinput_shape[kIndex2];
  dinput_h_ = dinput_shape[kIndex3];
  dinput_w_ = dinput_shape[kIndex4];
  if (!out_spatial_size_me_.empty() || align_corners_ == true) {
    scale_factors_.push_back(ScalingD(dinput_d_, grad_d_, align_corners_));
    scale_factors_.push_back(ScalingD(dinput_h_, grad_h_, align_corners_));
    scale_factors_.push_back(ScalingD(dinput_w_, grad_w_, align_corners_));
  } else {
    scale_factors_.push_back(ScalingS(scale_factors_[kIndex0], kIndex0, grad_d_));
    scale_factors_.push_back(ScalingS(scale_factors_[kIndex1], kIndex1, grad_h_));
    scale_factors_.push_back(ScalingS(scale_factors_[kIndex2], kIndex2, grad_w_));
  }
  input_size_list_.push_back(grad_size);
  output_size_list_.push_back(dinput_size);
  return KRET_OK;
}

template <typename T>
bool UpsampleTrilinear3DGradGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                       const std::vector<AddressPtr> &workspace,
                                                       const std::vector<AddressPtr> &outputs) {
  auto grad = reinterpret_cast<T *>(inputs.at(kIndex0)->addr);
  auto dinput = reinterpret_cast<T *>(outputs.at(kIndex0)->addr);

  CalUpsampleTrilinear3DGrad(grad, n_, c_, grad_d_, grad_h_, grad_w_, dinput_d_, dinput_h_, dinput_w_,
                             scale_factors_[kIndex0], scale_factors_[kIndex1], scale_factors_[kIndex2], align_corners_,
                             dinput, reinterpret_cast<cudaStream_t>(cuda_stream_));
  return true;
}
std::vector<std::pair<KernelAttr, UpsampleTrilinear3DGradGpuKernelMod::UpsampleTrilinear3DGradFunc>>
  UpsampleTrilinear3DGradGpuKernelMod::func_list_ = {
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
     &UpsampleTrilinear3DGradGpuKernelMod::LaunchKernel<half>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &UpsampleTrilinear3DGradGpuKernelMod::LaunchKernel<float>}};
std::vector<KernelAttr> UpsampleTrilinear3DGradGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, UpsampleTrilinear3DGradFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, UpsampleTrilinear3DGrad, UpsampleTrilinear3DGradGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
