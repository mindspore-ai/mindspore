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

#include "plugin/device/gpu/kernel/nn/upsample_nearest3d_gpu_kernel.h"
#include <algorithm>
#include <vector>
#include <memory>
#include <map>
#include <string>
#include <utility>
#include <functional>
#include "abstract/utils.h"
#include "kernel/common_utils.h"
#include "mindspore/core/ops/upsample_nearest_3d.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/upsample_nearest_impl.cuh"

namespace mindspore {
namespace kernel {
float UpsampleNearest3dGpuKernelMod::Scaling(const size_t in_size, const size_t out_size, int idx) {
  // for input/output size
  if (out_size == 0) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', output_size dimension " << idx << " is equal to 0.";
  } else {
    return 1.0f * in_size / out_size;
  }
}

float UpsampleNearest3dGpuKernelMod::Scaling(const float scale_value, int idx) {
  // for scale factors
  if (scale_value == 0) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', scales dimension " << idx << " is <= 0.";
  } else {
    return 1.0f / scale_value;
  }
}

bool UpsampleNearest3dGpuKernelMod::GetUpsampleNearest3dAttr(const BaseOperatorPtr &base_operator) {
  if (kernel_name_ != prim::kPrimUpsampleNearest3D->name()) {
    MS_LOG(ERROR) << "For '" << prim::kPrimUpsampleNearest3D->name()
                  << "' , it's kernel name must be equal to UpsampleNearest3D, but got " << kernel_name_;
    return false;
  }
  auto kernel_ptr = std::make_shared<ops::UpsampleNearest3D>(base_operator->GetPrim());
  output_volumetric_size_ = kernel_ptr->get_output_size_attr();
  scale_factors_ = kernel_ptr->get_scales_attr();
  if (output_volumetric_size_.empty() && scale_factors_.empty()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', either output_size or scales should be defined.";
  } else if (!output_volumetric_size_.empty() && !scale_factors_.empty()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', only one of output_size or scales should be defined.";
  }
  return true;
}

bool UpsampleNearest3dGpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                         const std::vector<KernelTensorPtr> &inputs,
                                         const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it got empty inputs or outputs, which is invalid.";
    return false;
  }
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), 1, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), 1, kernel_name_);
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return GetUpsampleNearest3dAttr(base_operator);
}

int UpsampleNearest3dGpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                          const std::vector<KernelTensorPtr> &inputs,
                                          const std::vector<KernelTensorPtr> &outputs,
                                          const std::map<uint32_t, tensor::TensorPtr> &) {
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }

  input_shape_.clear();
  auto input_shape = inputs.at(kIndex0)->GetShapeVector();
  (void)std::transform(input_shape.begin(), input_shape.end(), std::back_inserter(input_shape_), LongToSize);
  size_t input_elements =
    std::accumulate(input_shape_.begin(), input_shape_.end(), size_t(1), std::multiplies<size_t>());
  is_null_input_ = (input_elements == 0);
  if (is_null_input_) {
    return KRET_OK;
  }

  output_shape_.clear();
  auto output_shape = outputs.at(kIndex0)->GetShapeVector();
  (void)std::transform(output_shape.begin(), output_shape.end(), std::back_inserter(output_shape_), LongToSize);

  if (!output_volumetric_size_.empty()) {
    scale_factors_.clear();
    scale_factors_.push_back(Scaling(input_shape_[kIndex2], output_volumetric_size_[kIndex0], kIndex0));
    scale_factors_.push_back(Scaling(input_shape_[kIndex3], output_volumetric_size_[kIndex1], kIndex1));
    scale_factors_.push_back(Scaling(input_shape_[kIndex4], output_volumetric_size_[kIndex2], kIndex2));
  } else {
    float d_scale = scale_factors_[kIndex0];
    float h_scale = scale_factors_[kIndex1];
    float w_scale = scale_factors_[kIndex2];
    scale_factors_.clear();
    scale_factors_.push_back(Scaling(d_scale, kIndex0));
    scale_factors_.push_back(Scaling(h_scale, kIndex1));
    scale_factors_.push_back(Scaling(w_scale, kIndex2));
  }
  return KRET_OK;
}

template <typename T>
bool UpsampleNearest3dGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                 const std::vector<AddressPtr> &workspace,
                                                 const std::vector<AddressPtr> &outputs) {
  auto input = reinterpret_cast<T *>(inputs.at(kIndex0)->addr);
  auto output = reinterpret_cast<T *>(outputs.at(kIndex0)->addr);
  CalUpsampleNearest3d(input, input_shape_[kIndex0], input_shape_[kIndex1], input_shape_[kIndex2],
                       input_shape_[kIndex3], input_shape_[kIndex4], output_shape_[kIndex2], output_shape_[kIndex3],
                       output_shape_[kIndex4], scale_factors_[kIndex0], scale_factors_[kIndex1],
                       scale_factors_[kIndex2], output, reinterpret_cast<cudaStream_t>(cuda_stream_));
  return true;
}

std::vector<std::pair<KernelAttr, UpsampleNearest3dGpuKernelMod::UpsampleNearest3dFunc>>
  UpsampleNearest3dGpuKernelMod::func_list_ = {
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
     &UpsampleNearest3dGpuKernelMod::LaunchKernel<half>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &UpsampleNearest3dGpuKernelMod::LaunchKernel<float>}};
std::vector<KernelAttr> UpsampleNearest3dGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, UpsampleNearest3dFunc> &pair) { return pair.first; });
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, UpsampleNearest3D, UpsampleNearest3dGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
