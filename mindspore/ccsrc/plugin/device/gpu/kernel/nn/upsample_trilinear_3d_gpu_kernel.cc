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

#include "plugin/device/gpu/kernel/nn/upsample_trilinear_3d_gpu_kernel.h"
#include <algorithm>
#include <vector>
#include <memory>
#include <map>
#include <string>
#include <utility>
#include <functional>
#include "abstract/utils.h"
#include "kernel/common_utils.h"
#include "mindspore/core/ops/upsample_trilinear_3d.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/upsample_trilinear_3d_impl.cuh"

namespace mindspore {
namespace kernel {
namespace {
constexpr int kInputsNum = 1;
constexpr int kOutputsNum = 1;
}  // namespace
void UpsampleTrilinear3DGpuKernelMod::ResetResource() noexcept {
  n_ = 0;
  c_ = 0;
  input_d_ = 0;
  input_h_ = 0;
  input_w_ = 0;
  output_d_ = 0;
  output_h_ = 0;
  output_w_ = 0;
  scale_factors_.clear();
  input_size_list_.clear();
  output_size_list_.clear();
  workspace_size_list_.clear();
}

template <typename F>
void UpsampleTrilinear3DGpuKernelMod::CheckDims(string check_dim_name, int expected_size, std::vector<F> check_vector) {
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

float UpsampleTrilinear3DGpuKernelMod::ScalingD(const size_t in_size, const size_t out_size, bool align_corners) {
  // for input/output size
  if (out_size > 1) {
    return align_corners ? (in_size - 1) / static_cast<float>(out_size - 1) : in_size / static_cast<float>(out_size);
  } else {
    return static_cast<float>(0.0);
  }
}

float UpsampleTrilinear3DGpuKernelMod::ScalingS(float scale_value, int idx, const size_t out_size) {
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

bool UpsampleTrilinear3DGpuKernelMod::GetUpsampleTrilinear3DAttr(const BaseOperatorPtr &base_operator) {
  if (kernel_name_ != prim::kPrimUpsampleTrilinear3D->name()) {
    MS_LOG(ERROR) << "For '" << prim::kPrimUpsampleTrilinear3D->name()
                  << "' , it's kernel name must be equal to UpsampleTrilinear3D, but got " << kernel_name_;
    return false;
  }
  auto kernel_ptr = std::make_shared<ops::UpsampleTrilinear3D>(base_operator->GetPrim());
  align_corners_ = kernel_ptr->get_align_corners();
  out_spatial_size_me_ = kernel_ptr->get_output_size_attr();
  scale_factors_ = kernel_ptr->get_scales_attr();
  return true;
}

bool UpsampleTrilinear3DGpuKernelMod::Init(const BaseOperatorPtr &base_operator,
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
  return GetUpsampleTrilinear3DAttr(base_operator);
}

int UpsampleTrilinear3DGpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                            const std::vector<KernelTensorPtr> &inputs,
                                            const std::vector<KernelTensorPtr> &outputs,
                                            const std::map<uint32_t, tensor::TensorPtr> &) {
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  ResetResource();
  std::vector<int64_t> input_shape = inputs[kIndex0]->GetShapeVector();
  size_t input_size = std::accumulate(input_shape.begin(), input_shape.end(), t_size_, std::multiplies<size_t>());
  string error_type = "input_dim";
  int exp_shape_size = kIndex5;
  CheckDims(error_type, exp_shape_size, input_shape);
  std::vector<int64_t> output_shape = outputs[kIndex0]->GetShapeVector();
  size_t output_size = std::accumulate(output_shape.begin(), output_shape.end(), t_size_, std::multiplies<size_t>());
  error_type = "output_dim";
  CheckDims(error_type, exp_shape_size, input_shape);
  n_ = input_shape[kIndex0];
  c_ = input_shape[kIndex1];
  input_d_ = input_shape[kIndex2];
  input_h_ = input_shape[kIndex3];
  input_w_ = input_shape[kIndex4];
  output_d_ = output_shape[kIndex2];
  output_h_ = output_shape[kIndex3];
  output_w_ = output_shape[kIndex4];
  exp_shape_size = kIndex3;
  if (!out_spatial_size_me_.empty() || align_corners_ == true) {
    scale_factors_.push_back(ScalingD(input_d_, output_d_, align_corners_));
    scale_factors_.push_back(ScalingD(input_h_, output_h_, align_corners_));
    scale_factors_.push_back(ScalingD(input_w_, output_w_, align_corners_));
  } else {
    scale_factors_.push_back(ScalingS(scale_factors_[kIndex0], kIndex0, output_d_));
    scale_factors_.push_back(ScalingS(scale_factors_[kIndex1], kIndex1, output_h_));
    scale_factors_.push_back(ScalingS(scale_factors_[kIndex2], kIndex2, output_w_));
  }
  input_size_list_.push_back(input_size);
  output_size_list_.push_back(output_size);
  return KRET_OK;
}

template <typename T>
bool UpsampleTrilinear3DGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                   const std::vector<AddressPtr> &workspace,
                                                   const std::vector<AddressPtr> &outputs) {
  auto input = reinterpret_cast<T *>(inputs.at(kIndex0)->addr);
  auto output = reinterpret_cast<T *>(outputs.at(kIndex0)->addr);
  CalUpsampleTrilinear3D(input, n_, c_, input_d_, input_h_, input_w_, output_d_, output_h_, output_w_,
                         scale_factors_[kIndex0], scale_factors_[kIndex1], scale_factors_[kIndex2], align_corners_,
                         output, reinterpret_cast<cudaStream_t>(cuda_stream_));
  return true;
}

std::vector<std::pair<KernelAttr, UpsampleTrilinear3DGpuKernelMod::UpsampleTrilinear3DFunc>>
  UpsampleTrilinear3DGpuKernelMod::func_list_ = {
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
     &UpsampleTrilinear3DGpuKernelMod::LaunchKernel<half>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &UpsampleTrilinear3DGpuKernelMod::LaunchKernel<float>}};
std::vector<KernelAttr> UpsampleTrilinear3DGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, UpsampleTrilinear3DFunc> &pair) { return pair.first; });
  return support_list;
}
}  // namespace kernel
}  // namespace mindspore
