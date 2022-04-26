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

#include "plugin/device/gpu/kernel/nn/dropout_nd_gpu_kernel.h"
#include <functional>
#include <utility>
#include <string>
#include <algorithm>
#include <memory>
#include "include/curand.h"
#include "mindspore/core/ops/dropout_nd.h"
#include "abstract/utils.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/dropout_nd_impl.cuh"

namespace mindspore {
namespace kernel {
void DropoutNDGpuKernelMod::CheckDropOutNdShape() {
  size_t nd_dims = input_shape_.size();
  size_t expected_dims = 0;
  if (kernel_name_ == prim::kPrimDropout2D->name()) {
    // Dropout2D ---> data format NCHW(4 dims)
    expected_dims = 4;
  } else if (kernel_name_ == prim::kPrimDropout3D->name()) {
    // Dropout3D ---> data format NCDHW(5 dims)
    expected_dims = 5;
  } else {
    MS_LOG(EXCEPTION) << "For 'DropoutNd' should only support Dropout2D or Dropout3D, right now, but got "
                      << kernel_name_;
  }
  if (expected_dims != nd_dims) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << " input dims should be " << expected_dims << "D, but got  "
                      << nd_dims << "D.";
  }
}

bool DropoutNDGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs) {
  // A Code Block For getting launch_kernel function.
  {
    kernel_name_ = base_operator->name();
    if (kernel_name_ == prim::kPrimDropout2D->name()) {
      kernel_ptr_ = std::make_shared<ops::Dropout2D>(base_operator->GetPrim());
    } else if (kernel_name_ == prim::kPrimDropout3D->name()) {
      kernel_ptr_ = std::make_shared<ops::Dropout3D>(base_operator->GetPrim());
    } else {
      MS_LOG(ERROR) << "For 'DropoutNDGpuKernelMod' should get Dropout2D or Dropout3D but get invalid kernel name : "
                    << kernel_name_;
      return false;
    }
    auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
    auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
    if (!is_match) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "' does not support this kernel type: " << kernel_attr;
      return false;
    }
    kernel_func_ = func_list_[index].second;
    unit_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex0).first);
  }

  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' got empty inputs or outputs, which is invalid.";
    return false;
  }

  // A Code Block For setting input and output shape.
  {
    input_shape_ = std::vector<size_t>(inputs.at(kIndex0)->GetDeviceShapeAdaptively().begin(),
                                       inputs.at(kIndex0)->GetDeviceShapeAdaptively().end());
    input_elements_ = std::accumulate(input_shape_.begin(), input_shape_.end(), 1, std::multiplies<size_t>());
    is_null_input_ = (input_elements_ == 0);
    if (is_null_input_) {
      InitSizeLists();
      return true;
    }
    outputs_ = outputs;
    CheckDropOutNdShape();
    CheckTensorSize({input_shape_});
    // Get N and C values from 5 dim input tensor
    n_ = input_shape_.at(kDim0);
    c_ = input_shape_.at(kDim1);
    num_chan_ = n_ * c_;
    MS_EXCEPTION_IF_ZERO("num channel", num_chan_);
    num_per_chan_ = input_elements_ / num_chan_;  // number of elements per channel
    InitSizeLists();
  }
  constexpr auto keep_prob = "keep_prob";
  if (!kernel_ptr_->HasAttr(keep_prob)) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "',has no attribute of keep_prob";
  }
  keep_prob_ = GetValue<float>(kernel_ptr_->GetAttr(keep_prob));
  if ((keep_prob_ < 0.0) || (keep_prob_ > 1.0)) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the value of 'keep_prob' should be in range [0.0, 1.0], "
                      << "but got " << keep_prob_;
  }
  if (!states_init_) {
    CHECK_CURAND_RET_WITH_EXCEPT(curandCreateGenerator(&curand_generator_, CURAND_RNG_PSEUDO_DEFAULT),
                                 "Failed to create generator");
    MS_EXCEPTION_IF_NULL(curand_generator_);
    CHECK_CURAND_RET_WITH_EXCEPT(curandSetPseudoRandomGeneratorSeed(curand_generator_, time(NULL)),
                                 "Failed to SetPseudoRandomGeneratorSeed");
    states_init_ = true;
  }
  cudnn_handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCudnnHandle();

  // A Code Block For dealing with input_dynamic_shape.
  {
    if (!is_input_dynamic_shape_.has_value()) {
      bool is_input_dynamic_shape = false;
      for (const auto &input : inputs) {
        auto input_shape = input->GetShapeVector();
        if (std::any_of(input_shape.begin(), input_shape.end(), [](int64_t dim) { return dim < 0; })) {
          is_input_dynamic_shape = true;
          break;
        }
      }
      is_input_dynamic_shape_ = is_input_dynamic_shape;
    }
  }
  return true;
}

bool DropoutNDGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                   const std::vector<KernelTensorPtr> &outputs,
                                   const std::map<uint32_t, tensor::TensorPtr> &) {
  if (is_input_dynamic_shape_.has_value() && is_input_dynamic_shape_.value()) {
    DestroyResource();
    ResetResource();
    return Init(base_operator, inputs, outputs);
  } else {
    kernel_ptr_ = base_operator;
    outputs_ = outputs;
    return true;
  }
}

template <typename T>
bool DropoutNDGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                         const std::vector<AddressPtr> &workspace,
                                         const std::vector<AddressPtr> &outputs) {
  T *input = GetDeviceAddress<T>(inputs, kIndex0);
  T *output = GetDeviceAddress<T>(outputs, kIndex0);
  bool *mask = GetDeviceAddress<bool>(outputs, kIndex1);
  auto *rand_f = GetDeviceAddress<float>(workspace, kIndex0);
  // When keep_prob equal to 0.0, output default to zero, mask default to false.
  if (keep_prob_ == 0.0) {
    CHECK_CUDA_RET_WITH_ERROR_NOTRACE(cudaMemset(output, 0, outputs.at(kIndex0)->size),
                                      "For DropoutNDGpuKernelMod failed to cudaMemset");
    // Default zero to be false.
    CHECK_CUDA_RET_WITH_ERROR_NOTRACE(cudaMemset(mask, 0, outputs.at(kIndex1)->size),
                                      "For DropoutNDGpuKernelMod failed to cudaMemset");
    return true;
  }
  CHECK_CURAND_RET_WITH_EXCEPT(curandSetStream(curand_generator_, reinterpret_cast<cudaStream_t>(cuda_stream_)),
                               "For DropoutNDGpuKernelMod failed to set stream for generator");
  // For curandGen only supports float or double.
  // To generate random float data for every channel.
  CHECK_CURAND_RET_WITH_EXCEPT(curandGenerateUniform(curand_generator_, rand_f, num_chan_),
                               "For DropoutNDGpuKernelMod failed to generate uniform");
  DropoutNDForward(input, mask, output, rand_f, input_elements_, keep_prob_, num_per_chan_,
                   reinterpret_cast<cudaStream_t>(cuda_stream_));

  return true;
}

std::vector<std::pair<KernelAttr, DropoutNDGpuKernelMod::DropoutNdFunc>> DropoutNDGpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeBool),
   &DropoutNDGpuKernelMod::LaunchKernel<int8_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeBool),
   &DropoutNDGpuKernelMod::LaunchKernel<int16_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeBool),
   &DropoutNDGpuKernelMod::LaunchKernel<int>},
  {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeBool),
   &DropoutNDGpuKernelMod::LaunchKernel<int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeBool),
   &DropoutNDGpuKernelMod::LaunchKernel<half>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeBool),
   &DropoutNDGpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeBool),
   &DropoutNDGpuKernelMod::LaunchKernel<double>}};

std::vector<KernelAttr> DropoutNDGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, DropoutNdFunc> &pair) { return pair.first; });
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, Dropout2D, DropoutNDGpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, Dropout3D, DropoutNDGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
