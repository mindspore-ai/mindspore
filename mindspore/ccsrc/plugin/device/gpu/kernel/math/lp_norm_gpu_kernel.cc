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

#include "plugin/device/gpu/kernel/math/lp_norm_gpu_kernel.h"
#include <functional>
#include <utility>
#include <string>
#include <algorithm>
#include <set>
#include "abstract/utils.h"
#include "kernel/common_utils.h"
#include "mindspore/core/ops/lp_norm.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/lp_norm_impl.cuh"

namespace mindspore {
namespace kernel {
void LpNormGpuKernelMod::GetLpNormAttr() {
  const std::string axis = "axis";
  if (!kernel_ptr_->HasAttr(axis)) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "' has no kernel attribute: " << axis;
  }
  axis_ = GetValue<std::vector<int64_t>>(kernel_ptr_->GetAttr(axis));
  const std::string p = "p";
  if (!kernel_ptr_->HasAttr(p)) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "' has no kernel attribute: " << p;
  }
  p_ = static_cast<float>(GetValue<int64_t>(kernel_ptr_->GetAttr(p)));
  if (p_ == 0.0f) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "''s op attribute " << p << " equals to zero is invalid.";
  }
  const std::string epsilon = "epsilon";
  if (!kernel_ptr_->HasAttr(epsilon)) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "' has no kernel attribute: " << epsilon;
  }
  epsilon_ = GetValue<float>(kernel_ptr_->GetAttr(epsilon));
}

bool LpNormGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                              const std::vector<KernelTensorPtr> &outputs) {
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' got empty inputs or outputs, which is invalid.";
    return false;
  }

  // A Code Block For getting launch_kernel function.
  {
    kernel_ptr_ = std::make_shared<ops::LpNorm>(base_operator->GetPrim());
    kernel_name_ = kernel_ptr_->name();
    auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
    auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
    if (!is_match) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "' does not support this kernel type: " << kernel_attr;
      return false;
    }
    kernel_func_ = func_list_[index].second;
    unit_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex0).first);
  }

  GetLpNormAttr();

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
    output_shape_ = std::vector<size_t>(outputs.at(kIndex0)->GetDeviceShapeAdaptively().begin(),
                                        outputs.at(kIndex0)->GetDeviceShapeAdaptively().end());

    std::vector<size_t> output_shape;
    // Ignore dim equal to one.
    std::copy_if(output_shape_.begin(), output_shape_.end(), std::back_inserter(output_shape),
                 [](size_t dim) { return dim != 1; });
    output_shape_ = output_shape;
    std::set<size_t> axis_set(axis_.begin(), axis_.end());
    for (size_t i = 0; i < input_shape_.size(); ++i) {
      if (!axis_set.count(i)) {
        output_axis_.emplace_back(i);
      }
    }
    output_stride_.resize(output_shape_.size());
    output_stride_[output_stride_.size() - 1] = 1;
    for (int i = static_cast<int>(output_stride_.size() - 2); i >= 0; --i) {
      output_stride_[i] = output_stride_[i + 1] * output_shape[i + 1];
    }
    output_elements_ = std::accumulate(output_shape_.begin(), output_shape_.end(), 1, std::multiplies<size_t>());
    InitSizeLists();
  }

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

bool LpNormGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                const std::vector<KernelTensorPtr> &outputs,
                                const std::map<uint32_t, tensor::TensorPtr> &others) {
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
bool LpNormGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                      const std::vector<AddressPtr> &outputs) {
  auto input = reinterpret_cast<T *>(inputs.at(kIndex0)->addr);

  auto device_input_shape = reinterpret_cast<size_t *>(workspace.at(kIndex0)->addr);
  auto device_axis_output = reinterpret_cast<size_t *>(workspace.at(kIndex1)->addr);
  auto device_output_stride = reinterpret_cast<size_t *>(workspace.at(kIndex2)->addr);
  auto output = reinterpret_cast<T *>(outputs.at(kIndex0)->addr);

  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(device_input_shape, &input_shape_[0], input_shape_.size() * sizeof(size_t), cudaMemcpyHostToDevice,
                    reinterpret_cast<cudaStream_t>(cuda_stream_)),
    "cudaMemcpyAsync input_shape_ failed");

  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(device_axis_output, &output_axis_[0], output_axis_.size() * sizeof(size_t), cudaMemcpyHostToDevice,
                    reinterpret_cast<cudaStream_t>(cuda_stream_)),
    "cudaMemcpyAsync output_axis_ failed");

  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(device_output_stride, &output_stride_[0], output_stride_.size() * sizeof(size_t),
                    cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(cuda_stream_)),
    "cudaMemcpyAsync output_shape_ failed");

  // The workspace for device output high precision.
  if constexpr (std::is_same_v<T, half>) {
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(cuda_stream_)),
                                       "cudaStremSynchronize failed");
    constexpr auto high_precision_unit = 2;
    size_t device_output_stride_size = output_elements_ * unit_size_ * high_precision_unit;
    float *middle_output = nullptr;
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMalloc(&middle_output, device_output_stride_size),
                                       "cudaMalloc output_shape_ failed");
    CalLpNorm(input, device_input_shape, input_shape_.size(), input_elements_, device_axis_output, device_output_stride,
              output_axis_.size(), output_elements_, p_, epsilon_, middle_output, output,
              reinterpret_cast<cudaStream_t>(cuda_stream_));
  } else {
    CalLpNorm(input, device_input_shape, input_shape_.size(), input_elements_, device_axis_output, device_output_stride,
              output_axis_.size(), output_elements_, p_, epsilon_, nullptr, output,
              reinterpret_cast<cudaStream_t>(cuda_stream_));
  }

  return true;
}

std::vector<std::pair<KernelAttr, LpNormGpuKernelMod::LpNormFunc>> LpNormGpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
   &LpNormGpuKernelMod::LaunchKernel<half>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
   &LpNormGpuKernelMod::LaunchKernel<float>}};

std::vector<KernelAttr> LpNormGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, LpNormFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, LpNorm, LpNormGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
