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

#include "plugin/device/gpu/kernel/math/bernoulli_gpu_kernel.h"
#include <functional>
#include <utility>
#include <string>
#include <algorithm>
#include <memory>
#include "include/curand.h"
#include "mindspore/core/ops/bernoulli.h"
#include "abstract/utils.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/bernoulli_impl.cuh"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace kernel {
constexpr int MAX_DIMS = 7;
bool BernoulliGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it got empty inputs or outputs, which is invalid.";
    return false;
  }
  kernel_ptr_ = std::make_shared<ops::Bernoulli>(base_operator->GetPrim());
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' does not support this kernel type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  unit_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex0).dtype);
  p_unit_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex1).dtype);
  if (!states_init_) {
    constexpr auto seed_str = "seed";
    if (!kernel_ptr_->HasAttr(seed_str)) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', has no attribute of seed";
    }
    seed_ = GetValue<int64_t>(kernel_ptr_->GetAttr(seed_str));
    if (seed_ < 0 && seed_ != -1) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', the value of 'seed' must be -1 or positive integer, "
                    << "but got " << seed_;
      return false;
    }
    states_init_ = true;
  }
  cudnn_handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCudnnHandle();
  return true;
}

int BernoulliGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                  const std::vector<KernelTensorPtr> &outputs,
                                  const std::map<uint32_t, tensor::TensorPtr> &) {
  ResetResource();
  for (const auto &input : inputs) {
    // If any input shape contains -1, means input shape is dynamic, so just return do nothing.
    auto input_shape = input->GetShapeVector();
    if (!IsValidShape(input_shape)) {
      return KRET_UNKNOWN_SHAPE;
    }
  }
  auto x_long_shape = inputs.at(kIndex0)->GetShapeVector();
  auto p_long_shape = inputs.at(kIndex1)->GetShapeVector();
  std::vector<size_t> x_shape;
  std::vector<size_t> p_shape;
  (void)std::transform(x_long_shape.begin(), x_long_shape.end(), std::back_inserter(x_shape), LongToSize);
  (void)std::transform(p_long_shape.begin(), p_long_shape.end(), std::back_inserter(p_shape), LongToSize);
  need_broadcast_ = common::AnfAlgo::IsTensorBroadcast(x_shape, p_shape);
  if (x_shape.size() > MAX_DIMS || p_shape.size() > MAX_DIMS) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of input cannot be greater than " << MAX_DIMS
                      << ", but got x: " << x_shape.size() << ", p: " << p_shape.size();
  }
  if (p_shape.size() > x_shape.size()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the shape of input p " << p_shape << " cannot be broadcast to"
                      << " the shape of input x " << x_shape;
  }
  x_shape_.resize(MAX_DIMS, 1);
  p_shape_.resize(MAX_DIMS, 1);

  for (size_t i = 0; i < x_shape.size(); i++) {
    if (need_broadcast_) {
      x_shape_[i] = x_shape[i];
    }
    x_count_ *= x_shape[i];
  }
  int p_offset = x_shape.size() - p_shape.size();
  for (size_t j = 0; j < p_shape.size(); j++) {
    if (need_broadcast_) {
      p_shape_[j + p_offset] = p_shape[j];
    }
    p_count_ *= p_shape[j];
  }
  if (need_broadcast_) {
    for (size_t k = 0; k < x_shape.size(); k++) {
      if (x_shape_[k] != p_shape_[k] && p_shape_[k] != 1) {
        MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the shape of input p " << p_shape
                          << " cannot be broadcast to the shape of input x " << x_shape;
      }
    }
  }
  size_t input_size = x_count_ * unit_size_;
  input_size_list_.emplace_back(input_size);
  size_t p_size = p_count_ * p_unit_size_;
  input_size_list_.emplace_back(p_size);
  output_size_list_.emplace_back(input_size);
  size_t workspace_size = 0;
  workspace_size_list_.emplace_back(workspace_size);
  return KRET_OK;
}

void BernoulliGpuKernelMod::ResetResource() noexcept {
  is_null_input_ = false;
  x_count_ = 1;
  p_count_ = 1;
  x_shape_.clear();
  p_shape_.clear();
  input_size_list_.clear();
  output_size_list_.clear();
  workspace_size_list_.clear();
}

template <typename T, typename S>
bool BernoulliGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                         const std::vector<AddressPtr> &workspace,
                                         const std::vector<AddressPtr> &outputs) {
  T *p = GetDeviceAddress<T>(inputs, kIndex1);
  S *y = GetDeviceAddress<S>(outputs, kIndex0);
  uint64_t seed;
  if (seed_ == -1) {
    seed = static_cast<uint64_t>(time(NULL));
  } else {
    seed = static_cast<uint64_t>(seed_);
  }
  if (need_broadcast_) {
    auto status = BroadcastBernoulliForward(x_shape_, p_shape_, p, y, seed, x_count_, device_id_,
                                            reinterpret_cast<cudaStream_t>(cuda_stream_));
    CHECK_CUDA_LAUNCH_STATUS(status, kernel_name_);
  } else {
    auto status = BernoulliForward(p, y, seed, x_count_, device_id_, reinterpret_cast<cudaStream_t>(cuda_stream_));
    CHECK_CUDA_LAUNCH_STATUS(status, kernel_name_);
  }
  return true;
}

std::vector<std::pair<KernelAttr, BernoulliGpuKernelMod::BernoulliFunc>> BernoulliGpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt8),
   &BernoulliGpuKernelMod::LaunchKernel<float, int8_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeUInt8),
   &BernoulliGpuKernelMod::LaunchKernel<float, uint8_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt16),
   &BernoulliGpuKernelMod::LaunchKernel<float, int16_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt32),
   &BernoulliGpuKernelMod::LaunchKernel<float, int>},
  {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt64),
   &BernoulliGpuKernelMod::LaunchKernel<float, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeBool).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeBool),
   &BernoulliGpuKernelMod::LaunchKernel<float, bool>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
   &BernoulliGpuKernelMod::LaunchKernel<float, float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat64),
   &BernoulliGpuKernelMod::LaunchKernel<float, double>},
  {KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeInt8),
   &BernoulliGpuKernelMod::LaunchKernel<double, int8_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeUInt8),
   &BernoulliGpuKernelMod::LaunchKernel<double, uint8_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeInt16),
   &BernoulliGpuKernelMod::LaunchKernel<double, int16_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeInt32),
   &BernoulliGpuKernelMod::LaunchKernel<double, int>},
  {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeInt64),
   &BernoulliGpuKernelMod::LaunchKernel<double, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeBool).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeBool),
   &BernoulliGpuKernelMod::LaunchKernel<double, bool>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat32),
   &BernoulliGpuKernelMod::LaunchKernel<double, float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
   &BernoulliGpuKernelMod::LaunchKernel<double, double>}};

std::vector<KernelAttr> BernoulliGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, BernoulliFunc> &pair) { return pair.first; });
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, Bernoulli, BernoulliGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
