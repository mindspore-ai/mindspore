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
#include "plugin/device/gpu/kernel/nn/bias_dropout_add_gpu_kernel.h"

#include <algorithm>
#include <string>
#include <vector>
#include <map>
#include <utility>
#include <functional>
#include "abstract/utils.h"
#include "ops/fusion/bias_dropout_add_fusion.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/bias_dropout_add_impl.cuh"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kInputNum = 3;
constexpr size_t kInputXIndex = 0;
constexpr size_t kInputBiasIndex = 1;
constexpr size_t kInputResidualIndex = 2;

constexpr size_t kOutputNum = 2;
constexpr size_t kOutputYIndex = 0;
constexpr size_t kOutputMaskIndex = 1;
}  // namespace

bool BiasDropoutAddGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                      const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  if (inputs.size() != kInputNum || outputs.size() != kOutputNum) {
    MS_LOG(ERROR) << kernel_name_ << ": input and output size should be " << kInputNum << " and " << kOutputNum
                  << ", but get " << inputs.size() << " and " << outputs.size();
    return false;
  }

  auto kernel_ptr = std::dynamic_pointer_cast<ops::BiasDropoutAdd>(base_operator);
  if (kernel_ptr == nullptr) {
    MS_LOG(ERROR) << "Cast BiasDropoutAdd failed!";
    return false;
  }

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' does not support this kernel type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;

  keep_prob_ = kernel_ptr->get_keep_prob();
  int64_t seed = kernel_ptr->get_seed0();
  if (seed == 0) {
    seed = kernel_ptr->get_seed1();
    if (seed == 0) {
      seed = time(NULL);
    }
  }
  seed_ = static_cast<uint64_t>(seed);
  return true;
}

int BiasDropoutAddGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                       const std::vector<KernelTensorPtr> &outputs,
                                       const std::map<uint32_t, tensor::TensorPtr> &) {
  int ret = KernelMod::Resize(base_operator, inputs, outputs);
  if (ret != 0) {
    return ret;
  }
  if (input_size_list_.size() != kInputNum || output_size_list_.size() != kOutputNum) {
    MS_LOG(ERROR) << kernel_name_ << " resize : input and output size should be " << kInputNum << " and " << kOutputNum
                  << ", but get " << input_size_list_.size() << " and " << output_size_list_.size();
    return KRET_RESIZE_FAILED;
  }
  auto x_shape = inputs[kInputXIndex]->GetShapeVector();
  num_count_ = 1;
  n_strides_ = 1;
  channel_strides_ = 1;
  for (size_t i = 0; i < x_shape.size(); ++i) {
    auto dim = x_shape[i];
    if (dim < 0) {
      dim = 0;
    }
    auto dim_length = LongToSize(dim);
    num_count_ *= dim_length;
    if (i > 0) {
      n_strides_ *= dim_length;
    }
    if (i > 1) {
      channel_strides_ *= dim_length;
    }
  }
  return KRET_OK;
}

template <typename T>
bool BiasDropoutAddGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                              const std::vector<AddressPtr> &outputs) {
  T *x = GetDeviceAddress<T>(inputs, kInputXIndex);
  T *bias = GetDeviceAddress<T>(inputs, kInputBiasIndex);
  T *residual = GetDeviceAddress<T>(inputs, kInputResidualIndex);
  T *y = GetDeviceAddress<T>(outputs, kOutputYIndex);
  T *mask = GetDeviceAddress<T>(outputs, kOutputMaskIndex);
  BiasDropoutAdd(x, bias, residual, y, mask, num_count_, n_strides_, channel_strides_, keep_prob_, seed_, seed_offset_,
                 cuda_stream_);
  seed_offset_ += num_count_;
  return true;
}

std::vector<std::pair<KernelAttr, BiasDropoutAddGpuKernelMod::KernelFunc>> BiasDropoutAddGpuKernelMod::func_list_ = {
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16),
   &BiasDropoutAddGpuKernelMod::LaunchKernel<half>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   &BiasDropoutAddGpuKernelMod::LaunchKernel<float>}};

std::vector<KernelAttr> BiasDropoutAddGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, KernelFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, BiasDropoutAdd, BiasDropoutAddGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
