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

#include "plugin/device/cpu/kernel/instance_norm_v2_cpu_kernel.h"
#include <algorithm>
#include <string>
#include <vector>
#include "kernel/common_utils.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "plugin/device/cpu/kernel/eigen/eigen_common_utils.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr auto kInstanceNormV2InputsNum = 5;
constexpr auto kInstanceNormV2OutputNum = 3;
// GRAIN_SIZE for Parallel
constexpr size_t kGrainSize = 4 * 1024;
constexpr float float_init_zero = 0.0;
constexpr float float_init_one = 1.0;
constexpr double double_init_zero = 0.0;
constexpr double double_init_one = 1.0;
constexpr int32_t int32_init_one = 1;
constexpr int64_t int64_init_one = 1;
constexpr float momentum_min = 0.0;
constexpr float momentum_max = 1.0;

template <typename T>
struct InvStd {
  T operator()(T var, double epsilon) const {
    T invstd = 0;
    if (var != static_cast<T>(0) || epsilon != static_cast<T>(0)) {
      invstd = static_cast<T>(int32_init_one) / std::sqrt(var + epsilon);
    }
    return invstd;
  }
};
}  // namespace

template <typename T>
void InstanceNormV2CpuKernelMod::CollectStatsKernel(const kernel::AddressPtr &x, float *_mean_, float *_var_sum) const {
  const int64_t batch = x_shape_4d_[kIndex0];
  const int64_t channel = x_shape_4d_[kIndex3];
  const int64_t image_size = x_shape_4d_[kIndex1] * x_shape_4d_[kIndex2];
  MS_EXCEPTION_IF_ZERO("channel", channel);
  MS_EXCEPTION_IF_ZERO("image_size", image_size);
  // cast (B, H, W, C) to (B, H*W, C)
  std::vector<int64_t> shape_3d = {batch, image_size, channel};
  auto x_3d = EigenTensor(shape_3d, x->addr).tensor<T, kDim3>();
  auto loop_batch = [&](int64_t begin, int64_t end) {
    for (int64_t batch_idx = begin; batch_idx < end; ++batch_idx) {
      for (int64_t channel_idx = 0; channel_idx < channel; ++channel_idx) {
        // compute mean per input
        double sum = double_init_zero;
        for (int64_t idx = 0; idx < image_size; ++idx) {
          sum += static_cast<double>(x_3d(batch_idx, idx, channel_idx));
        }
        double cur_mean = sum / static_cast<double>(image_size);
        _mean_[batch_idx * channel + channel_idx] = static_cast<float>(cur_mean);
        // compute variance per input
        double cur_var_sum = double_init_zero;
        for (int64_t idx = 0; idx < image_size; ++idx) {
          double cur_piexl = static_cast<double>(x_3d(batch_idx, idx, channel_idx));
          cur_var_sum += (cur_piexl - cur_mean) * (cur_piexl - cur_mean);
        }
        _var_sum[batch_idx * channel + channel_idx] = static_cast<float>(cur_var_sum);
      }
    }
  };
  float block_size = std::max(float_init_one, static_cast<float>(kGrainSize / (channel * image_size)));
  CPUKernelUtils::ParallelFor(loop_batch, batch, block_size);
}

template <typename T, template <typename S> class VarTransform>
void InstanceNormV2CpuKernelMod::UpdateStatsTemplate(const std::vector<kernel::AddressPtr> &inputs,
                                                     const std::vector<kernel::AddressPtr> &outputs) {
  std::vector<float> _var_sum(instance_num_, float_init_zero);
  std::vector<float> _mean_(instance_num_, float_init_zero);
  CollectStatsKernel<T>(inputs[kIndex0], _mean_.data(), _var_sum.data());
  const int64_t image_size = x_shape_4d_[kIndex1] * x_shape_4d_[kIndex2];
  MS_EXCEPTION_IF_ZERO("image_size", image_size);
  std::vector<int64_t> batch_channels_1d_ = {batch_channels_2d_.front() * batch_channels_2d_.back()};
  auto running_mean_vec = EigenTensor(batch_channels_1d_, inputs[kIndex3]->addr).vec<float>();
  auto running_var_vec = EigenTensor(batch_channels_1d_, inputs[kIndex4]->addr).vec<float>();
  auto save_mean_vec = EigenTensor(batch_channels_1d_, outputs[kIndex1]->addr).vec<float>();
  auto save_var_vec = EigenTensor(batch_channels_1d_, outputs[kIndex2]->addr).vec<float>();

  auto loop_momentum = [&](int64_t begin, int64_t end) {
    for (int64_t idx = begin; idx < end; ++idx) {
      save_mean_vec(idx) = _mean_[idx];
      save_var_vec(idx) =
        VarTransform<double>{}(static_cast<double>(_var_sum[idx]) / static_cast<double>(image_size), epsilon_);
      running_mean_vec(idx) =
        static_cast<float>(momentum_ * static_cast<double>(_mean_[idx]) +
                           (double_init_one - momentum_) * static_cast<double>(running_mean_vec(idx)));
      double unbiased_var = double_init_zero;
      if (image_size - int64_init_one == 0) {
        unbiased_var = static_cast<double>(_var_sum[idx]);
      } else {
        unbiased_var = static_cast<double>(_var_sum[idx]) / static_cast<double>(image_size - int64_init_one);
      }
      running_var_vec(idx) = static_cast<float>(momentum_ * unbiased_var + (double_init_one - momentum_) *
                                                                             static_cast<double>(running_var_vec(idx)));
    }
  };
  CPUKernelUtils::ParallelFor(loop_momentum, instance_num_, static_cast<float>(kGrainSize));
}

void InstanceNormV2CpuKernelMod::CollectLinearAndConstant(const typename TTypes<float>::Vec &gamma,
                                                          const typename TTypes<float>::Vec &beta,
                                                          const typename TTypes<float>::Vec &running_mean,
                                                          const typename TTypes<float>::Vec &running_var,
                                                          const typename TTypes<float>::Vec &save_mean,
                                                          const typename TTypes<float>::Vec &save_invstd,
                                                          float *_alpha_, float *_beta_) {
  auto loop_instance = [&](int64_t begin, int64_t end) {
    for (int64_t idx = begin; idx < end; ++idx) {
      float mean = float_init_zero, invstd = float_init_zero;
      if (is_training_) {
        mean = save_mean(idx);
        invstd = save_invstd(idx);
      } else {
        mean = running_mean(idx);
        float _std_ = std::sqrt(running_var(idx) + static_cast<float>(epsilon_));
        MS_EXCEPTION_IF_ZERO("_std_", _std_);
        invstd = float_init_one / _std_;
      }
      _alpha_[idx] = invstd * gamma(idx);
      _beta_[idx] = beta(idx) - mean * _alpha_[idx];
    }
  };
  CPUKernelUtils::ParallelFor(loop_instance, instance_num_, static_cast<float>(kGrainSize));
}

template <typename T>
void InstanceNormV2CpuKernelMod::TransformInput(const std::vector<kernel::AddressPtr> &inputs,
                                                const std::vector<kernel::AddressPtr> &outputs) {
  const int64_t batch = x_shape_4d_[kIndex0];
  const int64_t channel = x_shape_4d_[kIndex3];
  const int64_t image_size = x_shape_4d_[kIndex1] * x_shape_4d_[kIndex2];
  std::vector<float> _alpha_(instance_num_, float_init_zero);
  std::vector<float> _beta_(instance_num_, float_init_zero);
  std::vector<int64_t> batch_channels_1d_ = {batch_channels_2d_.front() * batch_channels_2d_.back()};
  auto gamma = EigenTensor(batch_channels_1d_, inputs[kIndex1]->addr).vec<float>();
  auto beta = EigenTensor(batch_channels_1d_, inputs[kIndex2]->addr).vec<float>();
  auto running_mean = EigenTensor(batch_channels_1d_, inputs[kIndex3]->addr).vec<float>();
  auto running_var = EigenTensor(batch_channels_1d_, inputs[kIndex4]->addr).vec<float>();
  auto save_mean = EigenTensor(batch_channels_1d_, outputs[kIndex1]->addr).vec<float>();
  auto save_invstd = EigenTensor(batch_channels_1d_, outputs[kIndex2]->addr).vec<float>();
  CollectLinearAndConstant(gamma, beta, running_mean, running_var, save_mean, save_invstd, _alpha_.data(),
                           _beta_.data());
  // cast (B, H, W, C) to (B, H*W, C)
  std::vector<int64_t> shape_3d = {batch, image_size, channel};
  auto x_3d = EigenTensor(shape_3d, inputs[kIndex0]->addr).tensor<T, kDim3>();
  auto y_3d = EigenTensor(shape_3d, outputs[kIndex0]->addr).tensor<T, kDim3>();
  // Apply the linear terms to the input,
  auto loop_transform = [&](int64_t begin, int64_t end) {
    for (int64_t batch_idx = begin; batch_idx < end; ++batch_idx) {
      for (int64_t idx = 0; idx < image_size; ++idx) {
        for (int64_t channel_idx = 0; channel_idx < channel; ++channel_idx) {
          float alpha = _alpha_[batch_idx * channel + channel_idx];
          float beta = _beta_[batch_idx * channel + channel_idx];
          y_3d(batch_idx, idx, channel_idx) =
            static_cast<T>(alpha * static_cast<float>(x_3d(batch_idx, idx, channel_idx)) + beta);
        }
      }
    }
  };
  float block_size = std::max(float_init_one, static_cast<float>(kGrainSize / (channel * image_size)));
  CPUKernelUtils::ParallelFor(loop_transform, batch, block_size);
}

bool InstanceNormV2CpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                      const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  auto prim = base_operator->GetPrim();
  MS_EXCEPTION_IF_NULL(prim);

  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInstanceNormV2InputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kInstanceNormV2OutputNum, kernel_name_);

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto is_match = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match.first) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
  }

  in_type_ = inputs[kIndex0]->GetDtype();
  is_training_ = GetValue<bool>(prim->GetAttr(kAttrIsTraining));
  momentum_ = GetValue<float>(prim->GetAttr(kAttrMomentum));
  epsilon_ = GetValue<float>(prim->GetAttr(kAttrEpsilon));
  if (momentum_ > momentum_max || momentum_ < momentum_min) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_
                             << "momentum value should be in [0, 1], but get momentum = " << momentum_ << ".";
  }
  if (epsilon_ >= momentum_max || epsilon_ < momentum_min) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_
                             << "epsilon value should be in [0, 1), but get epsilon = " << epsilon_ << ".";
  }

  return true;
}

int InstanceNormV2CpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                       const std::vector<KernelTensorPtr> &outputs,
                                       const std::map<uint32_t, tensor::TensorPtr> &) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }

  std::vector<int64_t> x_shape = inputs[kIndex0]->GetShapeVector();
  std::vector<int64_t> batch_channels = inputs[kIndex1]->GetShapeVector();

  if (x_shape.size() != kDim4 && x_shape.size() != kDim5) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', the dimension of 'x' should be 4D or 5D, but got "
                             << x_shape.size() << "D.";
  }
  input_x_is_4d_ = (x_shape.size() == kDim4);
  // Format NCHW could be considered as a situation of format NC1HWC0 when C0 = 1.
  if (input_x_is_4d_) {
    // extern (N, C, H, W) to (N, C, H, W, 1)
    x_shape.push_back(SizeToLong(kDim1));
    // extern (N, C, 1, 1) to (N, C1=C, 1, 1, C0=1)
    batch_channels.push_back(SizeToLong(kDim1));
  }
  // consider (N, C1, H, W, C0) as (N*C1, H, W, C0), similar to (N, H, W, C)
  x_shape_4d_ = {x_shape[kIndex0] * x_shape[kIndex1], x_shape[kIndex2], x_shape[kIndex3], x_shape[kIndex4]};
  // consider (N, C1, 1, 1 C0) as (N*C1, 1, 1, C0), similar to (N, 1, 1, C)
  batch_channels_2d_ = {batch_channels[kIndex0] * batch_channels[kIndex1], batch_channels[kIndex4]};
  instance_num_ = CPUKernelUtils::CalcElementNum(batch_channels_2d_);

  return KRET_OK;
}

bool InstanceNormV2CpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                        const std::vector<kernel::AddressPtr> &,
                                        const std::vector<kernel::AddressPtr> &outputs) {
  bool res = false;
  switch (in_type_) {
    case kNumberTypeFloat16:
      res = LaunchKernel<float16>(inputs, outputs);
      break;
    case kNumberTypeFloat32:
      res = LaunchKernel<float>(inputs, outputs);
      break;
    default:
      MS_EXCEPTION(TypeError) << "For '" << kernel_name_ << "', the dtype of 'x' should be float16, float32, but got "
                              << TypeIdLabel(in_type_);
  }
  return res;
}

template <typename T>
bool InstanceNormV2CpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                              const std::vector<kernel::AddressPtr> &outputs) {
  auto batch_mean_ptr = static_cast<float *>(outputs[kIndex1]->addr);
  auto batch_var_ptr = static_cast<float *>(outputs[kIndex2]->addr);
  (void)std::fill_n(batch_mean_ptr, CPUKernelUtils::CalcElementNum(batch_channels_2d_), float_init_zero);
  (void)std::fill_n(batch_var_ptr, CPUKernelUtils::CalcElementNum(batch_channels_2d_), float_init_zero);

  if (is_training_) {
    // UpdateStatsTemplate to init save_mean and save_var
    UpdateStatsTemplate<T, InvStd>(inputs, outputs);
  }
  TransformInput<T>(inputs, outputs);
  return true;
}

std::vector<KernelAttr> InstanceNormV2CpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list = {KernelAttr()
                                                   .AddInputAttr(kNumberTypeFloat16)
                                                   .AddInputAttr(kNumberTypeFloat32)
                                                   .AddInputAttr(kNumberTypeFloat32)
                                                   .AddInputAttr(kNumberTypeFloat32)
                                                   .AddInputAttr(kNumberTypeFloat32)
                                                   .AddOutputAttr(kNumberTypeFloat16)
                                                   .AddOutputAttr(kNumberTypeFloat32)
                                                   .AddOutputAttr(kNumberTypeFloat32),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeFloat32)
                                                   .AddInputAttr(kNumberTypeFloat32)
                                                   .AddInputAttr(kNumberTypeFloat32)
                                                   .AddInputAttr(kNumberTypeFloat32)
                                                   .AddInputAttr(kNumberTypeFloat32)
                                                   .AddOutputAttr(kNumberTypeFloat32)
                                                   .AddOutputAttr(kNumberTypeFloat32)
                                                   .AddOutputAttr(kNumberTypeFloat32)};
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, InstanceNormV2, InstanceNormV2CpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
