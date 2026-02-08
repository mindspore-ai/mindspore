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

#include "kernel/gpu/cudnn/instance_norm_gpu_kernel.h"
#include <map>
#include <utility>
#include "mindspore/ops/infer/instance_norm.h"

namespace mindspore {
namespace kernel {
namespace {
using KernelRunFunc = InstanceNormGpuKernelMod::KernelRunFunc;
constexpr auto kNCDims = 2;
}  // namespace
bool InstanceNormGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                    const std::vector<KernelTensor *> &outputs) {
  handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCudnnHandle();
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateTensorDescriptor(&x_desc_),
                                      "For 'InstanceNormGpuKernelMod', it create x desc failed");
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateTensorDescriptor(&y_desc_),
                                      "For 'InstanceNormGpuKernelMod', it create y desc failed");
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateTensorDescriptor(&scale_bias_mean_var_desc_),
                                      "For 'InstanceNormGpuKernelMod', it create para desc failed");
  if (primitive_->HasAttr(ops::kBatchRank)) {
    batch_rank_ = static_cast<size_t>(GetValue<int64_t>(primitive_->GetAttr(ops::kBatchRank)));
  }
  exp_avg_factor_ = static_cast<double>(GetValue<float>(primitive_->GetAttr(kAttrMomentum)));
  epsilon_ = static_cast<double>(GetValue<float>(primitive_->GetAttr(kAttrEpsilon)));

  cudnn_data_type_ = GetCudnnDataType(TypeIdLabel(inputs[kIndex0]->dtype_id()));

  if (!MatchKernelFunc(kernel_name_, inputs, outputs)) {
    return false;
  }

  return true;
}

int InstanceNormGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                     const std::vector<KernelTensor *> &outputs) {
  if (int ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  auto input_shape = inputs[kIndex0]->GetShapeVector();
  is_null_input_ = CHECK_SHAPE_NULL(input_shape, kernel_name_, "input_x");
  if (is_null_input_) {
    return KRET_OK;
  }

  batch_rank_cum_ =
    std::accumulate(input_shape.begin(), input_shape.begin() + batch_rank_, int64_t(1), std::multiplies{});
  batch_ = LongToSize(input_shape[batch_rank_ + kIndex0]);
  channel_ = LongToSize(input_shape[batch_rank_ + kIndex1]);

  CheckTensorSize({input_shape});
  const int batch = 1;
  const int channel = SizeToInt(batch_) * SizeToInt(channel_);
  const int height = 1;
  const int width =
    std::accumulate(input_shape.begin() + batch_rank_ + kNCDims, input_shape.end(), int64_t(1), std::multiplies{});

  input_offset_ = channel * width;
  para_offset_ = channel_;
  updated_para_offset_ = channel;

  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
    cudnnSetTensor4dDescriptor(x_desc_, CUDNN_TENSOR_NCHW, cudnn_data_type_, batch, channel, height, width),
    "For 'InstanceNormGpuKernelMod', it set x desc failed");

  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
    cudnnSetTensor4dDescriptor(y_desc_, CUDNN_TENSOR_NCHW, cudnn_data_type_, batch, channel, height, width),
    "For 'InstanceNormGpuKernelMod', it set y desc failed");

  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
    cudnnSetTensor4dDescriptor(scale_bias_mean_var_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, channel, 1, 1),
    "For 'InstanceNormGpuKernelMod', it set para desc failed");

  size_t para_size = 0;

  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnGetTensorSizeInBytes(scale_bias_mean_var_desc_, &para_size),
                                      "For 'InstanceNormGpuKernelMod', it get para size failed");
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
    cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(handle_, mode_, bn_ops_, x_desc_, z_desc_, y_desc_,
                                                             scale_bias_mean_var_desc_, nullptr, &workspace_size_),
    "For 'InstanceNormGpuKernelMod', it launch cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize failed");

  workspace_size_list_.clear();
  workspace_size_list_ = {
    para_size,  // ws gamma
    para_size,  // ws beta
    para_size,  // ws mean
    para_size,  // ws variance
    workspace_size_,
  };
  return KRET_OK;
}

template <typename T>
bool InstanceNormGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                            const std::vector<KernelTensor *> &workspace,
                                            const std::vector<KernelTensor *> &outputs) {
  auto x_addr = GetDeviceAddress<T>(inputs, kIndex0);
  auto gamma_addr = GetDeviceAddress<float>(inputs, kIndex1);
  auto beta_addr = GetDeviceAddress<float>(inputs, kIndex2);
  auto runing_mean_addr = GetDeviceAddress<float>(inputs, kIndex3);
  auto runnig_variance_addr = GetDeviceAddress<float>(inputs, kIndex4);
  T *z = nullptr;

  auto y_addr = GetDeviceAddress<T>(outputs, kIndex0);
  auto save_mean_addr = GetDeviceAddress<float>(outputs, kIndex1);
  auto save_variance_addr = GetDeviceAddress<float>(outputs, kIndex2);

  float *ws_gamma = GetDeviceAddress<float>(workspace, kIndex0);
  float *ws_beta = GetDeviceAddress<float>(workspace, kIndex1);
  float *ws_mean = GetDeviceAddress<float>(workspace, kIndex2);
  float *ws_var = GetDeviceAddress<float>(workspace, kIndex3);
  T *workspace_addr = GetPossiblyNullDeviceAddress<T>(workspace, kIndex4);

  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnSetStream(handle_, stream_ptr_),
                                      "For 'InstanceNormGpuKernelMod', cudnnSetStream failed.")

  const float alpha = 1;
  const float beta = 0;

  for (size_t i = 0; i < batch_rank_cum_; ++i) {
    auto ith_x_addr = x_addr + input_offset_ * i;
    auto ith_gamma_addr = gamma_addr + para_offset_ * i;
    auto ith_beta_addr = beta_addr + para_offset_ * i;
    auto ith_runing_mean_addr = runing_mean_addr + para_offset_ * i;
    auto ith_runnig_variance_addr = runnig_variance_addr + para_offset_ * i;

    auto ith_y_addr = y_addr + input_offset_ * i;
    auto ith_save_mean_addr = save_mean_addr + updated_para_offset_ * i;
    auto ith_save_variance_addr = save_variance_addr + updated_para_offset_ * i;

    auto status = CopyMemDevice2Device(batch_, channel_, ith_gamma_addr, ith_beta_addr, ith_runing_mean_addr,
                                       ith_runnig_variance_addr, ws_gamma, ws_beta, ws_mean, ws_var, stream_ptr_);
    CHECK_CUDA_STATUS(status, kernel_name_);

    float *reserve_addr = nullptr;
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnBatchNormalizationForwardTrainingEx(
        handle_, mode_, bn_ops_, &alpha, &beta, x_desc_, ith_x_addr, z_desc_, z, y_desc_, ith_y_addr,
        scale_bias_mean_var_desc_, ws_gamma, ws_beta, exp_avg_factor_, ws_mean, ws_var, epsilon_, ith_save_mean_addr,
        ith_save_variance_addr, nullptr, workspace_addr, workspace_size_, reserve_addr, 0),
      "For 'InstanceNormGpuKernelMod', it launch cudnnBatchNormalizationForwardTrainingEx failed")
  }

  return true;
}

const std::vector<std::pair<KernelAttr, KernelRunFunc>> &InstanceNormGpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, KernelRunFunc>> func_list = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32),
     &InstanceNormGpuKernelMod::LaunchKernel<float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat16)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32),
     &InstanceNormGpuKernelMod::LaunchKernel<half>},
  };
  return func_list;
}
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, InstanceNorm, InstanceNormGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
