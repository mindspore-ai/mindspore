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

#include "kernel/gpu/cudnn/instance_norm_grad_gpu_kernel.h"
#include <map>
#include <utility>
#include "mindspore/ops/infer/instance_norm_grad.h"

namespace mindspore {
namespace kernel {
namespace {
using KernelRunFunc = InstanceNormGradGpuKernelMod::KernelRunFunc;
constexpr auto kNCDims = 2;
}  // namespace
bool InstanceNormGradGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                        const std::vector<KernelTensor *> &outputs) {
  handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCudnnHandle();
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateTensorDescriptor(&x_desc_),
                                      "For 'InstanceNormGradGpuKernelMod', it create x desc failed");
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateTensorDescriptor(&dy_desc_),
                                      "For 'InstanceNormGradGpuKernelMod', it create dy desc failed");
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateTensorDescriptor(&dx_desc_),
                                      "For 'InstanceNormGradGpuKernelMod', it create dx desc failed");
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateTensorDescriptor(&scale_bias_diff_desc_),
                                      "For 'InstanceNormGradGpuKernelMod', it create para desc failed");

  if (primitive_->HasAttr("batch_rank")) {
    int64_t batch_rank = GetValue<int64_t>(primitive_->GetAttr("batch_rank"));
    batch_rank_ = LongToSize(batch_rank);
  }
  epsilon_ = static_cast<double>(GetValue<float>(primitive_->GetAttr("epsilon")));
  auto inplace_algo =
    primitive_->HasAttr("inplace_algo") ? GetValue<std::string>(primitive_->GetAttr("inplace_algo")) : "cover";
  beta_data_diff_ = inplace_algo == "cover" ? 0 : 1;
  cudnn_data_type_ = GetCudnnDataType(TypeIdLabel(inputs[kIndex0]->dtype_id()));

  if (!MatchKernelFunc(kernel_name_, inputs, outputs)) {
    return false;
  }

  return true;
}
int InstanceNormGradGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
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

  int batch = 1;
  int channel = SizeToInt(batch_) * SizeToInt(channel_);
  int height = 1;
  const int width =
    std::accumulate(input_shape.begin() + batch_rank_ + kNCDims, input_shape.end(), int64_t(1), std::multiplies{});

  input_offset_ = channel * width;
  para_offset_ = channel_;
  updated_para_offset_ = channel;

  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
    cudnnSetTensor4dDescriptor(x_desc_, CUDNN_TENSOR_NCHW, cudnn_data_type_, batch, channel, height, width),
    "For 'InstanceNormGradGpuKernelMod', it set x desc failed");
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
    cudnnSetTensor4dDescriptor(dy_desc_, CUDNN_TENSOR_NCHW, cudnn_data_type_, batch, channel, height, width),
    "For 'InstanceNormGradGpuKernelMod', it set dy desc failed");
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
    cudnnSetTensor4dDescriptor(dx_desc_, CUDNN_TENSOR_NCHW, cudnn_data_type_, batch, channel, height, width),
    "For 'InstanceNormGradGpuKernelMod', it set dx desc failed");
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
    cudnnSetTensor4dDescriptor(scale_bias_diff_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, channel, 1, 1),
    "For 'InstanceNormGradGpuKernelMod', it set para desc failed");

  size_t para_size;
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnGetTensorSizeInBytes(scale_bias_diff_desc_, &para_size),
                                      "For 'InstanceNormGradGpuKernelMod', it get para size failed");
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
    cudnnGetBatchNormalizationBackwardExWorkspaceSize(handle_, mode_, bn_ops_, x_desc_, y_desc_, dy_desc_, dz_desc_,
                                                      dx_desc_, scale_bias_diff_desc_, activation_desc_,
                                                      &workspace_size_),
    "For 'InstanceNormGradGpuKernelMod', it launch cudnnGetBatchNormalizationBackwardExWorkspaceSize failed");
  workspace_size_list_.clear();
  workspace_size_list_ = {
    para_size,  // ws gamma
    para_size,  // ws dgamma
    para_size,  // ws dbeta
    workspace_size_,
  };
  return KRET_OK;
}

template <typename T>
bool InstanceNormGradGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                                const std::vector<KernelTensor *> &workspace,
                                                const std::vector<KernelTensor *> &outputs) {
  auto dy = GetDeviceAddress<T>(inputs, kIndex0);
  auto x = GetDeviceAddress<T>(inputs, kIndex1);
  auto gamma = GetDeviceAddress<float>(inputs, kIndex2);
  auto save_mean = GetDeviceAddress<float>(inputs, kIndex3);
  auto save_variance = GetDeviceAddress<float>(inputs, kIndex4);
  void *beta = nullptr;
  T *y = nullptr;

  auto dx = GetDeviceAddress<T>(outputs, kIndex0);
  auto dgamma = GetDeviceAddress<float>(outputs, kIndex1);
  auto dbeta = GetDeviceAddress<float>(outputs, kIndex2);
  T *dz = nullptr;

  float *ws_gamma = GetDeviceAddress<float>(workspace, kIndex0);
  float *ws_dgamma = GetDeviceAddress<float>(workspace, kIndex1);
  float *ws_dbeta = GetDeviceAddress<float>(workspace, kIndex2);
  void *workspace_addr = GetPossiblyNullDeviceAddress<T>(workspace, kIndex3);

  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnSetStream(handle_, stream_ptr_),
                                      "For 'InstanceNormGradGpuKernelMod', cudnnSetStream failed.")

  for (size_t i = 0; i < batch_rank_cum_; ++i) {
    auto ith_dy = dy + input_offset_ * i;
    auto ith_x = x + input_offset_ * i;
    auto ith_gamma = gamma + para_offset_ * i;
    auto ith_save_mean = save_mean + updated_para_offset_ * i;
    auto ith_save_variance = save_variance + updated_para_offset_ * i;

    auto ith_dx = dx + input_offset_ * i;
    auto ith_dgamma = dgamma + para_offset_ * i;
    auto ith_dbeta = dbeta + para_offset_ * i;

    CopyMemDevice2Device(batch_, channel_, ith_gamma, nullptr, nullptr, nullptr, ws_gamma, nullptr, nullptr, nullptr,
                         stream_ptr_);
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(stream_ptr_),
                                       "For 'InstanceNormGradGpuKernelMod', it launch cudaStreamSynchronized failed");

    const float alpha_data_diff = 1;
    const float alpha_param_diff = 1;
    const float beta_param_diff = 0;
    float *reserve_addr = nullptr;
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnBatchNormalizationBackwardEx(handle_, mode_, bn_ops_, &alpha_data_diff, &beta_data_diff_, &alpha_param_diff,
                                        &beta_param_diff, x_desc_, ith_x, y_desc_, y, dy_desc_, ith_dy, dz_desc_, dz,
                                        dx_desc_, ith_dx, scale_bias_diff_desc_, ws_gamma, beta, ws_dgamma, ws_dbeta,
                                        epsilon_, ith_save_mean, ith_save_variance, activation_desc_, workspace_addr,
                                        workspace_size_, reserve_addr, 0),
      "For 'InstanceNormGradGpuKernelMod', it launch cudnnBatchNormalizationBackwardEx failed");
    auto status = ComputeMean(batch_, channel_, ith_dgamma, ith_dbeta, ws_dgamma, ws_dbeta, stream_ptr_);
    CHECK_CUDA_STATUS(status, kernel_name_);
  }

  return true;
}
const std::vector<std::pair<KernelAttr, KernelRunFunc>> &InstanceNormGradGpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, KernelRunFunc>> func_list = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)    // dy
       .AddInputAttr(kNumberTypeFloat32)    // x
       .AddInputAttr(kNumberTypeFloat32)    // scale
       .AddInputAttr(kNumberTypeFloat32)    // save_mean
       .AddInputAttr(kNumberTypeFloat32)    // save_variance
       .AddOutputAttr(kNumberTypeFloat32)   // dx
       .AddOutputAttr(kNumberTypeFloat32)   // dscale
       .AddOutputAttr(kNumberTypeFloat32),  // dbias
     &InstanceNormGradGpuKernelMod::LaunchKernel<float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)    // dy
       .AddInputAttr(kNumberTypeFloat16)    // x
       .AddInputAttr(kNumberTypeFloat32)    // scale
       .AddInputAttr(kNumberTypeFloat32)    // save_mean
       .AddInputAttr(kNumberTypeFloat32)    // save_variance
       .AddOutputAttr(kNumberTypeFloat16)   // dx
       .AddOutputAttr(kNumberTypeFloat32)   // dscale
       .AddOutputAttr(kNumberTypeFloat32),  // dbias
     &InstanceNormGradGpuKernelMod::LaunchKernel<half>},
  };
  return func_list;
}
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, InstanceNormGrad, InstanceNormGradGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
