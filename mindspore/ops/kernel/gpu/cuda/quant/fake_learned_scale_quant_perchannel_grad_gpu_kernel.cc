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
#include "kernel/gpu/cuda/quant/fake_learned_scale_quant_perchannel_grad_gpu_kernel.h"
#include "kernel/gpu/cuda_impl/cuda_ops/fake_learned_scale_quant_perchannel_impl.cuh"
#include "kernel/gpu/cuda/quant/quant_op_const.h"

namespace mindspore {
namespace kernel {
FakeLearnedScaleQuantPerChannelGradGpuKernelMod::FakeLearnedScaleQuantPerChannelGradGpuKernelMod()
    : input_size_(0),
      workspace_size_(0),
      quant_num_(1),
      quant_delay_(0),
      global_step_(0),
      neg_trunc_(false),
      num_channels_(0) {}

bool FakeLearnedScaleQuantPerChannelGradGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                                           const std::vector<KernelTensor *> &outputs) {
  quant_delay_ = static_cast<int>(GetValue<int64_t>(primitive_->GetAttr("quant_delay")));
  if (quant_delay_ < 0) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the value of quant_delay_ cannot be less than 0, but got "
                      << quant_delay_;
  }
  neg_trunc_ = GetValue<bool>(primitive_->GetAttr("neg_trunc"));
  return true;
}

int FakeLearnedScaleQuantPerChannelGradGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                                            const std::vector<KernelTensor *> &outputs) {
  output_size_list_.clear();
  workspace_size_list_.clear();
  // init size
  auto input_shape = inputs[kIndex0]->GetShapeVector();
  num_channels_ = LongToInt(input_shape[0]);
  for (size_t i = 0; i < input_shape.size(); ++i) {
    quant_num_ *= LongToInt(input_shape[i]);
  }
  input_size_ = sizeof(float) * quant_num_;
  output_size_list_.push_back(input_size_);                    //  grad_input
  output_size_list_.push_back(sizeof(float) * num_channels_);  // grad_alpha
  workspace_size_list_.push_back(input_size_);                 // input_div_alpha
  workspace_size_list_.push_back(input_size_);                 // input_quant
  return KRET_OK;
}

bool FakeLearnedScaleQuantPerChannelGradGpuKernelMod::Launch(const std::vector<KernelTensor *> &inputs,
                                                             const std::vector<KernelTensor *> &workspace,
                                                             const std::vector<KernelTensor *> &outputs,
                                                             void *stream_ptr) {
  float *grad_input = GetDeviceAddress<float>(outputs, kIndex0);
  float *grad_alpha = GetDeviceAddress<float>(outputs, kIndex1);
  float *gradient = GetDeviceAddress<float>(inputs, kIndex0);
  float *input = GetDeviceAddress<float>(inputs, kIndex1);
  float *input_alpha = GetDeviceAddress<float>(inputs, kIndex2);
  float *input_quant_max = GetDeviceAddress<float>(inputs, kIndex3);
  float *input_div_alpha = GetDeviceAddress<float>(workspace, kIndex0);
  float *input_quant = GetDeviceAddress<float>(workspace, kIndex1);

  MS_EXCEPTION_IF_NULL(grad_input);
  MS_EXCEPTION_IF_NULL(grad_alpha);
  MS_EXCEPTION_IF_NULL(gradient);
  MS_EXCEPTION_IF_NULL(input);
  MS_EXCEPTION_IF_NULL(input_alpha);
  MS_EXCEPTION_IF_NULL(input_quant_max);
  MS_EXCEPTION_IF_NULL(input_div_alpha);
  MS_EXCEPTION_IF_NULL(input_quant);
  const int kChannelLen = num_channels_;
  std::vector<float> alpha_no_grad(kChannelLen);
  if (memset_s(alpha_no_grad.data(), kChannelLen * sizeof(float), 0, kChannelLen * sizeof(float)) != EOK) {
    MS_LOG(EXCEPTION) << "Failed to set memory.";
  }
  if (global_step_ >= quant_delay_) {
    CHECK_CUDA_RET_WITH_ERROR_NOTRACE(
      cudaMemcpyAsync(grad_alpha, alpha_no_grad.data(), sizeof(float) * kChannelLen, cudaMemcpyHostToDevice,
                      reinterpret_cast<cudaStream_t>(stream_ptr)),
      "Copy gpu memory failed");
    auto status = CalLSQNudgePerChannel(input, quant_num_, input_alpha, input_quant_max, input_div_alpha, input_quant,
                                        neg_trunc_, num_channels_, reinterpret_cast<cudaStream_t>(stream_ptr));
    CHECK_CUDA_STATUS(status, kernel_name_);
    status =
      CalFakeLearnedScaleQuantPerChannelGrad(grad_input, grad_alpha, gradient, quant_num_, input_div_alpha, input_quant,
                                             neg_trunc_, num_channels_, reinterpret_cast<cudaStream_t>(stream_ptr));
    CHECK_CUDA_STATUS(status, kernel_name_);
  } else {
    CHECK_CUDA_RET_WITH_ERROR_NOTRACE(
      cudaMemcpyAsync(grad_alpha, alpha_no_grad.data(), sizeof(float) * kChannelLen, cudaMemcpyHostToDevice,
                      reinterpret_cast<cudaStream_t>(stream_ptr)),
      "Copy gpu memory failed");
    CHECK_CUDA_RET_WITH_ERROR_NOTRACE(cudaMemcpyAsync(grad_input, gradient, input_size_, cudaMemcpyDeviceToDevice,
                                                      reinterpret_cast<cudaStream_t>(stream_ptr)),
                                      "Copy gpu memory failed");
  }
  global_step_++;
  return true;
}

MS_REG_GPU_KERNEL(FakeLearnedScaleQuantPerChannelGrad, FakeLearnedScaleQuantPerChannelGradGpuKernelMod)
}  // namespace kernel
}  // namespace mindspore
