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

#include "kernel/gpu/cuda/quant/fake_learned_scale_quant_perlayer_gpu_kernel.h"
#include <thrust/extrema.h>
#include <thrust/pair.h>
#include <thrust/device_vector.h>
#include <cuda_runtime_api.h>
#include "kernel/gpu/cuda_impl/cuda_ops/fake_learned_scale_quant_perlayer_impl.cuh"
#include "kernel/gpu/cuda/quant/quant_op_const.h"

namespace mindspore {
namespace kernel {
FakeLearnedScaleQuantPerLayerGpuKernelMod::FakeLearnedScaleQuantPerLayerGpuKernelMod()
    : input_size_(0), quant_num_(1), global_step_(0), quant_delay_(0), training_(false), neg_trunc_(false) {}

bool FakeLearnedScaleQuantPerLayerGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                                     const std::vector<KernelTensor *> &outputs) {
  quant_delay_ = static_cast<int>(GetValue<int64_t>(primitive_->GetAttr("quant_delay")));
  training_ = GetValue<bool>(primitive_->GetAttr("training"));
  neg_trunc_ = GetValue<bool>(primitive_->GetAttr("neg_trunc"));
  return true;
}

int FakeLearnedScaleQuantPerLayerGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                                      const std::vector<KernelTensor *> &outputs) {
  output_size_list_.clear();
  workspace_size_list_.clear();
  // init size
  auto input_shape = inputs[kIndex0]->GetShapeVector();
  for (size_t i = 0; i < input_shape.size(); ++i) {
    quant_num_ *= LongToInt(input_shape[i]);
  }
  input_size_ = sizeof(float) * quant_num_;
  output_size_list_.push_back(input_size_);     // y
  workspace_size_list_.push_back(input_size_);  // input_div_alpha
  workspace_size_list_.push_back(input_size_);  // input_quant
  return KRET_OK;
}

bool FakeLearnedScaleQuantPerLayerGpuKernelMod::Launch(const std::vector<KernelTensor *> &inputs,
                                                       const std::vector<KernelTensor *> &workspace,
                                                       const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  float *input = GetDeviceAddress<float>(inputs, kIndex0);
  float *input_alpha = GetDeviceAddress<float>(inputs, kIndex1);
  float *input_quant_max = GetDeviceAddress<float>(inputs, kIndex2);
  float *output = GetDeviceAddress<float>(outputs, kIndex0);
  float *input_div_alpha = GetDeviceAddress<float>(workspace, kIndex0);
  float *input_quant = GetDeviceAddress<float>(workspace, kIndex1);

  MS_EXCEPTION_IF_NULL(input);
  MS_EXCEPTION_IF_NULL(input_alpha);
  MS_EXCEPTION_IF_NULL(input_quant_max);
  MS_EXCEPTION_IF_NULL(output);
  MS_EXCEPTION_IF_NULL(input_div_alpha);
  MS_EXCEPTION_IF_NULL(input_quant);

  if (training_) {
    // control flow for quant_delay
    if (global_step_ >= quant_delay_) {
      // real launch
      auto status = CalLSQNudgePerLayer(input, quant_num_, input_alpha, input_quant_max, input_div_alpha, input_quant,
                                        neg_trunc_, reinterpret_cast<cudaStream_t>(stream_ptr));
      CHECK_CUDA_STATUS(status, kernel_name_);
      status = CalFakeLearnedScaleQuantPerLayer(output, quant_num_, input_alpha, input_quant,
                                                reinterpret_cast<cudaStream_t>(stream_ptr));
      CHECK_CUDA_STATUS(status, kernel_name_);
    } else {
      CHECK_CUDA_RET_WITH_ERROR_NOTRACE(cudaMemcpyAsync(output, input, input_size_, cudaMemcpyDeviceToDevice,
                                                        reinterpret_cast<cudaStream_t>(stream_ptr)),
                                        "Copy gpu memory failed");
    }
    global_step_++;
  } else {
    // real launch
    auto status = CalLSQNudgePerLayer(input, quant_num_, input_alpha, input_quant_max, input_div_alpha, input_quant,
                                      neg_trunc_, reinterpret_cast<cudaStream_t>(stream_ptr));
    CHECK_CUDA_STATUS(status, kernel_name_);
    status = CalFakeLearnedScaleQuantPerLayer(output, quant_num_, input_alpha, input_quant,
                                              reinterpret_cast<cudaStream_t>(stream_ptr));
    CHECK_CUDA_STATUS(status, kernel_name_);
  }

  return true;
}

MS_REG_GPU_KERNEL(FakeLearnedScaleQuantPerLayer, FakeLearnedScaleQuantPerLayerGpuKernelMod)
}  // namespace kernel
}  // namespace mindspore
