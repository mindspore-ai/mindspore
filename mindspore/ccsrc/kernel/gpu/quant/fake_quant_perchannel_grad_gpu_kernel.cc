/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "kernel/gpu/quant/fake_quant_perchannel_grad_gpu_kernel.h"
#include "kernel/gpu/cuda_impl/fake_quant_perchannel_impl.cuh"

namespace mindspore {
namespace kernel {
FakeQuantPerChannelGradGpuKernel::FakeQuantPerChannelGradGpuKernel()
    : input_size_(0),
      num_bits_(0),
      quant_min_(0),
      quant_max_(0),
      num_channels_(0),
      quant_delay_(0),
      global_step_(0),
      narrow_range_(false),
      symmetric_(false) {}

const std::vector<size_t> &FakeQuantPerChannelGradGpuKernel::GetInputSizeList() const { return input_size_list_; }

const std::vector<size_t> &FakeQuantPerChannelGradGpuKernel::GetOutputSizeList() const { return output_size_list_; }

const std::vector<size_t> &FakeQuantPerChannelGradGpuKernel::GetWorkspaceSizeList() const {
  return workspace_size_list_;
}

bool FakeQuantPerChannelGradGpuKernel::Init(const CNodePtr &kernel_node) {
  size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
  if (input_num != 4) {
    MS_LOG(EXCEPTION) << "Input number is " << input_num << ", but FakeQuantGrad GpuKernel OP needs 4 output.";
  }

  size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
  if (output_num != 1) {
    MS_LOG(EXCEPTION) << "Output number is " << output_num << ", but FakeQuantGrad GpuKernel OP needs 1 output.";
  }

  num_bits_ = GetValue<int>(AnfAlgo::GetCNodePrimitive(kernel_node)->GetAttr("num_bits"));
  if (num_bits_ <= 2 || num_bits_ >= 16) {
    MS_LOG(EXCEPTION) << "Attr \'num_bits\' " << num_bits_ << " is out of range, expected between 2 and 16.";
  }

  quant_delay_ = GetValue<int>(AnfAlgo::GetCNodePrimitive(kernel_node)->GetAttr("quant_delay"));
  if (quant_delay_ < 0) {
    MS_LOG(EXCEPTION) << "Attr \'quant_delay_\' " << quant_delay_ << " is less then 0, require larger than 0.";
  }

  symmetric_ = GetValue<bool>(AnfAlgo::GetCNodePrimitive(kernel_node)->GetAttr("symmetric"));
  narrow_range_ = GetValue<bool>(AnfAlgo::GetCNodePrimitive(kernel_node)->GetAttr("narrow_range"));

  // quant min and max value
  quant_min_ = 0;
  quant_max_ = (1 << num_bits_) - 1;
  if (narrow_range_) {
    quant_min_++;
  }

  auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  num_channels_ = SizeToInt(input_shape[0]);
  input_size_ = sizeof(float);
  for (size_t i = 0; i < input_shape.size(); i++) {
    input_size_ *= input_shape[i];
  }
  InitSizeLists();
  return true;
}

void FakeQuantPerChannelGradGpuKernel::InitSizeLists() {
  input_size_list_.push_back(input_size_);                        // gradient
  input_size_list_.push_back(input_size_);                        // input
  input_size_list_.push_back(sizeof(float) * num_channels_);      // min
  input_size_list_.push_back(sizeof(float) * num_channels_);      // max
  output_size_list_.push_back(input_size_);                       // output
  workspace_size_list_.push_back(sizeof(float) * num_channels_);  // scale in channel
  workspace_size_list_.push_back(sizeof(float) * num_channels_);  // min in channel
  workspace_size_list_.push_back(sizeof(float) * num_channels_);  // max in channel
}

bool FakeQuantPerChannelGradGpuKernel::Launch(const std::vector<AddressPtr> &inputs,
                                              const std::vector<AddressPtr> &workspace,
                                              const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  (void)workspace;
  float *output = GetDeviceAddress<float>(outputs, 0);
  float *gradient = GetDeviceAddress<float>(inputs, 0);
  float *input = GetDeviceAddress<float>(inputs, 1);
  float *input_min = GetDeviceAddress<float>(inputs, 2);
  float *input_max = GetDeviceAddress<float>(inputs, 3);
  float *scale = GetDeviceAddress<float>(workspace, 0);
  float *nudge_min = GetDeviceAddress<float>(workspace, 1);
  float *nudge_max = GetDeviceAddress<float>(workspace, 2);

  if (gradient == nullptr) {
    MS_LOG(EXCEPTION) << "FakeQuantPerChannelGradGpuKernel gradient is null";
  }
  if (input == nullptr) {
    MS_LOG(EXCEPTION) << "FakeQuantPerChannelGradGpuKernel input is null";
  }
  if (input_min == nullptr) {
    MS_LOG(EXCEPTION) << "FakeQuantPerChannelGradGpuKernel input min is null";
  }
  if (input_max == nullptr) {
    MS_LOG(EXCEPTION) << "FakeQuantPerChannelGradGpuKernel input max is null";
  }

  int total_size = input_size_ / sizeof(float);
  if (global_step_ >= quant_delay_) {
    CalNudgePerChannel(input_min, input_max, quant_min_, quant_max_, nudge_min, nudge_max, scale, num_channels_,
                       symmetric_, reinterpret_cast<cudaStream_t>(stream_ptr));
    CalFakeQuantPerChannelGrad(input, gradient, output, total_size, num_channels_, nudge_min, nudge_max,
                               reinterpret_cast<cudaStream_t>(stream_ptr));
  } else {
    CHECK_CUDA_RET_WITH_ERROR(cudaMemcpyAsync(output, gradient, input_size_, cudaMemcpyDeviceToDevice,
                                              reinterpret_cast<cudaStream_t>(stream_ptr)),
                              "Copy gpu memory failed.");
  }
  global_step_++;
  return true;
}

MS_REG_GPU_KERNEL(FakeQuantPerChannelGrad, FakeQuantPerChannelGradGpuKernel)
}  // namespace kernel
}  // namespace mindspore
