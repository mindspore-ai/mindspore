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

#include "kernel/gpu/quant/fake_quant_per_channel_gpu_kernel.h"
#include "kernel/gpu/cuda_impl/fake_quant_per_channel_impl.cuh"
#include <thrust/extrema.h>
#include <thrust/pair.h>
#include <thrust/device_vector.h>
#include <cuda_runtime_api.h>

namespace mindspore {
namespace kernel {
FakeQuantPerChannelGpuKernel::FakeQuantPerChannelGpuKernel()
    : input_size_(0),
      min_size_(0),
      max_size_(0),
      output_size_(0),
      workspace_size_(0),
      num_bits_(0),
      quant_min_(0),
      quant_max_(0),
      quant_delay_(0),
      ema_(false),
      ema_decay_(0),
      global_step_(0),
      training_(false),
      channel_out_(0),
      narrow_range_(false),
      symmetric_(false) {}

const std::vector<size_t> &FakeQuantPerChannelGpuKernel::GetInputSizeList() const { return input_size_list_; }

const std::vector<size_t> &FakeQuantPerChannelGpuKernel::GetOutputSizeList() const { return output_size_list_; }

const std::vector<size_t> &FakeQuantPerChannelGpuKernel::GetWorkspaceSizeList() const { return workspace_size_list_; }

bool FakeQuantPerChannelGpuKernel::Init(const CNodePtr &kernel_node) {
  size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
  if (input_num != 3) {
    MS_LOG(EXCEPTION) << "Input number is " << input_num << ", but FakeQuant GpuKernel OP needs 3 input.";
    return false;
  }

  size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
  if (output_num != 1) {
    MS_LOG(EXCEPTION) << "Output number is " << output_num << " but FakeQuant GpuKernel OP needs 1 output.";
    return false;
  }

  num_bits_ = GetValue<int>(AnfAlgo::GetCNodePrimitive(kernel_node)->GetAttr("num_bits"));
  ema_ = GetValue<bool>(AnfAlgo::GetCNodePrimitive(kernel_node)->GetAttr("ema"));
  ema_decay_ = 1.0 - GetValue<float>(AnfAlgo::GetCNodePrimitive(kernel_node)->GetAttr("ema_decay"));

  if (num_bits_ <= 2 || num_bits_ >= 16) {
    MS_LOG(EXCEPTION) << "Attr \'num_bits\' " << num_bits_ << "is out of range, expected between 2 and 16.";
    return false;
  }

  quant_delay_ = GetValue<int>(AnfAlgo::GetCNodePrimitive(kernel_node)->GetAttr("quant_delay"));
  if (quant_delay_ < 0) {
    MS_LOG(EXCEPTION) << "Attr \'quant_delay\' " << num_bits_ << " is less then 0, require larger than 0.";
    return false;
  }

  training_ = GetValue<bool>(AnfAlgo::GetCNodePrimitive(kernel_node)->GetAttr("training"));

  symmetric_ = GetValue<bool>(AnfAlgo::GetCNodePrimitive(kernel_node)->GetAttr("symmetric"));
  if (symmetric_) {
    quant_min_ = 0 - (1 << (num_bits_ - 1));
    quant_max_ = (1 << (num_bits_ - 1)) - 1;
  } else {
    quant_min_ = 0;
    quant_max_ = (1 << num_bits_) - 1;
  }

  narrow_range_ = GetValue<bool>(AnfAlgo::GetCNodePrimitive(kernel_node)->GetAttr("narrow_range"));
  if (narrow_range_) {
    quant_min_++;
  }

  // shape info for gpu
  auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  channel_out_ = SizeToInt(input_shape[0]);
  min_size_ = sizeof(float) * channel_out_;
  max_size_ = sizeof(float) * channel_out_;
  input_size_ = sizeof(float);
  for (size_t i = 0; i < input_shape.size(); i++) {
    input_size_ *= input_shape[i];
  }
  output_size_ = input_size_;

  InitSizeLists();
  return true;
}

void FakeQuantPerChannelGpuKernel::InitSizeLists() {
  input_size_list_.push_back(input_size_);                       // input in tensor
  input_size_list_.push_back(min_size_);                         // min one scalar
  input_size_list_.push_back(max_size_);                         // max on scalar
  output_size_list_.push_back(output_size_);                     // output in tensor
  workspace_size_list_.push_back(sizeof(float) * channel_out_);  // scale in channel
  workspace_size_list_.push_back(sizeof(float) * channel_out_);  // min in channel
  workspace_size_list_.push_back(sizeof(float) * channel_out_);  // max in channel
}

void FakeQuantPerChannelGpuKernel::CalFakeQuantizeForTraining(float *input, float *output, float *input_min,
                                                              float *input_max, float *d_nudge_min, float *d_nudge_max,
                                                              float *d_scale, void *stream_ptr) {
  // calculate the input min and max according by the parameter ema and ema_decay.
  CalMinMaxPerChannel(input, input_min, input_max, input_size_ / sizeof(float), channel_out_, ema_decay_, ema_,
                      reinterpret_cast<cudaStream_t>(stream_ptr));
  // control flow for quant_delay
  if (global_step_ >= quant_delay_) {
    // real launch
    CalNudgePerChannel(input_min, input_max, quant_min_, quant_max_, d_nudge_min, d_nudge_max, d_scale, channel_out_,
                       reinterpret_cast<cudaStream_t>(stream_ptr));
    CalFakeQuantizePerChannel(input, output, input_size_ / sizeof(float), channel_out_, d_nudge_min, d_nudge_max,
                              d_scale, symmetric_, reinterpret_cast<cudaStream_t>(stream_ptr));
  } else {
    CHECK_CUDA_RET_WITH_ERROR(
      cudaMemcpyAsync(output, input, input_size_, cudaMemcpyDeviceToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
      "Copy gpu memory failed.");
  }
  global_step_++;
}

void FakeQuantPerChannelGpuKernel::CalFakeQuantizeForInfer(float *input, float *output, float *input_min,
                                                           float *input_max, float *d_nudge_min, float *d_nudge_max,
                                                           float *d_scale, void *stream_ptr) {
  // real launch
  CalNudgePerChannel(input_min, input_max, quant_min_, quant_max_, d_nudge_min, d_nudge_max, d_scale, channel_out_,
                     reinterpret_cast<cudaStream_t>(stream_ptr));
  CalFakeQuantizePerChannel(input, output, input_size_ / sizeof(float), channel_out_, d_nudge_min, d_nudge_max, d_scale,
                            symmetric_, reinterpret_cast<cudaStream_t>(stream_ptr));
}

bool FakeQuantPerChannelGpuKernel::Launch(const std::vector<AddressPtr> &inputs,
                                          const std::vector<AddressPtr> &workspace,
                                          const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  (void)workspace;
  float *output = GetDeviceAddress<float>(outputs, 0);
  float *input = GetDeviceAddress<float>(inputs, 0);
  float *input_min = GetDeviceAddress<float>(inputs, 1);
  float *input_max = GetDeviceAddress<float>(inputs, 2);
  float *d_scale = GetDeviceAddress<float>(workspace, 0);
  float *d_nudge_min = GetDeviceAddress<float>(workspace, 1);
  float *d_nudge_max = GetDeviceAddress<float>(workspace, 2);

  if (input == nullptr) {
    MS_LOG(EXCEPTION) << "FakeQuantPerChannelGpuKernel input is null.";
  }
  if (input_min == nullptr || input_max == nullptr) {
    MS_LOG(EXCEPTION) << "FakeQuantPerChannelGpuKernel input min or max is null.";
  }

  if (training_) {
    CalFakeQuantizeForTraining(input, output, input_min, input_max, d_nudge_min, d_nudge_max, d_scale, stream_ptr);
  } else {
    CalFakeQuantizeForInfer(input, output, input_min, input_max, d_nudge_min, d_nudge_max, d_scale, stream_ptr);
  }

  return true;
}

MS_REG_GPU_KERNEL(FakeQuantWithMinMaxPerChannel, FakeQuantPerChannelGpuKernel)
}  // namespace kernel
}  // namespace mindspore
