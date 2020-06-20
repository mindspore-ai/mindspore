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

#include "kernel/gpu/quant/minmax_update_perchannel_gpu_kernel.h"
#include "kernel/gpu/cuda_impl/minmax_update_impl.cuh"
#include <thrust/extrema.h>
#include <thrust/pair.h>
#include <thrust/device_vector.h>
#include <cuda_runtime_api.h>

namespace mindspore {
namespace kernel {
MinMaxUpdatePerChannelGpuKernel::MinMaxUpdatePerChannelGpuKernel()
    : input_size_(0), quant_num_(1), ema_(false), ema_decay_(0), num_channels_(0) {}

const std::vector<size_t> &MinMaxUpdatePerChannelGpuKernel::GetInputSizeList() const { return input_size_list_; }

const std::vector<size_t> &MinMaxUpdatePerChannelGpuKernel::GetOutputSizeList() const { return output_size_list_; }

const std::vector<size_t> &MinMaxUpdatePerChannelGpuKernel::GetWorkspaceSizeList() const {
  return workspace_size_list_;
}

bool MinMaxUpdatePerChannelGpuKernel::Init(const CNodePtr &kernel_node) {
  size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
  if (input_num != 3) {
    MS_LOG(EXCEPTION) << "Input number is " << input_num << ", but FakeQuant GpuKernel OP needs 3 output.";
  }

  size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
  if (output_num != 2) {
    MS_LOG(EXCEPTION) << "Output number is " << output_num << ", but FakeQuant GpuKernel OP needs 1 output.";
  }

  ema_ = GetValue<bool>(AnfAlgo::GetCNodePrimitive(kernel_node)->GetAttr("ema"));
  ema_decay_ = GetValue<float>(AnfAlgo::GetCNodePrimitive(kernel_node)->GetAttr("ema_decay"));

  // init size
  auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  num_channels_ = SizeToInt(input_shape[0]);
  for (size_t i = 0; i < input_shape.size(); ++i) {
    quant_num_ *= SizeToInt(input_shape[i]);
  }
  input_size_ = sizeof(float);
  for (size_t i = 0; i < input_shape.size(); i++) {
    input_size_ *= input_shape[i];
  }
  InitSizeLists();
  return true;
}

void MinMaxUpdatePerChannelGpuKernel::InitSizeLists() {
  input_size_list_.push_back(input_size_);                     // input
  input_size_list_.push_back(sizeof(float) * num_channels_);   // min
  input_size_list_.push_back(sizeof(float) * num_channels_);   // max
  output_size_list_.push_back(sizeof(float) * num_channels_);  // output min
  output_size_list_.push_back(sizeof(float) * num_channels_);  // output max
}

bool MinMaxUpdatePerChannelGpuKernel::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                             const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  float *output_min = GetDeviceAddress<float>(outputs, 0);
  float *output_max = GetDeviceAddress<float>(outputs, 1);
  float *input = GetDeviceAddress<float>(inputs, 0);
  float *input_min = GetDeviceAddress<float>(inputs, 1);
  float *input_max = GetDeviceAddress<float>(inputs, 2);

  if (input == nullptr) {
    MS_LOG(EXCEPTION) << "MinMaxUpdatePerChannelGpuKernel input x is null.";
  }
  if (input_min == nullptr || input_max == nullptr) {
    MS_LOG(EXCEPTION) << "MinMaxUpdatePerChannelGpuKernel input min or input max is null.";
  }

  // calculate the input min and max according by the parameter ema and ema_decay.
  CalMinMaxPerChannel(input, input_min, input_max, output_min, output_max, input_size_ / sizeof(float), num_channels_,
                      ema_decay_, ema_, reinterpret_cast<cudaStream_t>(stream_ptr));
  return true;
}

MS_REG_GPU_KERNEL(MinMaxUpdatePerChannel, MinMaxUpdatePerChannelGpuKernel)
}  // namespace kernel
}  // namespace mindspore
