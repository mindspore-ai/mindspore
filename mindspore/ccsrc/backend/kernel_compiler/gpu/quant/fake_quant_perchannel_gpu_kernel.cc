/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "backend/kernel_compiler/gpu/quant/fake_quant_perchannel_gpu_kernel.h"
#include "backend/kernel_compiler/gpu/cuda_impl/fake_quant_perchannel_impl.cuh"
#include <thrust/extrema.h>
#include <thrust/pair.h>
#include <thrust/device_vector.h>
#include <cuda_runtime_api.h>

namespace mindspore {
namespace kernel {
FakeQuantPerChannelGpuKernel::FakeQuantPerChannelGpuKernel()
    : input_size_(0),
      num_channels_(0),
      num_bits_(0),
      training_(false),
      symmetric_(false),
      narrow_range_(false),
      is_null_input_(false),
      quant_delay_(0),
      quant_min_(0),
      quant_max_(0),
      global_step_(0) {}

const std::vector<size_t> &FakeQuantPerChannelGpuKernel::GetInputSizeList() const { return input_size_list_; }

const std::vector<size_t> &FakeQuantPerChannelGpuKernel::GetOutputSizeList() const { return output_size_list_; }

const std::vector<size_t> &FakeQuantPerChannelGpuKernel::GetWorkspaceSizeList() const { return workspace_size_list_; }

bool FakeQuantPerChannelGpuKernel::Init(const CNodePtr &kernel_node) {
  kernel_node_ = kernel_node;
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

  // get attribute
  auto prim = AnfAlgo::GetCNodePrimitive(kernel_node);
  MS_EXCEPTION_IF_NULL(prim);
  num_bits_ = static_cast<unsigned int>(GetValue<int64_t>(prim->GetAttr("num_bits")));
  training_ = GetValue<bool>(prim->GetAttr("training"));
  symmetric_ = GetValue<bool>(prim->GetAttr("symmetric"));
  narrow_range_ = GetValue<bool>(prim->GetAttr("narrow_range"));
  quant_delay_ = static_cast<int>(GetValue<int64_t>(prim->GetAttr("quant_delay")));

  if (num_bits_ <= 2 || num_bits_ >= 16) {
    MS_LOG(EXCEPTION) << "Attr \'num_bits\' " << num_bits_ << "is out of range, expected between 2 and 16.";
    return false;
  }

  if (quant_delay_ < 0) {
    MS_LOG(EXCEPTION) << "Attr \'quant_delay\' " << num_bits_ << " is less then 0, require larger than 0.";
    return false;
  }

  // quant min and max value
  quant_min_ = 0;
  quant_max_ = (1 << num_bits_) - 1;
  if (narrow_range_) {
    quant_min_++;
  }

  // shape info for gpu
  auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  is_null_input_ = CHECK_NULL_INPUT(input_shape);
  if (is_null_input_) {
    MS_LOG(WARNING) << "For 'FakeQuantPerchannelGpuKernel', input is null";
    InitSizeLists();
    return true;
  }
  if (input_shape.empty()) {
    MS_LOG(EXCEPTION) << "For 'FakeQuantPerchannelGpuKernel', input_shape is empty.";
  }
  num_channels_ = SizeToInt(input_shape[0]);
  input_size_ = sizeof(float);
  for (size_t i = 0; i < input_shape.size(); i++) {
    input_size_ *= input_shape[i];
  }
  InitSizeLists();
  return true;
}

void FakeQuantPerChannelGpuKernel::InitSizeLists() {
  input_size_list_.push_back(input_size_);                        // input in tensor
  input_size_list_.push_back(sizeof(float) * num_channels_);      // min one scalar
  input_size_list_.push_back(sizeof(float) * num_channels_);      // max on scalar
  output_size_list_.push_back(input_size_);                       // output in tensor
  workspace_size_list_.push_back(sizeof(float) * num_channels_);  // scale in channel
  workspace_size_list_.push_back(sizeof(float) * num_channels_);  // min in channel
  workspace_size_list_.push_back(sizeof(float) * num_channels_);  // max in channel
}

void FakeQuantPerChannelGpuKernel::CalFakeQuantize(const float *input, float *output, float *input_min,
                                                   float *input_max, float *nudge_min, float *nudge_max, float *scale,
                                                   void *stream_ptr) {
  CalNudgePerChannel(input_min, input_max, quant_min_, quant_max_, nudge_min, nudge_max, scale, num_channels_,
                     symmetric_, reinterpret_cast<cudaStream_t>(stream_ptr));
  CalFakeQuantPerChannel(input, output, input_size_ / sizeof(float), num_channels_, nudge_min, nudge_max, scale,
                         reinterpret_cast<cudaStream_t>(stream_ptr));
}

bool FakeQuantPerChannelGpuKernel::Launch(const std::vector<AddressPtr> &inputs,
                                          const std::vector<AddressPtr> &workspace,
                                          const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  if (is_null_input_) {
    return true;
  }
  (void)workspace;
  float *output = GetDeviceAddress<float>(outputs, 0);
  float *input = GetDeviceAddress<float>(inputs, 0);
  float *input_min = GetDeviceAddress<float>(inputs, 1);
  float *input_max = GetDeviceAddress<float>(inputs, 2);
  float *scale = GetDeviceAddress<float>(workspace, 0);
  float *nudge_min = GetDeviceAddress<float>(workspace, 1);
  float *nudge_max = GetDeviceAddress<float>(workspace, 2);

  if (training_) {
    if (global_step_ >= quant_delay_) {
      CalFakeQuantize(input, output, input_min, input_max, nudge_min, nudge_max, scale, stream_ptr);
    } else {
      CHECK_CUDA_RET_WITH_ERROR(kernel_node_,
                                cudaMemcpyAsync(output, input, input_size_, cudaMemcpyDeviceToDevice,
                                                reinterpret_cast<cudaStream_t>(stream_ptr)),
                                "Copy gpu memory failed.");
    }
    global_step_++;
  } else {
    CalFakeQuantize(input, output, input_min, input_max, nudge_min, nudge_max, scale, stream_ptr);
  }

  return true;
}

MS_REG_GPU_KERNEL(FakeQuantPerChannel, FakeQuantPerChannelGpuKernel)
}  // namespace kernel
}  // namespace mindspore
