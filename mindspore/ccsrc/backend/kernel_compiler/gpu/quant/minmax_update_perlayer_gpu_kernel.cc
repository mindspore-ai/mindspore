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

#include "backend/kernel_compiler/gpu/quant/minmax_update_perlayer_gpu_kernel.h"
#include "backend/kernel_compiler/gpu/cuda_impl/minmax_update_impl.cuh"
#include <thrust/extrema.h>
#include <thrust/pair.h>
#include <thrust/device_vector.h>
#include <cuda_runtime_api.h>

namespace mindspore {
namespace kernel {
MinMaxUpdatePerLayerGpuKernel::MinMaxUpdatePerLayerGpuKernel()
    : input_size_(0), quant_num_(1), ema_(false), is_null_input_(false), ema_decay_(0) {}

const std::vector<size_t> &MinMaxUpdatePerLayerGpuKernel::GetInputSizeList() const { return input_size_list_; }

const std::vector<size_t> &MinMaxUpdatePerLayerGpuKernel::GetOutputSizeList() const { return output_size_list_; }

const std::vector<size_t> &MinMaxUpdatePerLayerGpuKernel::GetWorkspaceSizeList() const { return workspace_size_list_; }

bool MinMaxUpdatePerLayerGpuKernel::Init(const CNodePtr &kernel_node) {
  size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
  if (input_num != 3) {
    MS_LOG(EXCEPTION) << "Input number is " << input_num << ", but FakeQuant GpuKernel OP needs 3 output.";
  }

  size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
  if (output_num != 2) {
    MS_LOG(EXCEPTION) << "Output number is " << output_num << ", but FakeQuant GpuKernel OP needs 1 output.";
  }

  auto prim = AnfAlgo::GetCNodePrimitive(kernel_node);
  MS_EXCEPTION_IF_NULL(prim);
  ema_ = GetValue<bool>(prim->GetAttr("ema"));
  ema_decay_ = GetValue<float>(prim->GetAttr("ema_decay"));

  // init size
  auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  is_null_input_ = CHECK_NULL_INPUT(input_shape);
  if (is_null_input_) {
    MS_LOG(WARNING) << "For 'MinMaxUpdatePerlayerGpuKernel', input is null";
    InitSizeLists();
    return true;
  }
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

void MinMaxUpdatePerLayerGpuKernel::InitSizeLists() {
  input_size_list_.push_back(input_size_);     // input
  input_size_list_.push_back(sizeof(float));   // input min
  input_size_list_.push_back(sizeof(float));   // input max
  output_size_list_.push_back(sizeof(float));  // output min
  output_size_list_.push_back(sizeof(float));  // output max
}

bool MinMaxUpdatePerLayerGpuKernel::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                           const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  if (is_null_input_) {
    return true;
  }
  float *output_min = GetDeviceAddress<float>(outputs, 0);
  float *output_max = GetDeviceAddress<float>(outputs, 1);
  float *input = GetDeviceAddress<float>(inputs, 0);
  float *input_min = GetDeviceAddress<float>(inputs, 1);
  float *input_max = GetDeviceAddress<float>(inputs, 2);

  CalMinMaxPerLayer(input, input_min, input_max, output_min, output_max, quant_num_, ema_decay_, ema_,
                    reinterpret_cast<cudaStream_t>(stream_ptr));

  return true;
}

MS_REG_GPU_KERNEL(MinMaxUpdatePerLayer, MinMaxUpdatePerLayerGpuKernel)
}  // namespace kernel
}  // namespace mindspore
