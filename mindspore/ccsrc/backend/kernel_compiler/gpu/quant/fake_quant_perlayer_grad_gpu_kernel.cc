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

#include "backend/kernel_compiler/gpu/quant/fake_quant_perlayer_grad_gpu_kernel.h"
#include "backend/kernel_compiler/gpu/cuda_impl/fake_quant_perlayer_impl.cuh"

namespace mindspore {
namespace kernel {
FakeQuantPerLayerGradGpuKernel::FakeQuantPerLayerGradGpuKernel()
    : input_size_(0),
      workspace_size_(0),
      num_bits_(0),
      quant_min_(0),
      quant_max_(0),
      quant_num_(1),
      quant_delay_(0),
      global_step_(0),
      narrow_range_(false),
      is_null_input_(false),
      symmetric_(false) {}

const std::vector<size_t> &FakeQuantPerLayerGradGpuKernel::GetInputSizeList() const { return input_size_list_; }

const std::vector<size_t> &FakeQuantPerLayerGradGpuKernel::GetOutputSizeList() const { return output_size_list_; }

const std::vector<size_t> &FakeQuantPerLayerGradGpuKernel::GetWorkspaceSizeList() const { return workspace_size_list_; }

bool FakeQuantPerLayerGradGpuKernel::Init(const CNodePtr &kernel_node) {
  kernel_node_ = kernel_node;
  size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
  if (input_num != 4) {
    MS_LOG(EXCEPTION) << "Input number is " << input_num << ", but FakeQuantGrad GpuKernel OP needs 4 output.";
  }

  size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
  if (output_num != 1) {
    MS_LOG(EXCEPTION) << "Output number is " << output_num << ", but FakeQuantGrad GpuKernel OP needs 1 output.";
  }

  auto prim = AnfAlgo::GetCNodePrimitive(kernel_node);
  MS_EXCEPTION_IF_NULL(prim);
  num_bits_ = static_cast<unsigned int>(GetValue<int64_t>(prim->GetAttr("num_bits")));
  if (num_bits_ <= 2 || num_bits_ >= 16) {
    MS_LOG(EXCEPTION) << "Attr \'num_bits\' " << num_bits_ << " is out of range, expected between 2 and 16.";
  }

  quant_delay_ = static_cast<int>(GetValue<int64_t>(prim->GetAttr("quant_delay")));
  if (quant_delay_ < 0) {
    MS_LOG(EXCEPTION) << "Attr \'quant_delay_\' " << quant_delay_ << " is less then 0, require larger than 0.";
  }

  symmetric_ = GetValue<bool>(prim->GetAttr("symmetric"));
  narrow_range_ = GetValue<bool>(prim->GetAttr("narrow_range"));

  // quant min and max value
  quant_min_ = 0;
  quant_max_ = (1 << num_bits_) - 1;
  if (narrow_range_) {
    quant_min_++;
  }

  // init size
  auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  is_null_input_ = CHECK_NULL_INPUT(input_shape);
  if (is_null_input_) {
    MS_LOG(WARNING) << "For 'FakeQuantPerlayerGradGpuKernel', input is null";
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

void FakeQuantPerLayerGradGpuKernel::InitSizeLists() {
  input_size_list_.push_back(input_size_);        // gradient
  input_size_list_.push_back(input_size_);        // input
  input_size_list_.push_back(sizeof(float));      // min
  input_size_list_.push_back(sizeof(float));      // max
  output_size_list_.push_back(input_size_);       // output
  workspace_size_list_.push_back(sizeof(float));  // scale
  workspace_size_list_.push_back(sizeof(float));  // nudge_min
  workspace_size_list_.push_back(sizeof(float));  // nudge_max
}

bool FakeQuantPerLayerGradGpuKernel::Launch(const std::vector<AddressPtr> &inputs,
                                            const std::vector<AddressPtr> &workspace,
                                            const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  if (is_null_input_) {
    return true;
  }
  float *output = GetDeviceAddress<float>(outputs, 0);
  float *gradient = GetDeviceAddress<float>(inputs, 0);
  float *input = GetDeviceAddress<float>(inputs, 1);
  float *input_min = GetDeviceAddress<float>(inputs, 2);
  float *input_max = GetDeviceAddress<float>(inputs, 3);
  float *scale = GetDeviceAddress<float>(workspace, 0);
  float *nudge_min = GetDeviceAddress<float>(workspace, 1);
  float *nudge_max = GetDeviceAddress<float>(workspace, 2);

  if (global_step_ >= quant_delay_) {
    CalNudgePerLayer(input_min, input_max, quant_min_, quant_max_, nudge_min, nudge_max, scale, symmetric_,
                     reinterpret_cast<cudaStream_t>(stream_ptr));
    CalFakeQuantPerLayerGrad(input, gradient, output, quant_num_, nudge_min, nudge_max,
                             reinterpret_cast<cudaStream_t>(stream_ptr));
  } else {
    CHECK_CUDA_RET_WITH_ERROR(kernel_node_,
                              cudaMemcpyAsync(output, gradient, input_size_, cudaMemcpyDeviceToDevice,
                                              reinterpret_cast<cudaStream_t>(stream_ptr)),
                              "Copy gpu memory failed");
  }
  global_step_++;
  return true;
}

MS_REG_GPU_KERNEL(FakeQuantPerLayerGrad, FakeQuantPerLayerGradGpuKernel)
}  // namespace kernel
}  // namespace mindspore
