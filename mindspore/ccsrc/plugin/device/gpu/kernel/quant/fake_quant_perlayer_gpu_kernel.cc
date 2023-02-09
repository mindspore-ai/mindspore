/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/gpu/kernel/quant/fake_quant_perlayer_gpu_kernel.h"
#include <thrust/extrema.h>
#include <thrust/pair.h>
#include <thrust/device_vector.h>
#include <cuda_runtime_api.h>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/fake_quant_perlayer_impl.cuh"
#include "plugin/device/gpu/kernel/quant/quant_op_const.h"

namespace mindspore {
namespace kernel {
FakeQuantPerLayerGpuKernelMod::FakeQuantPerLayerGpuKernelMod()
    : input_size_(0),
      quant_min_(0),
      quant_max_(0),
      quant_num_(1),
      global_step_(0),
      num_bits_(0),
      quant_delay_(0),
      training_(false),
      narrow_range_(false),
      is_null_input_(false),
      symmetric_(false) {}

bool FakeQuantPerLayerGpuKernelMod::Init(const CNodePtr &kernel_node) {
  auto kernel_name = common::AnfAlgo::GetCNodeName(kernel_node);
  kernel_node_ = kernel_node;
  size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
  if (input_num != kSize3) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the number of inputs should be 3, but got " << input_num;
  }

  size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
  if (output_num != kSize1) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the number of outputs should be 1, but got " << output_num;
  }

  auto prim = common::AnfAlgo::GetCNodePrimitive(kernel_node);
  MS_EXCEPTION_IF_NULL(prim);
  num_bits_ = static_cast<unsigned int>(GetValue<int64_t>(prim->GetAttr("num_bits")));
  quant_delay_ = static_cast<int>(GetValue<int64_t>(prim->GetAttr("quant_delay")));
  training_ = GetValue<bool>(prim->GetAttr("training"));
  symmetric_ = GetValue<bool>(prim->GetAttr("symmetric"));
  narrow_range_ = GetValue<bool>(prim->GetAttr("narrow_range"));

  if (num_bits_ <= kMinQuantBit || num_bits_ >= kMaxQuantBit) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the value of num_bits should be in (2, 16), but got "
                      << num_bits_;
  }

  if (quant_delay_ < 0) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the value of quant_delay_ cannot be less than 0, but got "
                      << quant_delay_;
  }
  // quant min and max value
  quant_min_ = 0;
  quant_max_ = (1 << num_bits_) - 1;
  if (narrow_range_) {
    quant_min_++;
  }

  // init size
  auto input_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, kIndex0);
  is_null_input_ = CHECK_SHAPE_NULL(input_shape, kernel_name, "input");
  if (is_null_input_) {
    InitSizeLists();
    return true;
  }
  auto size = SizeOf(input_shape);
  quant_num_ = SizeToInt(size);
  input_size_ = sizeof(float) * size;
  InitSizeLists();
  return true;
}

void FakeQuantPerLayerGpuKernelMod::InitSizeLists() {
  input_size_list_.push_back(input_size_);        // x
  input_size_list_.push_back(sizeof(float));      // min
  input_size_list_.push_back(sizeof(float));      // max
  output_size_list_.push_back(input_size_);       // y
  workspace_size_list_.push_back(sizeof(float));  // scale
  workspace_size_list_.push_back(sizeof(float));  // nudge_min
  workspace_size_list_.push_back(sizeof(float));  // nudge_max
}

bool FakeQuantPerLayerGpuKernelMod::Launch(const std::vector<AddressPtr> &inputs,
                                           const std::vector<AddressPtr> &workspace,
                                           const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  if (is_null_input_) {
    return true;
  }
  float *output = GetDeviceAddress<float>(outputs, kIndex0);
  float *input = GetDeviceAddress<float>(inputs, kIndex0);
  float *input_min = GetDeviceAddress<float>(inputs, kIndex1);
  float *input_max = GetDeviceAddress<float>(inputs, kIndex2);
  float *scale = GetDeviceAddress<float>(workspace, kIndex0);
  float *nudge_min = GetDeviceAddress<float>(workspace, kIndex1);
  float *nudge_max = GetDeviceAddress<float>(workspace, kIndex2);

  if (training_) {
    // control flow for quant_delay
    if (global_step_ >= quant_delay_) {
      // real launch
      CalNudgePerLayer(input_min, input_max, quant_min_, quant_max_, nudge_min, nudge_max, scale, symmetric_,
                       reinterpret_cast<cudaStream_t>(stream_ptr));
      CalFakeQuantPerLayer(input, output, quant_num_, nudge_min, nudge_max, scale,
                           reinterpret_cast<cudaStream_t>(stream_ptr));
    } else {
      CHECK_CUDA_RET_WITH_ERROR(kernel_node_,
                                cudaMemcpyAsync(output, input, input_size_, cudaMemcpyDeviceToDevice,
                                                reinterpret_cast<cudaStream_t>(stream_ptr)),
                                "Copy gpu memory failed");
    }
    global_step_++;
  } else {
    // real launch
    CalNudgePerLayer(input_min, input_max, quant_min_, quant_max_, nudge_min, nudge_max, scale, symmetric_,
                     reinterpret_cast<cudaStream_t>(stream_ptr));
    CalFakeQuantPerLayer(input, output, quant_num_, nudge_min, nudge_max, scale,
                         reinterpret_cast<cudaStream_t>(stream_ptr));
  }

  return true;
}

MS_REG_GPU_KERNEL(FakeQuantPerLayer, FakeQuantPerLayerGpuKernelMod)
}  // namespace kernel
}  // namespace mindspore
