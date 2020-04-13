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

#include "kernel/gpu/quant/fake_quant_gpu_kernel.h"
#include "kernel/gpu/cuda_impl/fake_quant_impl.cuh"
#include <thrust/extrema.h>
#include <thrust/pair.h>
#include <thrust/device_vector.h>
#include <cuda_runtime_api.h>

namespace mindspore {
namespace kernel {
FakeQuantGpuKernel::FakeQuantGpuKernel()
    : input_size_(0),
      min_size_(0),
      max_size_(0),
      output_size_(0),
      workspace_size_(0),
      num_bits_(0),
      quant_min_(0),
      quant_max_(0),
      quant_num_(0),
      quant_delay_(0),
      ema_(false),
      ema_decay_(0),
      global_step_(0),
      training_(false),
      narrow_range_(false),
      symmetric_(false) {}

const std::vector<size_t> &FakeQuantGpuKernel::GetInputSizeList() const { return input_size_list_; }

const std::vector<size_t> &FakeQuantGpuKernel::GetOutputSizeList() const { return output_size_list_; }

const std::vector<size_t> &FakeQuantGpuKernel::GetWorkspaceSizeList() const { return workspace_size_list_; }

bool FakeQuantGpuKernel::Init(const CNodePtr &kernel_node) {
  size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
  if (input_num != 3) {
    MS_LOG(EXCEPTION) << "Input number is " << input_num << ", but FakeQuant GpuKernel OP needs 3 output.";
  }

  size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
  if (output_num != 1) {
    MS_LOG(EXCEPTION) << "Output number is " << output_num << ", but FakeQuant GpuKernel OP needs 1 output.";
  }

  num_bits_ = GetValue<int>(AnfAlgo::GetCNodePrimitive(kernel_node)->GetAttr("num_bits"));
  ema_ = GetValue<bool>(AnfAlgo::GetCNodePrimitive(kernel_node)->GetAttr("ema"));
  ema_decay_ = 1.0 - GetValue<float>(AnfAlgo::GetCNodePrimitive(kernel_node)->GetAttr("ema_decay"));
  training_ = GetValue<bool>(AnfAlgo::GetCNodePrimitive(kernel_node)->GetAttr("training"));

  if (num_bits_ <= 2 || num_bits_ >= 16) {
    MS_LOG(EXCEPTION) << "Attr \'num_bits\' " << num_bits_ << " is out of range, expected between 2 and 16.";
  }

  quant_delay_ = GetValue<int>(AnfAlgo::GetCNodePrimitive(kernel_node)->GetAttr("quant_delay"));
  if (quant_delay_ < 0) {
    MS_LOG(EXCEPTION) << "Attr \'quant_delay\' " << num_bits_ << "is less then 0, require larger than 0.";
  }

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

  if (quant_num_ == 0) {
    quant_num_ = 1;
  }
  auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  for (size_t i = 0; i < input_shape.size(); ++i) {
    quant_num_ *= SizeToInt(input_shape[i]);
  }

  input_size_ = sizeof(float);
  min_size_ = sizeof(float);
  max_size_ = sizeof(float);
  for (size_t i = 0; i < input_shape.size(); i++) {
    input_size_ *= input_shape[i];
  }
  output_size_ = input_size_;
  InitSizeLists();
  return true;
}

void FakeQuantGpuKernel::InitSizeLists() {
  input_size_list_.push_back(input_size_);  // input
  input_size_list_.push_back(min_size_);    // min
  input_size_list_.push_back(max_size_);    // max
  output_size_list_.push_back(output_size_);
  workspace_size_list_.push_back(workspace_size_);
}

bool FakeQuantGpuKernel::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                const std::vector<AddressPtr> &outputs, uintptr_t stream_ptr) {
  (void)workspace;
  float *output = GetDeviceAddress<float>(outputs, 0);
  float *input = GetDeviceAddress<float>(inputs, 0);
  float *input_min = GetDeviceAddress<float>(inputs, 1);
  float *input_max = GetDeviceAddress<float>(inputs, 2);

  if (input == nullptr) {
    MS_LOG(EXCEPTION) << "FakeQuantGpuKernel input x is null.";
  }
  if (input_min == nullptr) {
    MS_LOG(EXCEPTION) << "FakeQuantGpuKernel input min is null.";
  }
  if (input_max == nullptr) {
    MS_LOG(EXCEPTION) << "FakeQuantGpuKernel input max is null.";
  }

  // Allocate space for device copies
  int size = sizeof(float);
  float *d_scale = nullptr;
  float *d_nudge_min = nullptr;
  float *d_nudge_max = nullptr;
  CHECK_CUDA_RET_WITH_ERROR(cudaMalloc(reinterpret_cast<void **>(&d_scale), size), "Malloc gpu memory failed");
  CHECK_CUDA_RET_WITH_ERROR(cudaMalloc(reinterpret_cast<void **>(&d_nudge_min), size), "Malloc gpu memory failed");
  CHECK_CUDA_RET_WITH_ERROR(cudaMalloc(reinterpret_cast<void **>(&d_nudge_max), size), "Malloc gpu memory failed");

  if (training_) {
    // calculate the input min and max according by the parameter ema and ema_decay.
    CalMinMax(input, input_min, input_max, quant_num_, ema_decay_, ema_, reinterpret_cast<cudaStream_t>(stream_ptr));
    // control flow for quant_delay
    if (global_step_ >= quant_delay_) {
      // real launch
      CalNudge(input_min, input_max, quant_min_, quant_max_, d_nudge_min, d_nudge_max, d_scale,
               reinterpret_cast<cudaStream_t>(stream_ptr));
      CalFakeQuantize(input, output, quant_num_, d_nudge_min, d_nudge_max, d_scale, symmetric_,
                      reinterpret_cast<cudaStream_t>(stream_ptr));
    } else {
      CHECK_CUDA_RET_WITH_ERROR(cudaMemcpy(output, input, input_size_, cudaMemcpyDeviceToDevice),
                                "Copy gpu memory failed");
    }
    global_step_++;
  } else {
    // real launch
    CalNudge(input_min, input_max, quant_min_, quant_max_, d_nudge_min, d_nudge_max, d_scale,
             reinterpret_cast<cudaStream_t>(stream_ptr));
    CalFakeQuantize(input, output, quant_num_, d_nudge_min, d_nudge_max, d_scale, symmetric_,
                    reinterpret_cast<cudaStream_t>(stream_ptr));
  }

  // Cleanup
  CHECK_CUDA_RET_WITH_ERROR(cudaFree(d_scale), "Free gpu memory failed");
  CHECK_CUDA_RET_WITH_ERROR(cudaFree(d_nudge_min), "Free gpu memory failed");
  CHECK_CUDA_RET_WITH_ERROR(cudaFree(d_nudge_max), "Free gpu memory failed");

  return true;
}

MS_REG_GPU_KERNEL(FakeQuantWithMinMax, FakeQuantGpuKernel)
}  // namespace kernel
}  // namespace mindspore
