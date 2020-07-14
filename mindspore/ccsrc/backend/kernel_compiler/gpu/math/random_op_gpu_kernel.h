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

#ifndef MINDSPORE_CCSRC_KERNEL_GPU_RANDOMOP_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_KERNEL_GPU_RANDOMOP_GPU_KERNEL_H_

#include <curand_kernel.h>
#include <cuda_runtime_api.h>
#include <vector>
#include <string>
#include <map>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/cuda_impl/random_op_impl.cuh"

namespace mindspore {
namespace kernel {
enum RandomOptype { RANDOM_OP_NORMAL = 0, RANDOM_OP_INVALID_TYPE = 255 };

const std::map<std::string, RandomOptype> kRandomOpTypeMap = {{"StandardNormal", RANDOM_OP_NORMAL}};
template <typename T>
class RandomOpGpuKernel : public GpuKernel {
 public:
  RandomOpGpuKernel()
      : random_op_type_(RANDOM_OP_INVALID_TYPE),
        input_size_0_(0),
        output_size_(sizeof(T)),
        workspace_size_(sizeof(curandState)) {}
  ~RandomOpGpuKernel() override = default;

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    void *workspace_addr = GetDeviceAddress<void *>(workspace, 0);
    curandState *devStates = reinterpret_cast<curandState *>(workspace_addr);
    T *output_addr = GetDeviceAddress<T>(outputs, 0);

    switch (random_op_type_) {
      case RANDOM_OP_NORMAL: {
        StandardNormal(seed_, seed2_, devStates, output_addr, outputs[0]->size / sizeof(T),
                       reinterpret_cast<cudaStream_t>(stream_ptr));
        break;
      }
      default: {
        MS_LOG(EXCEPTION) << "Random operation " << random_op_type_ << " is not supported.";
      }
    }
    return true;
  }
  bool Init(const CNodePtr &kernel_node) override {
    std::string kernel_name = AnfAlgo::GetCNodeName(kernel_node);
    auto iter = kRandomOpTypeMap.find(kernel_name);
    if (iter == kRandomOpTypeMap.end()) {
      MS_LOG(EXCEPTION) << "Random operation " << kernel_name << " is not supported.";
    } else {
      random_op_type_ = iter->second;
    }
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 1) {
      MS_LOG(ERROR) << "Input number is " << input_num << ", but random op needs 1 input.";
      return false;
    }
    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(ERROR) << "Output number is " << output_num << ", but random op needs 1 output.";
      return false;
    }
    auto input_shape_0 = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    for (size_t i = 0; i < input_shape_0.size(); i++) {
      input_size_0_ += input_shape_0[i];
    }
    input_size_0_ *= sizeof(int);
    auto output_shape = AnfAlgo::GetOutputInferShape(kernel_node, 0);
    for (size_t i = 0; i < output_shape.size(); i++) {
      output_size_ *= output_shape[i];
      workspace_size_ *= output_shape[i];
    }
    seed_ = GetValue<int>(AnfAlgo::GetCNodePrimitive(kernel_node)->GetAttr("seed"));
    seed2_ = GetValue<int>(AnfAlgo::GetCNodePrimitive(kernel_node)->GetAttr("seed2"));
    InitSizeLists();
    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(input_size_0_);
    output_size_list_.push_back(output_size_);
    workspace_size_list_.push_back(workspace_size_);
  }

 private:
  RandomOptype random_op_type_;
  size_t input_size_0_;
  size_t output_size_;
  size_t workspace_size_;
  int seed_;
  int seed2_;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_KERNEL_GPU_RANDOMOP_GPU_KERNEL_H_
