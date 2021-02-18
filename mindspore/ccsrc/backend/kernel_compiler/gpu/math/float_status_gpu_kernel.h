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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_FLOAT_STATUS_GPU_KERNEL_H
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_FLOAT_STATUS_GPU_KERNEL_H

#include <memory>
#include <vector>
#include <map>
#include <string>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/cuda_impl/float_status_impl.cuh"
#include "backend/kernel_compiler/gpu/cuda_impl/slice_impl.cuh"

namespace mindspore {
namespace kernel {
enum Optype { OP_STATUS = 0, OP_INF, OP_NAN, OP_FINITE, OP_INVALID = 255 };
static const std::map<std::string, Optype> kOpTypeMap = {
  {"FloatStatus", OP_STATUS}, {"IsInf", OP_INF}, {"IsNan", OP_NAN}, {"IsFinite", OP_FINITE}};
template <typename T>
class FloatStatusGpuKernel : public GpuKernel {
 public:
  FloatStatusGpuKernel() : kernel_name_(OP_INVALID), input_size_(0), output_size_(0) {}
  ~FloatStatusGpuKernel() override = default;
  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    T *input = GetDeviceAddress<T>(inputs, 0);

    switch (kernel_name_) {
      case OP_STATUS: {
        float *output = GetDeviceAddress<float>(outputs, 0);
        FillDeviceArray(outputs[0]->size / sizeof(float), output, 0.0f, reinterpret_cast<cudaStream_t>(stream_ptr));
        CalFloatStatus(input_size_ / sizeof(T), input, output, reinterpret_cast<cudaStream_t>(stream_ptr));
        break;
      }
      case OP_INF: {
        bool *output = GetDeviceAddress<bool>(outputs, 0);
        CalIsInf(input_size_ / sizeof(T), input, output, reinterpret_cast<cudaStream_t>(stream_ptr));
        break;
      }
      case OP_NAN: {
        bool *output = GetDeviceAddress<bool>(outputs, 0);
        CalIsNan(input_size_ / sizeof(T), input, output, reinterpret_cast<cudaStream_t>(stream_ptr));
        break;
      }
      case OP_FINITE: {
        bool *output = GetDeviceAddress<bool>(outputs, 0);
        CalIsFinite(input_size_ / sizeof(T), input, output, reinterpret_cast<cudaStream_t>(stream_ptr));
        break;
      }
      default: {
        MS_LOG(EXCEPTION) << "FloatStatus type " << kernel_name_ << " is not supported.";
      }
    }
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    if (!CheckParam(kernel_node)) {
      return false;
    }
    auto shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    input_size_ = sizeof(T);
    for (size_t x : shape) {
      input_size_ = input_size_ * x;
    }
    auto kernel_name = AnfAlgo::GetCNodeName(kernel_node);
    auto iter = kOpTypeMap.find(kernel_name);
    if (iter == kOpTypeMap.end()) {
      MS_LOG(EXCEPTION) << "FloatStatus kernel " << kernel_name << " is not supported.";
    } else {
      kernel_name_ = iter->second;
    }
    if (kernel_name_ == OP_STATUS) {
      output_size_ = sizeof(float);
    } else {
      output_size_ = input_size_ / sizeof(T) * sizeof(bool);
    }
    InitSizeLists();
    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(input_size_);
    output_size_list_.push_back(output_size_);
  }

 private:
  bool CheckParam(const CNodePtr &kernel_node) {
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 1) {
      MS_LOG(ERROR) << "Input number is " << input_num << ", but FloatStatusGpuKernel needs 1 output.";
      return false;
    }
    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(ERROR) << "Output number is " << output_num << ", but FloatStatusGpuKernel needs 1 output.";
      return false;
    }
    return true;
  }

  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
  Optype kernel_name_;
  size_t input_size_;
  size_t output_size_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_FLOAT_STATUS_GPU_KERNEL_H
