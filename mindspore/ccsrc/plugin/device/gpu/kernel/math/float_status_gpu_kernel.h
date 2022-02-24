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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_FLOAT_STATUS_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_FLOAT_STATUS_GPU_KERNEL_H_

#include <memory>
#include <vector>
#include <map>
#include <string>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/float_status_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/slice_impl.cuh"

namespace mindspore {
namespace kernel {
enum Optype { OP_STATUS = 0, OP_INF, OP_NAN, OP_FINITE, OP_INVALID = 255 };
static const std::map<std::string, Optype> kOpTypeMap = {
  {"FloatStatus", OP_STATUS}, {"IsInf", OP_INF}, {"IsNan", OP_NAN}, {"IsFinite", OP_FINITE}};
template <typename T>
class FloatStatusGpuKernelMod : public NativeGpuKernelMod {
 public:
  FloatStatusGpuKernelMod() : kernel_name_(OP_INVALID), input_size_(0), output_size_(0), is_null_input_(false) {}
  ~FloatStatusGpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
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
    auto kernel_name = AnfAlgo::GetCNodeName(kernel_node);
    (void)CheckParam(kernel_node);
    auto shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    is_null_input_ = CHECK_SHAPE_NULL(shape, kernel_name, "input");
    if (is_null_input_) {
      InitSizeLists();
      return true;
    }
    input_size_ = sizeof(T);
    for (size_t x : shape) {
      input_size_ = input_size_ * x;
    }
    auto iter = kOpTypeMap.find(kernel_name);
    if (iter == kOpTypeMap.end()) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << ", only support these types: FloatStatus, IsInf, IsNan, IsFinite "
                        << "currently, but got " << kernel_name;
    }
    kernel_name_ = iter->second;

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
  void CheckParam(const CNodePtr &kernel_node) {
    auto kernel_name = AnfAlgo::GetCNodeName(kernel_node);
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the number of inputs should be 1, but got " << input_num;
    }
    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the number of outputs should be 1, but got " << output_num;
    }
  }

  Optype kernel_name_;
  size_t input_size_;
  size_t output_size_;
  bool is_null_input_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_FLOAT_STATUS_GPU_KERNEL_H_
