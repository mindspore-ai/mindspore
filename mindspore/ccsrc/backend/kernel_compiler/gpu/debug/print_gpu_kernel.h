/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_DEBUG_PRINT_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_DEBUG_PRINT_GPU_KERNEL_H_

#include <utility>
#include <string>
#include <algorithm>
#include <vector>
#include <memory>
#include "ir/tensor.h"
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"

using mindspore::tensor::Tensor;

namespace mindspore {
namespace kernel {
template <typename T>
class PrintGpuKernel : public GpuKernel {
 public:
  PrintGpuKernel() { ResetResource(); }
  ~PrintGpuKernel() override = default;

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    VARIABLE_NOT_USED(workspace);
    for (size_t i = 0; i < inputs.size(); i++) {
      input_device_data_[i] = GetDeviceAddress<T>(inputs, i);
    }
    int *output_address = GetDeviceAddress<int>(outputs, 0);
    // host initialization
    std::vector<std::unique_ptr<T[]>> input_host_data;
    for (size_t i = 0; i < input_size_.size(); i++) {
      std::unique_ptr<T[]> value = std::make_unique<T[]>(input_size_[i]);
      input_host_data.push_back(std::move(value));
    }
    // check type
    T type_value = static_cast<T>(0.0f);
    auto type_id = CheckType(type_value);
    if (type_id == kTypeUnknown) {
      MS_LOG(EXCEPTION) << "GPU print does not support the input type.";
    }
    // print core function
    size_t string_idx = 0;
    for (size_t i = 0; i < input_flag_.size(); i++) {
      if (input_flag_[i] == -1) {
        std::cout << string_value_[string_idx] << std::endl;
        string_idx++;
      } else {
        size_t tensor_idx = LongToSize(input_flag_[i]);
        std::string error_msg = "cudaMemcpyAsync print loop failed at input_device_data[";
        error_msg.append(std::to_string(tensor_idx));
        error_msg.append("].");
        CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                                   cudaMemcpyAsync(input_host_data[tensor_idx].get(), input_device_data_[tensor_idx],
                                                   input_size_[tensor_idx] * sizeof(T), cudaMemcpyDeviceToHost,
                                                   reinterpret_cast<cudaStream_t>(stream_ptr)),
                                   error_msg);
        CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_, cudaDeviceSynchronize(), "cudaDeviceSyncFailed - Print");
        auto current_string = GetTensorString(&input_shape_, tensor_idx, type_id, &input_host_data, &input_size_);
        std::cout << current_string << std::endl;
      }
    }
    int output = 1;
    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                               cudaMemcpyAsync(output_address, &output, sizeof(int), cudaMemcpyHostToDevice,
                                               reinterpret_cast<cudaStream_t>(stream_ptr)),
                               "cudaMemcpyAsync output failed");
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    kernel_node_ = kernel_node;
    if (AnfAlgo::HasNodeAttr("string_pos", kernel_node)) {
      string_value_ = GetAttr<std::vector<std::string>>(kernel_node, "string_value");
      string_pos_ = GetAttr<std::vector<int64_t>>(kernel_node, "string_pos");
    }
    size_t input_tensor_num = AnfAlgo::GetInputTensorNum(kernel_node);
    input_flag_ = SetInputFlag(&string_pos_, input_tensor_num);
    input_device_data_ = std::make_unique<T *[]>(input_tensor_num);
    std::vector<size_t> value_shape;
    for (size_t i = 0; i < input_tensor_num; i++) {
      size_t value = 1;
      auto input_shape = AnfAlgo::GetInputDeviceShape(kernel_node, i);
      for (size_t j = 0; j < input_shape.size(); j++) {
        value *= input_shape[j];
        value_shape.push_back(input_shape[j]);
      }
      input_size_.push_back(value);
      input_shape_.push_back(value_shape);
      value_shape.clear();
    }
    InitSizeLists();
    return true;
  }

  void ResetResource() noexcept override {
    string_value_.clear();
    string_pos_.clear();
    input_flag_.clear();
    input_device_data_ = nullptr;
    input_size_.clear();
    input_shape_.clear();
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

 protected:
  void InitSizeLists() override {
    for (size_t i = 0; i < input_size_.size(); i++) {
      input_size_list_.push_back(input_size_[i] * sizeof(T));
    }
    output_size_list_.push_back(sizeof(int));
  }

  TypeId CheckType(T value) {
    if (std::is_same<T, bool>::value) {
      return kNumberTypeBool;
    } else if (std::is_same<T, int8_t>::value) {
      return kNumberTypeInt8;
    } else if (std::is_same<T, int16_t>::value) {
      return kNumberTypeInt16;
    } else if (std::is_same<T, int>::value) {
      return kNumberTypeInt32;
    } else if (std::is_same<T, int64_t>::value) {
      return kNumberTypeInt64;
    } else if (std::is_same<T, uint8_t>::value) {
      return kNumberTypeUInt8;
    } else if (std::is_same<T, uint16_t>::value) {
      return kNumberTypeUInt16;
    } else if (std::is_same<T, uint32_t>::value) {
      return kNumberTypeUInt32;
    } else if (std::is_same<T, uint64_t>::value) {
      return kNumberTypeUInt64;
    } else if (std::is_same<T, half>::value) {
      return kNumberTypeFloat16;
    } else if (std::is_same<T, float>::value) {
      return kNumberTypeFloat32;
    }
    return kTypeUnknown;
  }

  std::vector<int64_t> SetInputFlag(std::vector<int64_t> *string_pos, size_t input_tensor_num) {
    // -1 -> string position
    // others -> input tensor position
    std::vector<int64_t> res(string_pos->size() + input_tensor_num);
    // without string inputs
    int64_t value = 0;
    if (res.size() == input_tensor_num) {
      std::generate(res.begin(), res.end(), [&value]() { return value++; });
      return res;
    }
    for (size_t i = 0; i < string_pos->size(); i++) {
      if ((*string_pos)[i] < 0) {
        MS_LOG(EXCEPTION) << "string_pos cannot be a negative value";
      }
      auto index = IntToSize((*string_pos)[i]);
      res[index] = -1;
    }
    for (size_t i = 0; i < res.size(); i++) {
      if (res[i] != -1) {
        res[i] += value;
        value++;
      }
    }
    return res;
  }

  std::string GetTensorString(std::vector<std::vector<size_t>> *input_shape, size_t index, TypeId type_id,
                              std::vector<std::unique_ptr<T[]>> *input_host_data, std::vector<size_t> *input_size) {
    ShapeVector shape;
    (void)std::transform((*input_shape)[index].begin(), (*input_shape)[index].end(), std::back_inserter(shape),
                         [](const size_t &value) { return static_cast<int64_t>(value); });
    Tensor current_tensor(type_id, shape, (*input_host_data)[index].get(), (*input_size)[index] * sizeof(T));
    return current_tensor.ToStringNoLimit();
  }

 private:
  std::vector<std::string> string_value_;
  std::vector<int64_t> string_pos_;
  std::vector<int64_t> input_flag_;
  std::unique_ptr<T *[]> input_device_data_;
  std::vector<size_t> input_size_;
  std::vector<std::vector<size_t>> input_shape_;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_DEBUG_PRINT_GPU_KERNEL_H_
