/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ADDN_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ADDN_GPU_KERNEL_H_

#include <memory>
#include <vector>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/math/broadcast_gpu_kernel.h"
#include "backend/kernel_compiler/gpu/cuda_impl/slice_impl.cuh"
#include "backend/kernel_compiler/gpu/kernel_constants.h"

namespace mindspore {
namespace kernel {
template <typename T>
class AddNGpuFwdKernel : public GpuKernel {
 public:
  AddNGpuFwdKernel()
      : cudnn_handle_(nullptr),
        input_descriptor_(nullptr),
        cudnn_data_type_(CUDNN_DATA_FLOAT),
        input_size_(0),
        output_size_(0),
        workspace_size_(0),
        is_null_input_(false),
        num_input_(0) {}
  ~AddNGpuFwdKernel() override { DestroyResource(); }
  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *output_addr = GetDeviceAddress<T>(outputs, 0);
    auto work_addr = output_addr;
    for (size_t i = 0; i < num_input_; i++) {
      if (output_addr == GetDeviceAddress<T>(inputs, i)) {
        work_addr = GetDeviceAddress<T>(workspace, 0);
        break;
      }
    }
    if (cudnn_data_type_ == CUDNN_DATA_INT32) {
      FillDeviceArray(outputs[0]->size / sizeof(T), output_addr, 0.0f, reinterpret_cast<cudaStream_t>(stream_ptr));
      FillDeviceArray(outputs[0]->size / sizeof(T), work_addr, 0.0f, reinterpret_cast<cudaStream_t>(stream_ptr));
    }
    const float alpha = 1;
    const float beta = 0;
    for (size_t i = 0; i < num_input_; i++) {
      T *input_addr = GetDeviceAddress<T>(inputs, i);
      if (cudnn_data_type_ == CUDNN_DATA_INT32) {
        ElewiseArith(outputs[0]->size / sizeof(T), BROADCAST_TYPE_ADD, input_addr, work_addr, work_addr,
                     reinterpret_cast<cudaStream_t>(stream_ptr));
      } else {
        CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                    cudnnAddTensor(cudnn_handle_, &alpha, input_descriptor_, input_addr,
                                                   &(i > 0 ? alpha : beta), input_descriptor_, work_addr),
                                    "cudnnAddTensor failed");
      }
    }
    if (work_addr != output_addr) {
      CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                                 cudaMemcpyAsync(output_addr, work_addr, outputs[0]->size, cudaMemcpyDeviceToDevice,
                                                 reinterpret_cast<cudaStream_t>(stream_ptr)),
                                 "Addn cudaMemcpyAsync outputs failed");
    }
    return true;
  }
  bool Init(const CNodePtr &kernel_node) override {
    kernel_node_ = kernel_node;
    InitResource();
    cudnn_data_type_ = GetCudnnDataType(TypeIdLabel(AnfAlgo::GetInputDeviceDataType(kernel_node, 0)));
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    num_input_ = GetAttr<int64_t>(kernel_node, "n");
    if (num_input_ != input_num) {
      MS_LOG(ERROR) << "Input number is " << num_input_ << " in attr, but got " << input_num << "input.";
      return false;
    }
    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(ERROR) << "Output number is " << output_num << ", but cudnnAddTensor needs 1 output.";
      return false;
    }
    auto input_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
    is_null_input_ = CHECK_NULL_INPUT(input_shape);
    if (is_null_input_) {
      MS_LOG(WARNING) << "AddNGpuFwdKernel input is null";
      InitSizeLists();
      return true;
    }
    for (size_t i = input_shape.size(); i < 4; i++) {
      (void)input_shape.insert(input_shape.begin(), 1);
    }
    std::vector<int> dimA;
    for (size_t i = 0; i < input_shape.size(); i++) {
      dimA.push_back(SizeToInt(input_shape[i]));
    }
    auto input_format = AnfAlgo::GetInputFormat(kernel_node, 0);
    if (input_format == kOpFormat_NHWC) {
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                  cudnnSetTensorNdDescriptorEx(input_descriptor_, CUDNN_TENSOR_NHWC, cudnn_data_type_,
                                                               SizeToInt(input_shape.size()), dimA.data()),
                                  "cudnnSetTensorNdDescriptor failed");
    } else {
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                  cudnnSetTensorNdDescriptorEx(input_descriptor_, CUDNN_TENSOR_NCHW, cudnn_data_type_,
                                                               SizeToInt(input_shape.size()), dimA.data()),
                                  "cudnnSetTensorNdDescriptor failed");
    }
    InitSizeLists();
    return true;
  }

  void DestroyResource() noexcept override {
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyTensorDescriptor(input_descriptor_),
                               "cudnnDestroyTensorDescriptor failed");
  }

 protected:
  void InitResource() override {
    cudnn_handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCudnnHandle();
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateTensorDescriptor(&input_descriptor_),
                                "cudnnCreateTensorDescriptor failed");
  }
  void InitSizeLists() override {
    if (!is_null_input_) {
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnGetTensorSizeInBytes(input_descriptor_, &input_size_),
                                  "cudnnGetTensorSizeInBytes failed");
    }
    for (size_t i = 0; i < num_input_; i++) {
      input_size_list_.push_back(input_size_);
    }
    output_size_list_.push_back(input_size_);
    workspace_size_list_.push_back(input_size_);
  }

 private:
  cudnnHandle_t cudnn_handle_;
  cudnnTensorDescriptor_t input_descriptor_;
  cudnnDataType_t cudnn_data_type_;

  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;

  size_t input_size_;
  size_t output_size_;
  size_t workspace_size_;
  bool is_null_input_;
  size_t num_input_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ADDN_GPU_KERNEL_H_
