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

#ifndef MINDSPORE_CCSRC_KERNEL_GPU_ADDN_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_KERNEL_GPU_ADDN_GPU_KERNEL_H_

#include <memory>
#include <vector>
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/kernel_constants.h"

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

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, uintptr_t) override {
    if (is_null_input_) {
      return true;
    }
    T *output_addr = GetDeviceAddress<T>(outputs, 0);
    const float alpha = 1;
    const float beta = 0;
    for (size_t i = 0; i < IntToSize(num_input_); i++) {
      T *input_addr = GetDeviceAddress<T>(inputs, i);
      CHECK_CUDNN_RET_WITH_EXCEPT(cudnnAddTensor(cudnn_handle_, &alpha, input_descriptor_, input_addr,
                                                 &(i > 0 ? alpha : beta), input_descriptor_, output_addr),
                                  "cudnnAddTensor failed");
    }
    return true;
  }
  bool Init(const CNodePtr &kernel_node) override {
    InitResource();
    cudnn_data_type_ = kCudnnDtypeMap[TypeIdLabel(AnfAlgo::GetInputDeviceDataType(kernel_node, 0))];
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    num_input_ = GetAttr<int>(kernel_node, "n");
    if (IntToSize(num_input_) != input_num) {
      MS_LOG(ERROR) << "Input number is " << num_input_ << " in attr, but got " << input_num << "input.";
      return false;
    }
    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(ERROR) << "Output number is " << output_num << ", but cudnnAddTensor needs 1 output.";
      return false;
    }
    auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    is_null_input_ = CHECK_NULL_INPUT(input_shape);
    if (is_null_input_) {
      MS_LOG(WARNING) << "AddNGpuFwdKernel input is null";
      InitSizeLists();
      return true;
    }
    for (size_t i = input_shape.size(); i < 4; i++) {
      (void)input_shape.insert(input_shape.begin(), 1);
    }
    int dimA[4];
    for (size_t i = 0; i < input_shape.size(); i++) {
      dimA[i] = SizeToInt(input_shape[i]);
    }
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnSetTensorNdDescriptorEx(input_descriptor_, CUDNN_TENSOR_NCHW, cudnn_data_type_,
                                                             SizeToInt(input_shape.size()), dimA),
                                "cudnnSetTensorNdDescriptor failed");
    InitSizeLists();
    return true;
  }

 protected:
  void InitResource() override {
    cudnn_handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCudnnHandle();
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnCreateTensorDescriptor(&input_descriptor_), "cudnnCreateTensorDescriptor failed");
  }
  void InitSizeLists() override {
    if (!is_null_input_) {
      CHECK_CUDNN_RET_WITH_EXCEPT(
        cudnnGetTensorSizeInBytes(input_descriptor_, reinterpret_cast<size_t *>(&input_size_)),
        "cudnnGetTensorSizeInBytes failed");
    }
    for (int i = 0; i < num_input_; i++) {
      input_size_list_.push_back(input_size_);
    }
    output_size_list_.push_back(input_size_);
  }

 private:
  void DestroyResource() noexcept {
    CHECK_CUDNN_RET_WITH_ERROR(cudnnDestroyTensorDescriptor(input_descriptor_), "cudnnDestroyTensorDescriptor failed");
  }
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
  int num_input_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_KERNEL_GPU_ADDN_GPU_KERNEL_H_
