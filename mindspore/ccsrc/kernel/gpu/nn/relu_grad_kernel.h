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

#ifndef MINDSPORE_CCSRC_KERNEL_GPU_NN_RELU_GRAD_KERNEL_H_
#define MINDSPORE_CCSRC_KERNEL_GPU_NN_RELU_GRAD_KERNEL_H_

#include <vector>
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/kernel_constants.h"

namespace mindspore {
namespace kernel {
template <typename T>
class ReluGradGpuKernel : public GpuKernel {
 public:
  ReluGradGpuKernel()
      : cudnn_handle_(nullptr),
        activation_desc_(nullptr),
        mode_(CUDNN_ACTIVATION_RELU),
        data_descriptor_(nullptr),
        is_null_input_(false),
        cudnn_data_type_(CUDNN_DATA_FLOAT),
        input_size_(0) {}
  ~ReluGradGpuKernel() override { DestroyResource(); }
  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *) override {
    if (is_null_input_) {
      return true;
    }
    T *y = GetDeviceAddress<T>(inputs, 1);
    T *dy = GetDeviceAddress<T>(inputs, 0);
    T *dx = GetDeviceAddress<T>(outputs, 0);

    const float alpha = 1;
    const float beta = 0;
    CHECK_CUDNN_RET_WITH_EXCEPT(
      cudnnActivationBackward(cudnn_handle_, activation_desc_, &alpha, data_descriptor_, y, data_descriptor_, dy,
                              data_descriptor_, y, &beta, data_descriptor_, dx),
      "cudnnActivationBackward failed");

    return true;
  }
  bool Init(const CNodePtr &kernel_node) override {
    InitResource();
    cudnn_data_type_ = GetCudnnDataType(TypeIdLabel(AnfAlgo::GetInputDeviceDataType(kernel_node, 0)));
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 2) {
      MS_LOG(ERROR) << "Argument number is " << input_num << ", but ReluGradGpuKernel needs 2.";
      return false;
    }
    auto input_shape = AnfAlgo::GetOutputInferShape(kernel_node, 0);
    mode_ = CUDNN_ACTIVATION_RELU;
    is_null_input_ = CHECK_NULL_INPUT(input_shape);
    if (is_null_input_) {
      MS_LOG(WARNING) << "ReluGradGpuKernel input is null.";
      InitSizeLists();
      return true;
    }
    std::vector<int> shape;
    ShapeNdTo4d(input_shape, &shape);
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnSetActivationDescriptor(activation_desc_, mode_, CUDNN_PROPAGATE_NAN, 0.0),
                                "SetActivationDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnSetTensor4dDescriptor(data_descriptor_, CUDNN_TENSOR_NCHW, cudnn_data_type_,
                                                           shape[0], shape[1], shape[2], shape[3]),
                                "SetTensor4dDescriptor failed");

    InitSizeLists();
    return true;
  }

 protected:
  void InitResource() override {
    cudnn_handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCudnnHandle();
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnCreateTensorDescriptor(&data_descriptor_), "cudnnCreateTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnCreateActivationDescriptor(&activation_desc_),
                                "cudnnCreateActivationDescriptor failed");
  }
  void InitSizeLists() override {
    if (!is_null_input_) {
      CHECK_CUDNN_RET_WITH_EXCEPT(cudnnGetTensorSizeInBytes(data_descriptor_, &input_size_),
                                  "cudnnGetTensorSizeInBytes failed");
    }
    input_size_list_.push_back(input_size_);
    output_size_list_.push_back(input_size_);
    input_size_list_.push_back(input_size_);
  }

 private:
  void DestroyResource() noexcept {
    CHECK_CUDNN_RET_WITH_ERROR(cudnnDestroyActivationDescriptor(activation_desc_),
                               "cudnnDestroyActivationDescriptor failed");
    CHECK_CUDNN_RET_WITH_ERROR(cudnnDestroyTensorDescriptor(data_descriptor_), "cudnnDestroyTensorDescriptor failed");
  }

  cudnnHandle_t cudnn_handle_;
  cudnnActivationDescriptor_t activation_desc_;
  cudnnActivationMode_t mode_;
  cudnnTensorDescriptor_t data_descriptor_;
  bool is_null_input_;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
  cudnnDataType_t cudnn_data_type_;
  size_t input_size_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_KERNEL_GPU_NN_RELU_GRAD_KERNEL_H_
