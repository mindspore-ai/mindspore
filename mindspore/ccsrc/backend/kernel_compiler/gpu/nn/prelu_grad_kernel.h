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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_PRELU_GRAD_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_PRELU_GRAD_KERNEL_H_

#include <vector>
#include <string>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/kernel_constants.h"
#include "backend/kernel_compiler/gpu/cuda_impl/relu_grad_impl.cuh"

namespace mindspore {
namespace kernel {
template <typename T>
class PReLUGpuGradKernel : public GpuKernel {
 public:
  PReLUGpuGradKernel()
      : data_format_(kOpFormat_NCDHW),
        input_size_(0),
        weight_size_(0),
        reduce_workspace_size_(0),
        spatial_count_(1),
        is_null_input_(false),
        channel_shared_(false),
        channel_last_(false) {}
  ~PReLUGpuGradKernel() override { DestroyResource(); }
  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    T *dy_addr = GetDeviceAddress<T>(inputs, 0);
    T *x_addr = GetDeviceAddress<T>(inputs, 1);
    T *w_addr = GetDeviceAddress<T>(inputs, 2);
    T *dx_addr = GetDeviceAddress<T>(outputs, 0);
    T *dw_addr = GetDeviceAddress<T>(outputs, 1);
    T *dw_collector_addr = GetDeviceAddress<T>(workspace, 0);
    T *reduce_workspace_addr = GetDeviceAddress<T>(workspace, 1);

    PReluChannelSharedGrad(input_size_ / sizeof(T), dy_addr, x_addr, w_addr, dx_addr, dw_collector_addr,
                           reinterpret_cast<cudaStream_t>(stream_ptr));

    if (data_type_ == CUDNN_DATA_DOUBLE) {
      T alpha = static_cast<T>(1.0f);
      T beta = static_cast<T>(0.0f);
      CHECK_CUDNN_RET_WITH_EXCEPT(
        kernel_node_,
        cudnnReduceTensor(cudnn_handle_, reduce_tensor_descriptor_, nullptr, 0, reduce_workspace_addr,
                          reduce_workspace_size_, &alpha, grad_weight_collector_descriptor_, dw_collector_addr, &beta,
                          grad_weight_descriptor_, dw_addr),
        "cudnnReduceTensor failed.");
    } else {
      const float alphaf = static_cast<float>(1.0f);
      const float betaf = static_cast<float>(0.0f);
      CHECK_CUDNN_RET_WITH_EXCEPT(
        kernel_node_,
        cudnnReduceTensor(cudnn_handle_, reduce_tensor_descriptor_, nullptr, 0, reduce_workspace_addr,
                          reduce_workspace_size_, &alphaf, grad_weight_collector_descriptor_, dw_collector_addr, &betaf,
                          grad_weight_descriptor_, dw_addr),
        "cudnnReduceTensor failed.");
    }
    return true;
  }

  void InitResource() override {
    cudnn_handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCudnnHandle();
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateReduceTensorDescriptor(&reduce_tensor_descriptor_),
                                "cudnnCreateReduceTensorDescriptor failed.");
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateTensorDescriptor(&grad_weight_collector_descriptor_),
                                "cudnnCreateTensorDescriptor failed.");
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateTensorDescriptor(&grad_weight_descriptor_),
                                "cudnnCreateTensorDescriptor failed.");
  }

  void DestroyResource() noexcept override {
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyReduceTensorDescriptor(reduce_tensor_descriptor_),
                               "cudnnDestroyReduceTensorDescriptor failed.");
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyTensorDescriptor(grad_weight_collector_descriptor_),
                               "cudnnDestroyTensorDescriptor failed.");
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyTensorDescriptor(grad_weight_descriptor_),
                               "cudnnDestroyTensorDescriptor failed.");
  }

  bool Init(const CNodePtr &kernel_node) override {
    kernel_node_ = kernel_node;
    input_size_ = sizeof(T);
    auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    is_null_input_ = CHECK_NULL_INPUT(input_shape);
    if (is_null_input_) {
      MS_LOG(WARNING) << "PReLUGpuBwdKernel input is null.";
    }
    for (size_t i = 0; i < input_shape.size(); ++i) {
      input_size_ *= input_shape[i];
    }
    weight_size_ = sizeof(T);
    auto weight_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 2);
    is_null_input_ = CHECK_NULL_INPUT(weight_shape);
    if (is_null_input_) {
      MS_LOG(WARNING) << "PReLUGpuBwdKernel input is null.";
    }
    for (auto dim : weight_shape) {
      weight_size_ *= dim;
    }
    channel_shared_ = (weight_shape[0] == 1);
    if (!channel_shared_) {
      MS_LOG(WARNING)
        << "PReLUGpuBwdKernel shares weight for all channels, but the given weight tensor has more than one element.";
    }

    spatial_count_ = 1;
    if (channel_last_) {
      for (size_t i = 1; i < input_shape.size() - 1; ++i) {
        spatial_count_ *= input_shape[i];
      }
    } else {
      for (size_t i = 2; i < input_shape.size(); ++i) {
        spatial_count_ *= input_shape[i];
      }
    }

    data_type_ = GetCudnnDataType(TypeIdLabel(AnfAlgo::GetInputDeviceDataType(kernel_node, 0)));
    int input_dim_length = input_shape.size();
    std::vector<size_t> reduce_out_shape(input_dim_length, 1);
    if (channel_last_) {
      reduce_out_shape[input_dim_length - 1] = weight_shape[0];
    } else {
      reduce_out_shape[1] = weight_shape[0];
    }
    InitResource();
    CudnnSetTensorNdDescriptor(reduce_out_shape, grad_weight_descriptor_, data_type_, kernel_node_);
    CudnnSetTensorNdDescriptor(input_shape, grad_weight_collector_descriptor_, data_type_, kernel_node_);
    cudnnDataType_t comp_type = (data_type_ == CUDNN_DATA_DOUBLE) ? CUDNN_DATA_DOUBLE : CUDNN_DATA_FLOAT;
    CHECK_CUDNN_RET_WITH_EXCEPT(
      kernel_node_,
      cudnnSetReduceTensorDescriptor(reduce_tensor_descriptor_, CUDNN_REDUCE_TENSOR_ADD, comp_type,
                                     CUDNN_NOT_PROPAGATE_NAN, CUDNN_REDUCE_TENSOR_NO_INDICES, CUDNN_32BIT_INDICES),
      "cudnnSetReduceTensorDescriptor failed");
    InitSizeLists();
    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(input_size_);
    input_size_list_.push_back(input_size_);
    input_size_list_.push_back(weight_size_);
    output_size_list_.push_back(input_size_);
    output_size_list_.push_back(weight_size_);
    CHECK_CUDNN_RET_WITH_EXCEPT(
      kernel_node_,
      cudnnGetReductionWorkspaceSize(cudnn_handle_, reduce_tensor_descriptor_, grad_weight_collector_descriptor_,
                                     grad_weight_descriptor_, &reduce_workspace_size_),
      "cudnnGetReductionWorkspaceSize failed.");
    workspace_size_list_.push_back(input_size_);
    workspace_size_list_.push_back(reduce_workspace_size_);
  }

 private:
  cudnnHandle_t cudnn_handle_;
  cudnnDataType_t data_type_;
  cudnnReduceTensorDescriptor_t reduce_tensor_descriptor_;
  cudnnTensorDescriptor_t grad_weight_collector_descriptor_;
  cudnnTensorDescriptor_t grad_weight_descriptor_;

  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
  std::string data_format_ = kOpFormat_NCDHW;
  size_t input_size_;
  size_t weight_size_;
  size_t reduce_workspace_size_;
  size_t spatial_count_;
  bool is_null_input_ = false;
  bool channel_shared_ = false;
  bool channel_last_ = false;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_PRELU_GRAD_KERNEL_H_
