/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_GPU_KERNEL_H_

#include <stdint.h>
#include <vector>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/cuda_impl/cross_entropy_impl.cuh"
#include "backend/kernel_compiler/gpu/kernel_constants.h"

namespace mindspore {
namespace kernel {
template <typename T, typename S>
class SoftmaxCrossEntropyWithLogitsGpuKernel : public GpuKernel {
 public:
  SoftmaxCrossEntropyWithLogitsGpuKernel()
      : cudnn_handle_(nullptr),
        logits_descriptor_(nullptr),
        softmax_output_descriptor_(nullptr),
        algo_(CUDNN_SOFTMAX_ACCURATE),
        mode_(CUDNN_SOFTMAX_MODE_INSTANCE),
        cudnn_data_type_(CUDNN_DATA_FLOAT),
        is_null_input_(false),
        logits_size_(0),
        labels_size_(0),
        output1_size_(0),
        output2_size_(0),
        softmax_output_logits_size_(0),
        batch_size_(0),
        channel_size_(0),
        height_(0),
        width_(0) {}
  ~SoftmaxCrossEntropyWithLogitsGpuKernel() override { DestroyResource(); }

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *logits_addr = GetDeviceAddress<T>(inputs, 0);
    S *labels_addr = GetDeviceAddress<S>(inputs, 1);
    T *loss_addr = GetDeviceAddress<T>(outputs, 0);
    T *dlogits_addr = GetDeviceAddress<T>(outputs, 1);
    T *softmax_output_logits = GetDeviceAddress<T>(workspace, 0);

    const float alpha = 1;
    const float beta = 0;
    CHECK_CUDNN_RET_WITH_EXCEPT(
      kernel_node_,
      cudnnSoftmaxForward(cudnn_handle_, algo_, mode_, &alpha, logits_descriptor_, logits_addr, &beta,
                          softmax_output_descriptor_, softmax_output_logits),
      "cudnnSoftmaxForward failed.");

    CrossEntropy(softmax_output_logits, labels_addr, batch_size_, channel_size_, loss_addr, dlogits_addr,
                 reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }
  bool Init(const CNodePtr &kernel_node) override {
    kernel_node_ = kernel_node;
    InitResource();
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 2) {
      MS_LOG(ERROR) << "Input number is " << input_num
                    << ", but SoftmaxCrossEntropyWithLogitsGpuKernel needs 2 inputs.";
      return false;
    }
    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 2) {
      MS_LOG(ERROR) << "Output number is " << output_num
                    << ", but SoftmaxCrossEntropyWithLogitsGpuKernel needs 2 output.";
      return false;
    }
    cudnn_data_type_ = GetCudnnDataType(TypeIdLabel(AnfAlgo::GetInputDeviceDataType(kernel_node, 0)));

    InferInputOutputSize(kernel_node);
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                cudnnSetTensor4dDescriptor(logits_descriptor_, CUDNN_TENSOR_NCHW, cudnn_data_type_,
                                                           batch_size_, channel_size_, height_, width_),
                                "cudnnSetTensor4dDescriptor failed.");
    CHECK_CUDNN_RET_WITH_EXCEPT(
      kernel_node_,
      cudnnSetTensor4dDescriptor(softmax_output_descriptor_, CUDNN_TENSOR_NCHW, cudnn_data_type_, batch_size_,
                                 channel_size_, height_, width_),
      "cudnnSetTensor4dDescriptor failed.");
    InitSizeLists();
    return true;
  }

  void DestroyResource() noexcept override {
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyTensorDescriptor(softmax_output_descriptor_),
                               "cudnnDestroyTensorDescriptor failed.");
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyTensorDescriptor(logits_descriptor_),
                               "cudnnDestroyTensorDescriptor failed.");
  }

 protected:
  void InitResource() override {
    cudnn_handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCudnnHandle();
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateTensorDescriptor(&logits_descriptor_),
                                "cudnnCreateTensorDescriptor failed.");
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateTensorDescriptor(&softmax_output_descriptor_),
                                "cudnnCreateTensorDescriptor failed.");
  }
  void InitSizeLists() override {
    input_size_list_.push_back(logits_size_);
    input_size_list_.push_back(labels_size_);
    output_size_list_.push_back(output1_size_);
    output_size_list_.push_back(output2_size_);
    workspace_size_list_.push_back(softmax_output_logits_size_);
  }

 private:
  void InferInputOutputSize(const CNodePtr &kernel_node) {
    auto logits_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    auto labels_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
    is_null_input_ = CHECK_NULL_INPUT(logits_shape) || CHECK_NULL_INPUT(labels_shape);
    if (is_null_input_) {
      MS_LOG(WARNING) << "For 'SoftmaxCrossEntropyWithLogitsGpuKernel', input is null";
      InitSizeLists();
      return;
    }
    CheckShapeValidation(logits_shape, labels_shape);

    size_t logits_dims = logits_shape.size();
    batch_size_ = 1;
    for (size_t i = 0; i < logits_dims - 1; i++) {
      batch_size_ *= logits_shape[i];
    }
    channel_size_ = logits_shape[logits_dims - 1];
    height_ = 1;
    width_ = 1;
    logits_size_ = sizeof(T) * batch_size_ * channel_size_ * height_ * width_;

    labels_size_ = 1;
    size_t labels_dims = labels_shape.size();
    for (size_t i = 0; i < labels_dims; i++) {
      labels_size_ *= labels_shape[i];
    }
    labels_size_ *= sizeof(S);

    output1_size_ = logits_size_ / logits_shape[logits_dims - 1];
    output2_size_ = logits_size_;
    softmax_output_logits_size_ = logits_size_;
  }
  void CheckShapeValidation(const std::vector<size_t> &logits_shape, const std::vector<size_t> &labels_shape) {
    size_t logits_dim_length = logits_shape.size();
    size_t labels_dim_length = labels_shape.size();
    if (logits_dim_length == 0) {
      MS_LOG(EXCEPTION) << "Logits shape cannot be empty";
    }

    if (labels_dim_length != logits_dim_length) {
      MS_LOG(EXCEPTION) << "Labels shape length should be equal to Logits shape length for "
                           "SoftmaxCrossEntropyWithLogits, but got Labels "
                           "shape length:"
                        << labels_dim_length << ", Logits shape length:" << logits_dim_length;
    }
    if (!std::equal(labels_shape.begin(), labels_shape.end(), logits_shape.begin())) {
      MS_LOG(EXCEPTION) << "The shape of labels should be the same as the shape of logits except its last dimension.";
    }
  }

  cudnnHandle_t cudnn_handle_;
  cudnnTensorDescriptor_t logits_descriptor_;
  cudnnTensorDescriptor_t softmax_output_descriptor_;
  cudnnSoftmaxAlgorithm_t algo_;
  cudnnSoftmaxMode_t mode_;
  cudnnDataType_t cudnn_data_type_;
  bool is_null_input_;

  size_t logits_size_;
  size_t labels_size_;
  size_t output1_size_;
  size_t output2_size_;
  size_t softmax_output_logits_size_;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
  size_t batch_size_;
  size_t channel_size_;
  size_t height_;
  size_t width_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_GPU_KERNEL_H_
