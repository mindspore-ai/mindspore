/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_SPARSE_SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_SPARSE_SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_GPU_KERNEL_H_

#include <stdint.h>
#include <vector>
#include <string>
#include <map>
#include "infer/sparse_softmax_cross_entropy_with_logits.h"
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/cuda_impl/cuda_ops/cross_entropy_impl.cuh"
#include "kernel/gpu/kernel_constants.h"

namespace mindspore {
namespace kernel {
template <typename T, typename S>
class SparseSoftmaxCrossEntropyWithLogitsGpuKernelMod : public NativeGpuKernelMod {
 public:
  SparseSoftmaxCrossEntropyWithLogitsGpuKernelMod()
      : cudnn_handle_(nullptr),
        logits_descriptor_(nullptr),
        softmax_output_descriptor_(nullptr),
        algo_(CUDNN_SOFTMAX_ACCURATE),
        mode_(CUDNN_SOFTMAX_MODE_INSTANCE),
        cudnn_data_type_(CUDNN_DATA_FLOAT),
        is_grad_(false),
        is_null_input_(false),
        kernel_name_("SparseSoftmaxCrossEntropyWithLogits"),
        logits_size_(0),
        labels_size_(0),
        output_size_(0),
        softmax_output_logits_size_(0),
        batch_size_(0),
        channel_size_(0),
        height_(0),
        width_(0) {}
  ~SparseSoftmaxCrossEntropyWithLogitsGpuKernelMod() override { DestroyResource(); }

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *logits_addr = GetDeviceAddress<T>(inputs, 0);
    S *labels_addr = GetDeviceAddress<S>(inputs, 1);
    T *output_addr = GetDeviceAddress<T>(outputs, 0);
    T *softmax_output_logits = GetDeviceAddress<T>(workspace, 0);

    const float alpha = 1;
    const float beta = 0;
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnSoftmaxForward(cudnn_handle_, algo_, mode_, &alpha, logits_descriptor_, logits_addr, &beta,
                          softmax_output_descriptor_, softmax_output_logits),
      "cudnnSoftmaxForward failed.");

    cudaError_t status = cudaErrorNotReady;
    if (is_grad_) {
      status = CrossEntropyGradWithSparse(softmax_output_logits, labels_addr, batch_size_, channel_size_, output_addr,
                                          reinterpret_cast<cudaStream_t>(stream_ptr));
    } else {
      status = CrossEntropyWithSparse(softmax_output_logits, labels_addr, batch_size_, channel_size_, output_addr,
                                      reinterpret_cast<cudaStream_t>(stream_ptr));
    }
    CHECK_CUDA_STATUS(status, kernel_name_);
    return true;
  }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
    CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputsNum, kernel_name_);
    CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputsNum, kernel_name_);
    is_grad_ = GetValue<bool>(primitive_->GetAttr(ops::kIsGrad));
    cudnn_data_type_ = GetCudnnDataType(TypeIdLabel(inputs[kIndex0]->dtype_id()));
    InitResource();
    return true;
  }

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
    if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
      return ret;
    }
    auto logits_shape = inputs[kIndex0]->GetShapeVector();
    auto labels_shape = inputs[kIndex1]->GetShapeVector();
    InferInputOutputSize(logits_shape, labels_shape);
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnSetTensor4dDescriptor(logits_descriptor_, CUDNN_TENSOR_NCHW, cudnn_data_type_, batch_size_, channel_size_,
                                 height_, width_),
      "cudnnSetTensor4dDescriptor failed.");
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnSetTensor4dDescriptor(softmax_output_descriptor_, CUDNN_TENSOR_NCHW, cudnn_data_type_, batch_size_,
                                 channel_size_, height_, width_),
      "cudnnSetTensor4dDescriptor failed.");
    InitSizeLists();
    return KRET_OK;
  }

  void DestroyResource() noexcept override {
    CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnDestroyTensorDescriptor(softmax_output_descriptor_),
                                       "cudnnDestroyTensorDescriptor failed.");
    CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnDestroyTensorDescriptor(logits_descriptor_),
                                       "cudnnDestroyTensorDescriptor failed.");
  }

 protected:
  void InitResource() override {
    cudnn_handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCudnnHandle();
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateTensorDescriptor(&logits_descriptor_),
                                        "cudnnCreateTensorDescriptor failed.");
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateTensorDescriptor(&softmax_output_descriptor_),
                                        "cudnnCreateTensorDescriptor failed.");
  }

  void InitSizeLists() {
    output_size_list_.clear();
    workspace_size_list_.clear();
    output_size_list_.push_back(output_size_);
    workspace_size_list_.push_back(softmax_output_logits_size_);
  }

 private:
  void InferInputOutputSize(const ShapeVector &logits_shape, const ShapeVector &labels_shape) {
    is_null_input_ =
      CHECK_SHAPE_NULL(logits_shape, kernel_name_, "logits") || CHECK_SHAPE_NULL(labels_shape, kernel_name_, "labels");
    if (is_null_input_ || IsDynamic(logits_shape) || IsDynamic(labels_shape)) {
      InitSizeLists();
      return;
    }
    CheckShapeValidation(logits_shape, labels_shape);

    size_t logits_dims = logits_shape.size();
    batch_size_ = 1;
    for (size_t i = 0; i < logits_dims - 1; i++) {
      batch_size_ *= LongToSizeClipNeg(logits_shape[i]);
    }
    channel_size_ = LongToSizeClipNeg(logits_shape[logits_dims - 1]);
    height_ = 1;
    width_ = 1;
    logits_size_ = sizeof(T) * batch_size_ * channel_size_ * height_ * width_;

    labels_size_ = sizeof(S) * SizeOf(labels_shape);

    output_size_ = is_grad_ ? logits_size_ : sizeof(T);
    softmax_output_logits_size_ = logits_size_;
  }

  void CheckShapeValidation(const ShapeVector &logits_shape, const ShapeVector &labels_shape) {
    size_t logits_dim_length = logits_shape.size();
    size_t labels_dim_length = labels_shape.size();
    if (labels_dim_length != logits_dim_length - 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of logits and labels should satisfy this "
                        << "equation: len(labels.shape) = len(logits.shape) - 1, but got the dimension of labels: "
                        << labels_dim_length << ", the dimension of logits: " << logits_dim_length;
    }
    if (!std::equal(labels_shape.begin(), labels_shape.end(), logits_shape.begin())) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the shape of logits and labels must be the same except "
                        << "the last dimension, but got the shape of logits: " << logits_shape
                        << ", the shape of labels: " << labels_shape;
    }
  }

  cudnnHandle_t cudnn_handle_;
  cudnnTensorDescriptor_t logits_descriptor_;
  cudnnTensorDescriptor_t softmax_output_descriptor_;
  cudnnSoftmaxAlgorithm_t algo_;
  cudnnSoftmaxMode_t mode_;
  cudnnDataType_t cudnn_data_type_;
  bool is_grad_;
  bool is_null_input_;
  std::string kernel_name_;

  size_t logits_size_;
  size_t labels_size_;
  size_t output_size_;
  size_t softmax_output_logits_size_;

  size_t batch_size_;
  size_t channel_size_;
  size_t height_;
  size_t width_;

  const size_t kInputsNum = 2;
  const size_t kOutputsNum = 1;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_SPARSE_SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_GPU_KERNEL_H_
