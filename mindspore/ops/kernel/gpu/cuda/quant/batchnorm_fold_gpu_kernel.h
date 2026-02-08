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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_BATCHNORM_FOLD_GPUKERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_BATCHNORM_FOLD_GPUKERNEL_H_

#include <vector>
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/kernel_constants.h"
#include "kernel/gpu/cuda_impl/cuda_ops/batchnorm_fold_impl.cuh"
#include "kernel/gpu/cuda/quant/quant_op_const.h"

namespace mindspore {
namespace kernel {
template <typename T>
class BatchNormFoldGpuKernelMod : public NativeGpuKernelMod {
 public:
  BatchNormFoldGpuKernelMod()
      : input_size_(0),
        output_size_(0),
        exp_avg_factor_(0.9),
        epsilon_(1e-12),
        is_training_(true),
        is_null_input_(false),
        freeze_bn_(0),
        batch_(0),
        channel_(0),
        height_(0),
        width_(0),
        mode_(CUDNN_BATCHNORM_SPATIAL),
        x_desc_(nullptr),
        scale_bias_mean_var_desc_(nullptr),
        handle_(nullptr) {}

  ~BatchNormFoldGpuKernelMod() override { DestroyResource(); }

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    (void)workspace;
    auto x = GetDeviceAddress<T>(inputs, kIndex0);
    auto mean = GetDeviceAddress<T>(inputs, kIndex1);
    auto variance = GetDeviceAddress<T>(inputs, kIndex2);
    int *current_step = GetDeviceAddress<int>(inputs, kIndex3);
    int current_step_host[1];
    CHECK_CUDA_RET_WITH_ERROR_NOTRACE(
      cudaMemcpyAsync(current_step_host, current_step, sizeof(int), cudaMemcpyDeviceToHost,
                      reinterpret_cast<cudaStream_t>(stream_ptr)),
      "Copy gpu memoy failed.");
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream_ptr)),
                                       "cudaStreamSyncFailed");
    auto batch_mean = GetDeviceAddress<T>(outputs, kIndex0);
    auto batch_std = GetDeviceAddress<T>(outputs, kIndex1);
    auto running_mean = GetDeviceAddress<T>(outputs, kIndex2);
    auto running_std = GetDeviceAddress<T>(outputs, kIndex3);
    auto y = GetDeviceAddress<T>(workspace, kIndex0);

    CHECK_CUDA_RET_WITH_ERROR_NOTRACE(cudaMemcpyAsync(running_mean, mean, output_size_, cudaMemcpyDeviceToDevice,
                                                      reinterpret_cast<cudaStream_t>(stream_ptr)),
                                      "Failed to copy gpu memory.");
    CHECK_CUDA_RET_WITH_ERROR_NOTRACE(cudaMemcpyAsync(running_std, variance, output_size_, cudaMemcpyDeviceToDevice,
                                                      reinterpret_cast<cudaStream_t>(stream_ptr)),
                                      "Failed to copy gpu memory.");
    auto status = CalUpdateRunningStd(channel_, epsilon_, running_std, reinterpret_cast<cudaStream_t>(stream_ptr));
    CHECK_CUDA_STATUS(status, kernel_name_);
    if (!is_training_ || current_step_host[0] >= freeze_bn_) {
      CHECK_CUDA_RET_WITH_ERROR_NOTRACE(cudaMemset(batch_mean, 0, output_size_), "Failed to set gpu memory.");
      status = ThrustFillWith(batch_std, channel_, 1.f, reinterpret_cast<cudaStream_t>(stream_ptr));
      CHECK_CUDA_STATUS(status, kernel_name_);
      return true;
    }
    const T alpha = 1;
    const T beta = 0;
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnBatchNormalizationForwardTraining(handle_, mode_, &alpha, &beta, x_desc_, x, x_desc_, y,
                                             scale_bias_mean_var_desc_, mean, mean, exp_avg_factor_, mean, variance,
                                             epsilon_, batch_mean, batch_std),
      "Failed to launch kernel.")
    status = CalUpdateBatchStd(channel_, batch_std, reinterpret_cast<cudaStream_t>(stream_ptr));
    CHECK_CUDA_STATUS(status, kernel_name_);
    return true;
  }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override {
    InitResource();
    T momentum = GetValue<T>(primitive_->GetAttr("momentum"));
    exp_avg_factor_ = 1.0 - momentum;
    epsilon_ = GetValue<T>(primitive_->GetAttr("epsilon"));
    is_training_ = GetValue<bool>(primitive_->GetAttr("is_training"));
    freeze_bn_ = static_cast<int>(GetValue<int64_t>(primitive_->GetAttr("freeze_bn")));
    return true;
  }

  void DestroyResource() noexcept override {
    CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnDestroyTensorDescriptor(x_desc_), "Destroy x desc failed");
    CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnDestroyTensorDescriptor(scale_bias_mean_var_desc_),
                                       "Destroy para desc failed");
  }

  void SetSizeLists() {
    // batch_mean, batch_std, running_mean, running_std
    output_size_list_.push_back(output_size_);
    output_size_list_.push_back(output_size_);
    output_size_list_.push_back(output_size_);
    output_size_list_.push_back(output_size_);

    // store y
    workspace_size_list_.push_back(input_size_);
  }

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override {
    output_size_list_.clear();
    workspace_size_list_.clear();

    auto input_shape = inputs[0]->GetShapeVector();
    is_null_input_ = CHECK_SHAPE_NULL(input_shape, kernel_name_, "input");
    if (is_null_input_) {
      SetSizeLists();
      return KRET_UNKNOWN_SHAPE;
    }
    if (input_shape.size() != kSize4) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of input should be 4, but got "
                        << input_shape.size();
    }
    CheckTensorSize({input_shape});
    batch_ = LongToInt(input_shape[kIndex0]);
    channel_ = LongToInt(input_shape[kIndex1]);
    height_ = LongToInt(input_shape[kIndex2]);
    width_ = LongToInt(input_shape[kIndex3]);

    input_size_ = sizeof(T) * batch_ * channel_ * height_ * width_;
    output_size_ = sizeof(T) * channel_;

    cudnnDataType_t cudnnDataType = GetCudnnDataType(TypeIdLabel(inputs[0]->dtype_id()));
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnSetTensor4dDescriptor(x_desc_, CUDNN_TENSOR_NCHW, cudnnDataType, batch_, channel_, height_, width_),
      "Set x desc failed");

    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnSetTensor4dDescriptor(scale_bias_mean_var_desc_, CUDNN_TENSOR_NCHW, cudnnDataType, 1, channel_, 1, 1),
      "Set para desc failed");

    SetSizeLists();
    return KRET_OK;
  }

  void InitResource() override {
    handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCudnnHandle();
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateTensorDescriptor(&x_desc_), "Create x desc failed");
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateTensorDescriptor(&scale_bias_mean_var_desc_),
                                        "Create para desc failed");
  }

 private:
  size_t input_size_;
  size_t output_size_;

  double exp_avg_factor_;
  double epsilon_;
  bool is_training_;
  bool is_null_input_;
  int freeze_bn_;
  int batch_;
  int channel_;
  int height_;
  int width_;

  cudnnBatchNormMode_t mode_;
  cudnnTensorDescriptor_t x_desc_;
  cudnnTensorDescriptor_t scale_bias_mean_var_desc_;

  cudnnHandle_t handle_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_BATCHNORM_FOLD_GPUKERNEL_H_
