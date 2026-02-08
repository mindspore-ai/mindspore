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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_BATCHNORM_FOLD_GRAD_GPUKERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_BATCHNORM_FOLD_GRAD_GPUKERNEL_H_

#include <vector>
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/cuda_impl/cuda_ops/batchnorm_fold_impl.cuh"
#include "kernel/gpu/cuda/quant/quant_op_const.h"

namespace mindspore {
namespace kernel {
template <typename T>
class BatchNormFoldGradGpuKernelMod : public NativeGpuKernelMod {
 public:
  BatchNormFoldGradGpuKernelMod()
      : input_size_(0),
        channel_size_(0),
        workspace_size_(0),
        momentum_(0.1),
        epsilon_(1e-12),
        is_training_(true),
        is_null_input_(false),
        freeze_bn_(0),
        current_step_(0),
        batch_(0),
        channel_(0),
        height_(0),
        width_(0) {}
  ~BatchNormFoldGradGpuKernelMod() = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    // 'd_batch_mean', 'd_batch_std', 'x', 'batch_mean', 'batch_std', 'current_step'
    T *d_batch_mean = GetDeviceAddress<T>(inputs, kIndex0);
    T *d_batch_std = GetDeviceAddress<T>(inputs, kIndex1);
    T *x = GetDeviceAddress<T>(inputs, kIndex2);
    T *batch_mean = GetDeviceAddress<T>(inputs, kIndex3);
    T *batch_std = GetDeviceAddress<T>(inputs, kIndex4);
    int *current_step = GetDeviceAddress<int>(inputs, kIndex5);
    int current_step_host[1];
    CHECK_CUDA_RET_WITH_ERROR_NOTRACE(
      cudaMemcpyAsync(current_step_host, current_step, sizeof(int), cudaMemcpyDeviceToHost,
                      reinterpret_cast<cudaStream_t>(stream_ptr)),
      "Copy gpu memoy failed.");
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream_ptr)),
                                       "cudaStreamSyncFailed");
    T *dx = GetDeviceAddress<T>(outputs, kIndex0);

    cudaError_t status = cudaErrorNotReady;
    if (!is_training_ || current_step_host[0] >= freeze_bn_) {
      status =
        ThrustFillWith(dx, batch_ * channel_ * height_ * width_, 0.f, reinterpret_cast<cudaStream_t>(stream_ptr));
      CHECK_CUDA_STATUS(status, kernel_name_);
      return true;
    }
    status = CalBatchNormFoldGrad(d_batch_mean, d_batch_std, x, batch_mean, batch_std, batch_, channel_, height_,
                                  width_, dx, reinterpret_cast<cudaStream_t>(stream_ptr));
    CHECK_CUDA_STATUS(status, kernel_name_);
    return true;
  }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override {
    epsilon_ = GetValue<T>(primitive_->GetAttr("epsilon"));
    is_training_ = GetValue<bool>(primitive_->GetAttr("is_training"));
    freeze_bn_ = static_cast<int>(GetValue<int64_t>(primitive_->GetAttr("freeze_bn")));
    return true;
  }

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override {
    output_size_list_.clear();
    workspace_size_list_.clear();
    auto input_shape = inputs[kIndex2]->GetShapeVector();
    is_null_input_ = CHECK_SHAPE_NULL(input_shape, kernel_name_, "input");
    if (is_null_input_) {
      output_size_list_.push_back(input_size_);
      return KRET_UNKNOWN_SHAPE;
    }
    if (input_shape.size() != kSize4) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of input should be 4, but got "
                        << input_shape.size();
    }
    batch_ = input_shape[kIndex0];
    channel_ = input_shape[kIndex1];
    height_ = input_shape[kIndex2];
    width_ = input_shape[kIndex3];

    input_size_ = sizeof(T) * batch_ * channel_ * height_ * width_;
    channel_size_ = sizeof(T) * channel_;

    output_size_list_.push_back(input_size_);
    return KRET_OK;
  }

 private:
  size_t input_size_;
  size_t channel_size_;
  size_t workspace_size_;

  T momentum_;
  T epsilon_;
  bool is_training_;
  bool is_null_input_;
  int freeze_bn_;
  int current_step_;
  int batch_;
  int channel_;
  int height_;
  int width_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_BATCHNORM_FOLD_GRAD_GPUKERNEL_H_
