/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/cuda_impl/batchnorm_fold_impl.cuh"

namespace mindspore {
namespace kernel {
constexpr size_t INPUT_NUM = 6;
template <typename T>
class BatchNormFoldGradGpuKernel : public GpuKernel {
 public:
  BatchNormFoldGradGpuKernel()
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
  ~BatchNormFoldGradGpuKernel() = default;

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    // 'd_batch_mean', 'd_batch_std', 'x', 'batch_mean', 'batch_std', 'current_step'
    T *d_batch_mean = GetDeviceAddress<T>(inputs, 0);
    T *d_batch_std = GetDeviceAddress<T>(inputs, 1);
    T *x = GetDeviceAddress<T>(inputs, 2);
    T *batch_mean = GetDeviceAddress<T>(inputs, 3);
    T *batch_std = GetDeviceAddress<T>(inputs, 4);
    int *current_step = GetDeviceAddress<int>(inputs, 5);
    int current_step_host[1];
    CHECK_CUDA_RET_WITH_ERROR(kernel_node_,
                              cudaMemcpyAsync(current_step_host, current_step, sizeof(int), cudaMemcpyDeviceToHost,
                                              reinterpret_cast<cudaStream_t>(stream_ptr)),
                              "Copy gpu memoy failed.");
    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_, cudaDeviceSynchronize(), "cudaDeviceSyncFailed");
    T *dx = GetDeviceAddress<T>(outputs, 0);

    if (!is_training_ || current_step_host[0] >= freeze_bn_) {
      ThrustFillWith(dx, batch_ * channel_ * height_ * width_, 0.f, reinterpret_cast<cudaStream_t>(stream_ptr));
      return true;
    }
    CalBatchNormFoldGrad(d_batch_mean, d_batch_std, x, batch_mean, batch_std, batch_, channel_, height_, width_, dx,
                         reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    kernel_node_ = kernel_node;
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != INPUT_NUM) {
      MS_LOG(ERROR) << "Input number is " << input_num << ", but BatchNormFoldGrad GpuKernel OP needs " << INPUT_NUM
                    << " inputs.";
      return false;
    }

    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(ERROR) << "Output number is " << output_num << ", but BatchNormFoldGrad GpuKernel OP needs 4 output.";
      return false;
    }

    auto prim = AnfAlgo::GetCNodePrimitive(kernel_node);
    MS_EXCEPTION_IF_NULL(prim);
    epsilon_ = GetValue<T>(prim->GetAttr("epsilon"));
    is_training_ = GetValue<bool>(prim->GetAttr("is_training"));
    freeze_bn_ = static_cast<int>(GetValue<int64_t>(prim->GetAttr("freeze_bn")));

    auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 2);
    is_null_input_ = CHECK_NULL_INPUT(input_shape);
    if (is_null_input_) {
      MS_LOG(WARNING) << "For 'BatchNormFoldGradGpuKernel', input is null";
      InitSizeLists();
      return true;
    }
    if (input_shape.size() != 4) {
      MS_LOG(ERROR) << "Input shape is " << input_shape.size()
                    << ", but BatchNormFoldGrad GpuKernel OP needs 4DTensor input.";
      return false;
    }
    batch_ = input_shape[0];
    channel_ = input_shape[1];
    height_ = input_shape[2];
    width_ = input_shape[3];

    input_size_ = sizeof(T) * batch_ * channel_ * height_ * width_;
    channel_size_ = sizeof(T) * channel_;

    InitSizeLists();
    return true;
  }

 protected:
  void InitSizeLists() override {
    // 'd_batch_mean', 'd_batch_std', 'x', 'batch_mean', 'batch_std', 'current_step'
    input_size_list_.push_back(channel_size_);
    input_size_list_.push_back(channel_size_);
    input_size_list_.push_back(input_size_);
    input_size_list_.push_back(channel_size_);
    input_size_list_.push_back(channel_size_);
    input_size_list_.push_back(sizeof(int));
    // 'dx'
    output_size_list_.push_back(input_size_);
  }

 private:
  size_t input_size_;
  size_t channel_size_;
  size_t workspace_size_;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;

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
