/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_KERNEL_GPU_BATCHNORM_FOLD_GPUKERNEL_H_
#define MINDSPORE_CCSRC_KERNEL_GPU_BATCHNORM_FOLD_GPUKERNEL_H_

#include <vector>
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/kernel_constants.h"
#include "kernel/gpu/cuda_impl/batchnorm_fold_impl.cuh"

namespace mindspore {
namespace kernel {
template <typename T>
class BatchNormFoldGpuKernel : public GpuKernel {
 public:
  BatchNormFoldGpuKernel()
      : input_size_(0),
        output_size_(0),
        exp_avg_factor_(0.9),
        epsilon_(1e-12),
        is_training_(true),
        freeze_bn_(0),
        batch_(0),
        channel_(0),
        height_(0),
        width_(0),
        mode_(CUDNN_BATCHNORM_SPATIAL),
        x_desc_(nullptr),
        scale_bias_mean_var_desc_(nullptr),
        handle_(nullptr) {}

  ~BatchNormFoldGpuKernel() override { DestroyResource(); }

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }

  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }

  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, uintptr_t stream_ptr) override {
    (void)workspace;
    auto x = reinterpret_cast<T *>(inputs[0]->addr);
    auto mean = reinterpret_cast<T *>(inputs[1]->addr);
    auto variance = reinterpret_cast<T *>(inputs[2]->addr);
    int *current_step = reinterpret_cast<int *>(inputs[3]->addr);
    int current_step_host[1];
    CHECK_CUDA_RET_WITH_ERROR(cudaMemcpy(current_step_host, current_step, sizeof(int), cudaMemcpyDeviceToHost),
                              "Copy gpu memoy failed.");
    if (x == nullptr) {
      MS_LOG(ERROR) << "BatchNormFoldGpuKernel x is null.";
      return false;
    }
    if (mean == nullptr) {
      MS_LOG(ERROR) << "BatchNormFoldGpuKernel mean is null.";
      return false;
    }
    if (variance == nullptr) {
      MS_LOG(ERROR) << "BatchNormFoldGpuKernel variance is null.";
      return false;
    }
    if (current_step == nullptr) {
      MS_LOG(ERROR) << "BatchNormFoldGpuKernel current_step is null.";
      return false;
    }
    auto batch_mean = reinterpret_cast<T *>(outputs[0]->addr);
    auto batch_std = reinterpret_cast<T *>(outputs[1]->addr);
    auto running_mean = reinterpret_cast<T *>(outputs[2]->addr);
    auto running_std = reinterpret_cast<T *>(outputs[3]->addr);
    auto y = reinterpret_cast<T *>(workspace[0]->addr);

    CHECK_CUDA_RET_WITH_ERROR(cudaMemcpy(running_mean, mean, output_size_, cudaMemcpyDeviceToDevice),
                              "Failed to copy gpu memory.");
    CHECK_CUDA_RET_WITH_ERROR(cudaMemcpy(running_std, variance, output_size_, cudaMemcpyDeviceToDevice),
                              "Failed to copy gpu memory.");
    CalUpdateRunningStd(channel_, epsilon_, running_std, reinterpret_cast<cudaStream_t>(stream_ptr));
    if (!is_training_ || current_step_host[0] >= freeze_bn_) {
      CHECK_CUDA_RET_WITH_ERROR(cudaMemset(batch_mean, 0, output_size_), "Failed to set gpu memory.");
      ThrustFillWith(batch_std, channel_, 1.f, reinterpret_cast<cudaStream_t>(stream_ptr));
      return true;
    }
    const T alpha = 1;
    const T beta = 0;
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnBatchNormalizationForwardTraining(
                                  handle_, mode_, &alpha, &beta, x_desc_, x, x_desc_, y, scale_bias_mean_var_desc_,
                                  mean, mean, exp_avg_factor_, mean, variance, epsilon_, batch_mean, batch_std),
                                "Failed to launch kernel.")
    CalUpdateBatchStd(channel_, batch_std, reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    InitResource();
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 4) {
      MS_LOG(ERROR) << "Input number is " << input_num << " but BatchNormFold GpuKernel OP needs 4 input.";
      return false;
    }

    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 4) {
      MS_LOG(ERROR) << "Output number is " << output_num << ", but BatchNormFold GpuKernel OP needs 4 output.";
      return false;
    }

    T momentum = GetValue<T>(AnfAlgo::GetCNodePrimitive(kernel_node)->GetAttr("momentum"));
    exp_avg_factor_ = 1.0 - momentum;
    epsilon_ = GetValue<T>(AnfAlgo::GetCNodePrimitive(kernel_node)->GetAttr("epsilon"));
    is_training_ = GetValue<bool>(AnfAlgo::GetCNodePrimitive(kernel_node)->GetAttr("is_training"));
    freeze_bn_ = GetValue<int>(AnfAlgo::GetCNodePrimitive(kernel_node)->GetAttr("freeze_bn"));

    auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    if (input_shape.size() != 4) {
      MS_LOG(ERROR) << "Input shape is " << input_shape.size()
                    << ", but BatchNormFold GpuKernel OP needs 4DTensor input.";
      return false;
    }
    batch_ = input_shape[0];
    channel_ = input_shape[1];
    height_ = input_shape[2];
    width_ = input_shape[3];

    input_size_ = sizeof(T) * batch_ * channel_ * height_ * width_;
    output_size_ = sizeof(T) * channel_;

    cudnnDataType_t cudnnDataType = kCudnnDtypeMap[TypeIdLabel(AnfAlgo::GetInputDeviceDataType(kernel_node, 0))];
    CHECK_CUDNN_RET_WITH_EXCEPT(
      cudnnSetTensor4dDescriptor(x_desc_, CUDNN_TENSOR_NCHW, cudnnDataType, batch_, channel_, height_, width_),
      "Set x desc failed");

    CHECK_CUDNN_RET_WITH_EXCEPT(
      cudnnSetTensor4dDescriptor(scale_bias_mean_var_desc_, CUDNN_TENSOR_NCHW, cudnnDataType, 1, channel_, 1, 1),
      "Set para desc failed");

    InitSizeLists();
    return true;
  }

 protected:
  void InitSizeLists() override {
    // x, mean, variance, current_step
    input_size_list_.push_back(input_size_);
    input_size_list_.push_back(output_size_);
    input_size_list_.push_back(output_size_);
    input_size_list_.push_back(sizeof(int));

    // batch_mean, batch_std, running_mean, running_std
    output_size_list_.push_back(output_size_);
    output_size_list_.push_back(output_size_);
    output_size_list_.push_back(output_size_);
    output_size_list_.push_back(output_size_);

    // store y
    workspace_size_list_.push_back(input_size_);
  }

  void InitResource() override {
    handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCudnnHandle();
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnCreateTensorDescriptor(&x_desc_), "Create x desc failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnCreateTensorDescriptor(&scale_bias_mean_var_desc_), "Create para desc failed");
  }

 private:
  void DestroyResource() noexcept {
    CHECK_CUDNN_RET_WITH_ERROR(cudnnDestroyTensorDescriptor(x_desc_), "Destroy x desc failed");
    CHECK_CUDNN_RET_WITH_ERROR(cudnnDestroyTensorDescriptor(scale_bias_mean_var_desc_), "Destroy para desc failed");
  }

  size_t input_size_;
  size_t output_size_;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;

  double exp_avg_factor_;
  double epsilon_;
  bool is_training_;
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

#endif  // MINDSPORE_CCSRC_KERNEL_GPU_BATCHNORM_FOLD_GPUKERNEL_H_
