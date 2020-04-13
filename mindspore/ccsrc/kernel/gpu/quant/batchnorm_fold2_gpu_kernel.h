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

#ifndef MINDSPORE_CCSRC_KERNEL_GPU_NN_BATCHNORMFOLD2_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_KERNEL_GPU_NN_BATCHNORMFOLD2_GPU_KERNEL_H_

#include <vector>
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/cuda_impl/batchnorm_fold2_impl.cuh"

namespace mindspore {
namespace kernel {
template <typename T>
class BatchNormFold2GpuKernel : public GpuKernel {
 public:
  BatchNormFold2GpuKernel()
      : cudnn_handle_(nullptr),
        is_null_input_(false),
        batch_size_(0),
        channel_(0),
        height_(0),
        width_(0),
        freeze_bn_(0) {}

  ~BatchNormFold2GpuKernel() override { DestroyResource(); }

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }

  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }

  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, uintptr_t stream_ptr) override {
    if (is_null_input_) {
      return true;
    }

    auto *input = GetDeviceAddress<T>(inputs, 0);
    auto *beta = GetDeviceAddress<T>(inputs, 1);
    auto *gamma = GetDeviceAddress<T>(inputs, 2);
    auto *batch_std = GetDeviceAddress<T>(inputs, 3);
    auto *batch_mean = GetDeviceAddress<T>(inputs, 4);
    auto *running_std = GetDeviceAddress<T>(inputs, 5);
    auto *running_mean = GetDeviceAddress<T>(inputs, 6);
    auto *global_step = GetDeviceAddress<int32_t>(inputs, 7);
    auto *output = GetDeviceAddress<T>(outputs, 0);

    BatchNormFold2Forward(input, beta, gamma, batch_std, batch_mean, running_std, running_mean, global_step, output,
                          freeze_bn_, batch_size_, channel_, height_, width_,
                          reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    InitResource();

    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 8) {
      MS_LOG(ERROR) << "Argument number is " << input_num << ", but BatchNormFold2GpuKernel needs 8.";
      return false;
    }

    auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    is_null_input_ = CHECK_NULL_INPUT(input_shape);
    if (is_null_input_) {
      MS_LOG(WARNING) << "BatchNormFold2GpuKernel input is null";
      InitSizeLists();
      return true;
    }

    if (input_shape.size() != 4) {
      MS_LOG(ERROR) << "BatchNormFold2GpuKernel input shape needs (N,C,H,W).";
      return false;
    }
    batch_size_ = input_shape[0];
    channel_ = input_shape[1];
    height_ = input_shape[2];
    width_ = input_shape[3];
    freeze_bn_ = GetValue<int32_t>(AnfAlgo::GetCNodePrimitive(kernel_node)->GetAttr("freeze_bn"));

    InitSizeLists();
    return true;
  }

 protected:
  void InitResource() override { cudnn_handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCudnnHandle(); }

  void InitSizeLists() override {
    size_t input_size = batch_size_ * channel_ * height_ * width_ * sizeof(T);
    size_t weight_size = channel_ * sizeof(T);
    input_size_list_.push_back(input_size);
    input_size_list_.push_back(weight_size);      // beta
    input_size_list_.push_back(weight_size);      // gamma
    input_size_list_.push_back(weight_size);      // batch_std
    input_size_list_.push_back(weight_size);      // batch_mean
    input_size_list_.push_back(weight_size);      // running_std
    input_size_list_.push_back(weight_size);      // running_mean
    input_size_list_.push_back(sizeof(int32_t));  // global_step

    output_size_list_.push_back(input_size);

    size_t workspace_size = 0;
    workspace_size_list_.push_back(workspace_size);
  }

 private:
  void DestroyResource() noexcept {}

  cudnnHandle_t cudnn_handle_;
  bool is_null_input_;
  size_t batch_size_;
  size_t channel_;
  size_t height_;
  size_t width_;
  size_t freeze_bn_;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};

}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_KERNEL_GPU_NN_BATCHNORMFOLD2_GPU_KERNEL_H_
