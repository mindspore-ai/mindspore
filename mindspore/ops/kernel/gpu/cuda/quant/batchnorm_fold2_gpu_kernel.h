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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_BATCHNORMFOLD2_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_BATCHNORMFOLD2_GPU_KERNEL_H_

#include <vector>
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/cuda_impl/cuda_ops/batchnorm_fold2_impl.cuh"
#include "kernel/gpu/cuda/quant/quant_op_const.h"

namespace mindspore {
namespace kernel {
template <typename T>
class BatchNormFold2GpuKernelMod : public NativeGpuKernelMod {
 public:
  BatchNormFold2GpuKernelMod()
      : cudnn_handle_(nullptr),
        is_null_input_(false),
        batch_size_(0),
        channel_(0),
        height_(0),
        width_(0),
        freeze_bn_(0) {}

  ~BatchNormFold2GpuKernelMod() override { DestroyResource(); }

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }

    auto *input = GetDeviceAddress<T>(inputs, kIndex0);
    auto *beta = GetDeviceAddress<T>(inputs, kIndex1);
    auto *gamma = GetDeviceAddress<T>(inputs, kIndex2);
    auto *batch_std = GetDeviceAddress<T>(inputs, kIndex3);
    auto *batch_mean = GetDeviceAddress<T>(inputs, kIndex4);
    auto *running_std = GetDeviceAddress<T>(inputs, kIndex5);
    auto *running_mean = GetDeviceAddress<T>(inputs, kIndex6);
    auto *global_step = GetDeviceAddress<int32_t>(inputs, kIndex7);
    auto *output = GetDeviceAddress<T>(outputs, kIndex0);

    auto status = BatchNormFold2Forward(input, beta, gamma, batch_std, batch_mean, running_std, running_mean,
                                        global_step, output, freeze_bn_, batch_size_, channel_, height_, width_,
                                        reinterpret_cast<cudaStream_t>(stream_ptr));
    CHECK_CUDA_STATUS(status, kernel_name_);
    return true;
  }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override {
    InitResource();
    freeze_bn_ = static_cast<int>(GetValue<int64_t>(primitive_->GetAttr("freeze_bn")));
    return true;
  }

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override {
    output_size_list_.clear();
    workspace_size_list_.clear();
    auto shape_signed = inputs[kIndex0]->GetShapeVector();
    if (IsDynamic(shape_signed)) {
      return KRET_UNKNOWN_SHAPE;
    }
    auto input_shape = Convert2SizeTClipNeg(shape_signed);
    is_null_input_ = CHECK_SHAPE_NULL(input_shape, kernel_name_, "input");
    if (is_null_input_) {
      output_size_list_.push_back(batch_size_ * channel_ * height_ * width_ * sizeof(T));
      return KRET_UNKNOWN_SHAPE;
    }

    if (input_shape.size() != kSize4) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of input should be 4, but got "
                        << input_shape.size();
    }
    batch_size_ = input_shape[kIndex0];
    channel_ = input_shape[kIndex1];
    height_ = input_shape[kIndex2];
    width_ = input_shape[kIndex3];

    output_size_list_.push_back(batch_size_ * channel_ * height_ * width_ * sizeof(T));
    return KRET_OK;
  }

 protected:
  void InitResource() override { cudnn_handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCudnnHandle(); }

 private:
  cudnnHandle_t cudnn_handle_;
  bool is_null_input_;
  size_t batch_size_;
  size_t channel_;
  size_t height_;
  size_t width_;
  size_t freeze_bn_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_BATCHNORMFOLD2_GPU_KERNEL_H_
