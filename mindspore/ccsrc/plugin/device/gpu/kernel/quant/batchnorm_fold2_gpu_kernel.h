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
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/batchnorm_fold2_impl.cuh"
#include "plugin/device/gpu/kernel/quant/quant_op_const.h"

namespace mindspore {
namespace kernel {
constexpr size_t INPUT_NUM = 8;
template <typename T>
class BatchNormFold2GpuKernelMod : public DeprecatedNativeGpuKernelMod {
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

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
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

    BatchNormFold2Forward(input, beta, gamma, batch_std, batch_mean, running_std, running_mean, global_step, output,
                          freeze_bn_, batch_size_, channel_, height_, width_,
                          reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    auto kernel_name = common::AnfAlgo::GetCNodeName(kernel_node);
    kernel_node_ = kernel_node;
    InitResource();

    size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != INPUT_NUM) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the number of inputs should be " << INPUT_NUM << ", but got "
                        << input_num;
    }

    auto shape_signed = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    if (IsDynamic(shape_signed)) {
      return true;
    }
    auto input_shape = Convert2SizeTClipNeg(shape_signed);
    is_null_input_ = CHECK_SHAPE_NULL(input_shape, kernel_name, "input");
    if (is_null_input_) {
      InitSizeLists();
      return true;
    }

    if (input_shape.size() != kSize4) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the dimension of input should be 4, but got "
                        << input_shape.size();
    }
    batch_size_ = input_shape[kIndex0];
    channel_ = input_shape[kIndex1];
    height_ = input_shape[kIndex2];
    width_ = input_shape[kIndex3];
    auto prim = common::AnfAlgo::GetCNodePrimitive(kernel_node);
    MS_EXCEPTION_IF_NULL(prim);
    freeze_bn_ = static_cast<int>(GetValue<int64_t>(prim->GetAttr("freeze_bn")));

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
  }

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
