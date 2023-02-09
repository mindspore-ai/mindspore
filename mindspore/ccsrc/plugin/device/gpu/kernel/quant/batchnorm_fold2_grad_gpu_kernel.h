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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_BATCHNORMFOLD2_GRAD_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_BATCHNORMFOLD2_GRAD_GPU_KERNEL_H_

#include <vector>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/batchnorm_fold2_impl.cuh"
#include "plugin/device/gpu/kernel/quant/quant_op_const.h"

namespace mindspore {
namespace kernel {
constexpr size_t INPUT_NUM = 8;
template <typename T>
class BatchNormFold2GradGpuKernelMod : public DeprecatedNativeGpuKernelMod {
 public:
  BatchNormFold2GradGpuKernelMod()
      : cudnn_handle_(nullptr),
        is_null_input_(false),
        batch_size_(0),
        channel_(0),
        height_(0),
        width_(0),
        freeze_bn_(0) {}

  ~BatchNormFold2GradGpuKernelMod() override { DestroyResource(); }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }

    auto *dout = GetDeviceAddress<T>(inputs, kIndex0);
    auto *x = GetDeviceAddress<T>(inputs, kIndex1);
    auto *gamma = GetDeviceAddress<T>(inputs, kIndex2);
    auto *batch_std = GetDeviceAddress<T>(inputs, kIndex3);
    auto *batch_mean = GetDeviceAddress<T>(inputs, kIndex4);
    auto *running_std = GetDeviceAddress<T>(inputs, kIndex5);
    auto *running_mean = GetDeviceAddress<T>(inputs, kIndex6);
    auto *global_step = GetDeviceAddress<int32_t>(inputs, kIndex7);
    auto *d_batch_std = GetDeviceAddress<T>(outputs, kIndex0);
    auto *d_batch_mean = GetDeviceAddress<T>(outputs, kIndex1);
    auto *d_beta = GetDeviceAddress<T>(outputs, kIndex2);
    auto *d_gamma = GetDeviceAddress<T>(outputs, kIndex3);
    auto *d_x = GetDeviceAddress<T>(outputs, kIndex4);
    auto *tmp = GetDeviceAddress<T>(workspace, kIndex0);
    auto *tmp2 = GetDeviceAddress<T>(workspace, kIndex1);
    auto *reduce_x = GetDeviceAddress<T>(workspace, kIndex2);
    auto *tmp_x = GetDeviceAddress<T>(workspace, kIndex3);

    int32_t current_step_host[1];
    size_t x_size = batch_size_ * channel_ * height_ * width_ * sizeof(T);
    CHECK_CUDA_RET_WITH_ERROR(kernel_node_,
                              cudaMemcpyAsync(current_step_host, global_step, sizeof(int32_t), cudaMemcpyDeviceToHost,
                                              reinterpret_cast<cudaStream_t>(stream_ptr)),
                              "Failed to copy gpu memory.");
    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_, cudaDeviceSynchronize(), "cudaDeviceSyncFailed");
    CHECK_CUDA_RET_WITH_ERROR(
      kernel_node_,
      cudaMemcpyAsync(d_x, dout, x_size, cudaMemcpyDeviceToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
      "Failed to copy gpu memory.");

    BatchNormFold2GradReduce(dout, x, d_beta, tmp, reduce_x, tmp2, tmp_x, batch_size_, channel_, height_, width_,
                             reinterpret_cast<cudaStream_t>(stream_ptr));
    if (current_step_host[0] < freeze_bn_) {
      CalBatchNormFold2GradNotFreezeDxMul(batch_std, running_std, d_x, batch_size_, channel_, height_, width_,
                                          reinterpret_cast<cudaStream_t>(stream_ptr));
      CalBatchNormFold2GradNotFreeze(d_beta, reduce_x, batch_mean, batch_std, running_mean, running_std, gamma, d_gamma,
                                     d_batch_mean, d_batch_std, channel_, reinterpret_cast<cudaStream_t>(stream_ptr));
    } else {
      CalBatchNormFold2GradFreeze(d_beta, reduce_x, batch_mean, batch_std, running_mean, running_std, gamma, d_gamma,
                                  d_batch_mean, d_batch_std, channel_, reinterpret_cast<cudaStream_t>(stream_ptr));
    }
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
    freeze_bn_ = GetValue<int64_t>(prim->GetAttr("freeze_bn"));

    InitSizeLists();
    return true;
  }

 protected:
  void InitResource() override { cudnn_handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCudnnHandle(); }

  void InitSizeLists() override {
    size_t input_size = batch_size_ * channel_ * height_ * width_ * sizeof(T);
    size_t weight_size = channel_ * sizeof(T);
    size_t workspace_size = batch_size_ * channel_ * sizeof(T);
    input_size_list_.push_back(input_size);       // dout
    input_size_list_.push_back(input_size);       // x
    input_size_list_.push_back(weight_size);      // gamma
    input_size_list_.push_back(weight_size);      // batch_std
    input_size_list_.push_back(weight_size);      // batch_mean
    input_size_list_.push_back(weight_size);      // running_std
    input_size_list_.push_back(weight_size);      // running_mean
    input_size_list_.push_back(sizeof(int32_t));  // global_step

    output_size_list_.push_back(weight_size);  // d_batch_std
    output_size_list_.push_back(weight_size);  // d_batch_mean
    output_size_list_.push_back(weight_size);  // d_beta
    output_size_list_.push_back(weight_size);  // d_gamma
    output_size_list_.push_back(input_size);   // d_x

    workspace_size_list_.push_back(workspace_size);  // tmp
    workspace_size_list_.push_back(workspace_size);  // tmp2
    workspace_size_list_.push_back(weight_size);     // reduce_x
    workspace_size_list_.push_back(input_size);      // tmp_x
  }

 private:
  void DestroyResource() noexcept {}

  cudnnHandle_t cudnn_handle_;
  bool is_null_input_;
  size_t batch_size_;
  size_t channel_;
  size_t height_;
  size_t width_;
  int32_t freeze_bn_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_BATCHNORMFOLD2_GRAD_GPU_KERNEL_H_
