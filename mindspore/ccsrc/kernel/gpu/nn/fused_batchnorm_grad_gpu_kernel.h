/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_KERNEL_GPU_NN_FUSED_BATCHNORM_GRAD_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_KERNEL_GPU_NN_FUSED_BATCHNORM_GRAD_GPU_KERNEL_H_

#include <vector>
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/kernel_constants.h"

namespace mindspore {
namespace kernel {
template <typename T>
class FusedBatchNormGradGpuKernel : public GpuKernel {
 public:
  FusedBatchNormGradGpuKernel()
      : batch_(0),
        channel_(0),
        height_(0),
        width_(0),
        mode_(CUDNN_BATCHNORM_SPATIAL),
        epsilon_(10e-5),
        is_null_input_(false),
        x_desc_(nullptr),
        dy_desc_(nullptr),
        dx_desc_(nullptr),
        scale_bias_desc_(nullptr),
        handle_(nullptr),
        cudnn_data_type_(CUDNN_DATA_FLOAT) {}
  ~FusedBatchNormGradGpuKernel() override { DestroyResource(); }

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    VARIABLE_NOT_USED(workspace);
    VARIABLE_NOT_USED(stream_ptr);
    if (is_null_input_) {
      return true;
    }
    auto dy = GetDeviceAddress<T>(inputs, 0);
    auto x = GetDeviceAddress<T>(inputs, 1);
    auto scale = GetDeviceAddress<T>(inputs, 2);
    auto save_mean = GetDeviceAddress<T>(inputs, 3);
    auto save_variance = GetDeviceAddress<T>(inputs, 4);
    auto dx = GetDeviceAddress<T>(outputs, 0);
    auto bn_scale = GetDeviceAddress<T>(outputs, 1);
    auto bn_bias = GetDeviceAddress<T>(outputs, 2);

    const float alpha_data_diff = 1;
    const float beta_data_diff = 0;
    const float alpha_param_diff = 1;
    const float beta_param_diff = 0;
    CHECK_CUDNN_RET_WITH_EXCEPT(
      cudnnBatchNormalizationBackward(handle_, mode_, &alpha_data_diff, &beta_data_diff, &alpha_param_diff,
                                      &beta_param_diff, x_desc_, x, dy_desc_, dy, dx_desc_, dx, scale_bias_desc_, scale,
                                      bn_scale, bn_bias, epsilon_, save_mean, save_variance),
      "Kernel Launch Failed.");
    return true;
  }
  bool Init(const CNodePtr &kernel_node) override {
    InitResource();
    cudnn_data_type_ = kCudnnDtypeMap[TypeIdLabel(AnfAlgo::GetInputDeviceDataType(kernel_node, 0))];
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 5) {
      MS_LOG(EXCEPTION) << "input tensor size is " << input_num << ", FusedBatchNormGradGpuKernel should be 5";
    }

    auto shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    if (shape.size() != 4) {
      MS_LOG(EXCEPTION) << "tensor shape is " << shape.size() << ", FusedBatchNormGradGpuKernel should be 4";
      return false;
    }
    is_null_input_ = CHECK_NULL_INPUT(shape);
    if (is_null_input_) {
      MS_LOG(WARNING) << "FusedBatchNormGradGpuKernel input is null";
      InitSizeLists();
      return true;
    }
    batch_ = SizeToInt(shape[0]);
    channel_ = SizeToInt(shape[1]);
    height_ = SizeToInt(shape[2]);
    width_ = SizeToInt(shape[3]);

    mode_ = CUDNN_BATCHNORM_SPATIAL;
    epsilon_ = GetAttr<float>(kernel_node, "epsilon");

    CHECK_CUDNN_RET_WITH_EXCEPT(
      cudnnSetTensor4dDescriptor(x_desc_, CUDNN_TENSOR_NCHW, cudnn_data_type_, batch_, channel_, height_, width_),
      "Set x desc failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(
      cudnnSetTensor4dDescriptor(dy_desc_, CUDNN_TENSOR_NCHW, cudnn_data_type_, batch_, channel_, height_, width_),
      "Set dy desc failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(
      cudnnSetTensor4dDescriptor(dx_desc_, CUDNN_TENSOR_NCHW, cudnn_data_type_, batch_, channel_, height_, width_),
      "Set dx desc failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(
      cudnnSetTensor4dDescriptor(scale_bias_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, channel_, 1, 1),
      "Set para desc failed");

    InitSizeLists();
    return true;
  }

 protected:
  void InitResource() override {
    handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCudnnHandle();
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnCreateTensorDescriptor(&x_desc_), "Create x desc failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnCreateTensorDescriptor(&dy_desc_), "Create dy desc failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnCreateTensorDescriptor(&dx_desc_), "Create dx desc failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnCreateTensorDescriptor(&scale_bias_desc_), "Create para desc failed");
  }

  void InitSizeLists() override {
    size_t input_size = 0;
    size_t para_size = 0;
    if (!is_null_input_) {
      CHECK_CUDNN_RET_WITH_EXCEPT(cudnnGetTensorSizeInBytes(x_desc_, &input_size), "Get input size failed");
      CHECK_CUDNN_RET_WITH_EXCEPT(cudnnGetTensorSizeInBytes(scale_bias_desc_, &para_size), "Get input size failed");
    }

    input_size_list_.push_back(input_size);
    input_size_list_.push_back(input_size);
    input_size_list_.push_back(para_size);
    input_size_list_.push_back(para_size);
    input_size_list_.push_back(para_size);

    output_size_list_.push_back(input_size);
    output_size_list_.push_back(para_size);
    output_size_list_.push_back(para_size);
  }

 private:
  void DestroyResource() noexcept {
    CHECK_CUDNN_RET_WITH_ERROR(cudnnDestroyTensorDescriptor(scale_bias_desc_), "Destroy para desc failed");
    CHECK_CUDNN_RET_WITH_ERROR(cudnnDestroyTensorDescriptor(dx_desc_), "Destroy dx desc failed");
    CHECK_CUDNN_RET_WITH_ERROR(cudnnDestroyTensorDescriptor(dy_desc_), "Destroy dy desc failed");
    CHECK_CUDNN_RET_WITH_ERROR(cudnnDestroyTensorDescriptor(x_desc_), "Destroy x desc failed");
  }

  int batch_;
  int channel_;
  int height_;
  int width_;

  cudnnBatchNormMode_t mode_;
  double epsilon_;
  bool is_null_input_;
  cudnnTensorDescriptor_t x_desc_;
  cudnnTensorDescriptor_t dy_desc_;
  cudnnTensorDescriptor_t dx_desc_;
  cudnnTensorDescriptor_t scale_bias_desc_;

  cudnnHandle_t handle_;
  cudnnDataType_t cudnn_data_type_;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_KERNEL_GPU_NN_FUSED_BATCHNORM_GRAD_GPU_KERNEL_H_
