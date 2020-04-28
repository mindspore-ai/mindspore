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

#ifndef MINDSPORE_CCSRC_KERNEL_GPU_NN_FUSED_BATCH_NORM_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_KERNEL_GPU_NN_FUSED_BATCH_NORM_GPU_KERNEL_H_

#include <vector>
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/kernel_constants.h"

namespace mindspore {
namespace kernel {
template <typename T>
class FusedBatchNormGpuKernel : public GpuKernel {
 public:
  FusedBatchNormGpuKernel()
      : batch_(0),
        channel_(0),
        height_(0),
        width_(0),
        mode_(CUDNN_BATCHNORM_SPATIAL),
        epsilon_(10e-5),
        exp_avg_factor_(0.1),
        is_train_(false),
        is_null_input_(false),
        x_desc_(nullptr),
        y_desc_(nullptr),
        scale_bias_mean_var_desc_(nullptr),
        handle_(nullptr),
        cudnn_data_type_(CUDNN_DATA_FLOAT) {}
  ~FusedBatchNormGpuKernel() override { DestroyResource(); }

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, uintptr_t stream_ptr) override {
    VARIABLE_NOT_USED(workspace);
    VARIABLE_NOT_USED(stream_ptr);
    if (is_null_input_) {
      return true;
    }
    auto x = GetDeviceAddress<T>(inputs, 0);
    auto scale = GetDeviceAddress<T>(inputs, 1);
    auto bias = GetDeviceAddress<T>(inputs, 2);
    auto runing_mean = GetDeviceAddress<T>(inputs, 3);
    auto runnig_variance = GetDeviceAddress<T>(inputs, 4);
    auto y = GetDeviceAddress<T>(outputs, 0);

    const float alpha = 1;
    const float beta = 0;
    if (is_train_) {
      auto save_mean = GetDeviceAddress<T>(outputs, 3);
      auto save_variance = GetDeviceAddress<T>(outputs, 4);
      CHECK_CUDNN_RET_WITH_EXCEPT(
        cudnnBatchNormalizationForwardTraining(handle_, mode_, &alpha, &beta, x_desc_, x, y_desc_, y,
                                               scale_bias_mean_var_desc_, scale, bias, exp_avg_factor_, runing_mean,
                                               runnig_variance, epsilon_, save_mean, save_variance),
        "Kernel launch failed");
    } else {
      CHECK_CUDNN_RET_WITH_EXCEPT(cudnnBatchNormalizationForwardInference(handle_, mode_, &alpha, &beta, x_desc_, x,
                                                                          y_desc_, y, scale_bias_mean_var_desc_, scale,
                                                                          bias, runing_mean, runnig_variance, epsilon_),
                                  "Kernel launch failed");
    }
    return true;
  }
  bool Init(const CNodePtr &kernel_node) override {
    InitResource();
    cudnn_data_type_ = kCudnnDtypeMap[TypeIdLabel(AnfAlgo::GetInputDeviceDataType(kernel_node, 0))];
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 5) {
      MS_LOG(EXCEPTION) << "input tensor size is " << input_num << ", FusedBatchNormGpuKernel should be 5";
    }

    auto shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    if (shape.size() != 4) {
      MS_LOG(EXCEPTION) << "tensor shape is " << shape.size() << ", FusedBatchNormGpuKernel should be >= 4";
    }
    is_null_input_ = CHECK_NULL_INPUT(shape);
    if (is_null_input_) {
      MS_LOG(WARNING) << "FusedBatchNormGpuKernel input is null";
      InitSizeLists();
      return true;
    }
    batch_ = SizeToInt(shape[0]);
    channel_ = SizeToInt(shape[1]);
    height_ = SizeToInt(shape[2]);
    width_ = SizeToInt(shape[3]);

    mode_ = CUDNN_BATCHNORM_SPATIAL;
    epsilon_ = GetAttr<float>(kernel_node, "epsilon");
    // P.FusedBatchNorm is used for training; P.BatchNorm is used for inference
    auto node_name = AnfAlgo::GetCNodeName(kernel_node);
    if (node_name == "FusedBatchNorm") {
      is_train_ = true;
      exp_avg_factor_ = GetAttr<float>(kernel_node, "momentum");
    }

    CHECK_CUDNN_RET_WITH_EXCEPT(
      cudnnSetTensor4dDescriptor(x_desc_, CUDNN_TENSOR_NCHW, cudnn_data_type_, batch_, channel_, height_, width_),
      "Set x desc failed");

    CHECK_CUDNN_RET_WITH_EXCEPT(
      cudnnSetTensor4dDescriptor(y_desc_, CUDNN_TENSOR_NCHW, cudnn_data_type_, batch_, channel_, height_, width_),
      "Set y desc failed");

    CHECK_CUDNN_RET_WITH_EXCEPT(
      cudnnSetTensor4dDescriptor(scale_bias_mean_var_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, channel_, 1, 1),
      "Set para desc failed");

    InitSizeLists();

    return true;
  }

 protected:
  void InitResource() override {
    handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCudnnHandle();
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnCreateTensorDescriptor(&x_desc_), "Create x desc failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnCreateTensorDescriptor(&y_desc_), "Create y desc failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnCreateTensorDescriptor(&scale_bias_mean_var_desc_), "Create para desc failed");
  }
  void InitSizeLists() override {
    size_t input_size = 0;
    size_t para_size = 0;
    size_t output_size = 0;
    if (!is_null_input_) {
      CHECK_CUDNN_RET_WITH_EXCEPT(cudnnGetTensorSizeInBytes(x_desc_, &input_size), "Get input size failed");
      CHECK_CUDNN_RET_WITH_EXCEPT(cudnnGetTensorSizeInBytes(scale_bias_mean_var_desc_, &para_size),
                                  "Get para size failed");
      CHECK_CUDNN_RET_WITH_EXCEPT(cudnnGetTensorSizeInBytes(y_desc_, &output_size), "Get para size failed");
    }
    input_size_list_.push_back(input_size);
    input_size_list_.push_back(para_size);  // scale
    input_size_list_.push_back(para_size);  // bias
    input_size_list_.push_back(para_size);  // mean
    input_size_list_.push_back(para_size);  // variance

    output_size_list_.push_back(output_size);
    output_size_list_.push_back(para_size);  // running mean
    output_size_list_.push_back(para_size);  // running variance
    output_size_list_.push_back(para_size);  // save mean
    output_size_list_.push_back(para_size);  // save variance
    if (!is_train_) {
      output_size_list_.push_back(para_size);  // reserve
    }
    return;
  }

 private:
  void DestroyResource() noexcept {
    CHECK_CUDNN_RET_WITH_ERROR(cudnnDestroyTensorDescriptor(x_desc_), "Destroy x desc failed");
    CHECK_CUDNN_RET_WITH_ERROR(cudnnDestroyTensorDescriptor(y_desc_), "Destroy y desc failed");
    CHECK_CUDNN_RET_WITH_ERROR(cudnnDestroyTensorDescriptor(scale_bias_mean_var_desc_), "Destroy para desc failed");
  }

  int batch_;
  int channel_;
  int height_;
  int width_;
  cudnnBatchNormMode_t mode_;
  double epsilon_;
  double exp_avg_factor_;
  bool is_train_;
  bool is_null_input_;
  cudnnTensorDescriptor_t x_desc_;
  cudnnTensorDescriptor_t y_desc_;
  cudnnTensorDescriptor_t scale_bias_mean_var_desc_;
  cudnnHandle_t handle_;
  cudnnDataType_t cudnn_data_type_;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_KERNEL_GPU_NN_FUSED_BATCH_NORM_GPU_KERNEL_H_
