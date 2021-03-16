/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_INSTANCE_NORM_GRAD_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_INSTANCE_NORM_GRAD_GPU_KERNEL_H_

#include <string>
#include <vector>
#include "utils/utils.h"

#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/kernel_constants.h"
#include "backend/kernel_compiler/gpu/cuda_impl/instance_norm_impl.cuh"

namespace mindspore {
namespace kernel {
template <typename T>
class InstanceNormGradGpuKernel : public GpuKernel {
 public:
  InstanceNormGradGpuKernel()
      : x_size_(0),
        para_size_(0),
        workspace_size_(0),
        mode_(CUDNN_BATCHNORM_SPATIAL),
        bn_ops_(CUDNN_BATCHNORM_OPS_BN),
        epsilon_(10e-5),
        is_null_input_(false),
        x_desc_(nullptr),
        y_desc_(nullptr),
        dy_desc_(nullptr),
        dx_desc_(nullptr),
        dz_desc_(nullptr),
        scale_bias_diff_desc_(nullptr),
        activation_desc_(nullptr),
        handle_(nullptr),
        cudnn_data_type_(CUDNN_DATA_FLOAT),
        beta_data_diff_(0) {}
  ~InstanceNormGradGpuKernel() override { DestroyResource(); }

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
    auto gamma = GetDeviceAddress<float>(inputs, 2);
    auto save_mean = GetDeviceAddress<float>(inputs, 3);
    auto save_variance = GetDeviceAddress<float>(inputs, 4);
    void *beta = nullptr;
    T *y = nullptr;

    auto dx = GetDeviceAddress<T>(outputs, 0);
    auto dgamma = GetDeviceAddress<float>(outputs, 1);
    auto dbeta = GetDeviceAddress<float>(outputs, 2);
    T *dz = nullptr;

    float *ws_gamma = GetDeviceAddress<float>(workspace, 0);
    float *ws_dgamma = GetDeviceAddress<float>(workspace, 1);
    float *ws_dbeta = GetDeviceAddress<float>(workspace, 2);
    void *workspace_addr = nullptr;
    if (workspace_size_ != 0) {
      workspace_addr = GetDeviceAddress<T>(workspace, 3);
    }

    size_t N = input_shape_[0];
    size_t C = input_shape_[1];
    CopyMemDevice2Device(N, C, gamma, nullptr, nullptr, nullptr, ws_gamma, nullptr, nullptr, nullptr,
                         reinterpret_cast<cudaStream_t>(stream_ptr));
    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_, cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream_ptr)),
                               "cudaStreamSynchronized failed");

    const float alpha_data_diff = 1;
    const float alpha_param_diff = 1;
    const float beta_param_diff = 0;
    float *reserve_addr = nullptr;
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                cudnnBatchNormalizationBackwardEx(
                                  handle_, mode_, bn_ops_, &alpha_data_diff, &beta_data_diff_, &alpha_param_diff,
                                  &beta_param_diff, x_desc_, x, y_desc_, y, dy_desc_, dy, dz_desc_, dz, dx_desc_, dx,
                                  scale_bias_diff_desc_, ws_gamma, beta, ws_dgamma, ws_dbeta, epsilon_, save_mean,
                                  save_variance, activation_desc_, workspace_addr, workspace_size_, reserve_addr, 0),
                                "Kernel launch failed");
    ComputeMean(N, C, dgamma, dbeta, ws_dgamma, ws_dbeta, reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    kernel_node_ = kernel_node;
    MS_EXCEPTION_IF_NULL(kernel_node);
    std::string kernel_name = AnfAlgo::GetCNodeName(kernel_node);
    bn_ops_ = CUDNN_BATCHNORM_OPS_BN;

    InitResource();
    mode_ = CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
    epsilon_ = GetAttr<float>(kernel_node, "epsilon");

    cudnn_data_type_ = GetCudnnDataType(TypeIdLabel(AnfAlgo::GetInputDeviceDataType(kernel_node, 0)));
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 5) {
      MS_LOG(EXCEPTION) << "input tensor size is " << input_num << ", " << kernel_name << " should be 5";
    }

    input_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
    if (input_shape_.size() != 4) {
      MS_LOG(EXCEPTION) << "tensor shape is " << input_shape_.size() << ", InstanceNormGradGpuKernel should be 4";
    }
    is_null_input_ = CHECK_NULL_INPUT(input_shape_);
    if (is_null_input_) {
      MS_LOG(WARNING) << "InstanceNormGradGpuKernel input is null";
      InitSizeLists();
      return true;
    }
    beta_data_diff_ = GetAttrWithDefault(kernel_node, "inplace_algo", std::string("cover")) == "cover" ? 0 : 1;
    SetTensorDescriptor();
    InitSizeLists();
    return true;
  }

 protected:
  void InitResource() override {
    handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCudnnHandle();
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateTensorDescriptor(&x_desc_), "Create x desc failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateTensorDescriptor(&dy_desc_), "Create dy desc failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateTensorDescriptor(&dx_desc_), "Create dx desc failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateTensorDescriptor(&scale_bias_diff_desc_),
                                "Create para desc failed");
  }

  void InitSizeLists() override {
    if (!is_null_input_) {
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnGetTensorSizeInBytes(x_desc_, &x_size_), "Get x size failed");
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnGetTensorSizeInBytes(scale_bias_diff_desc_, &para_size_),
                                  "Get para size failed");
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                  cudnnGetBatchNormalizationBackwardExWorkspaceSize(
                                    handle_, mode_, bn_ops_, x_desc_, y_desc_, dy_desc_, dz_desc_, dx_desc_,
                                    scale_bias_diff_desc_, activation_desc_, &workspace_size_),
                                  "cudnnGetBatchNormalizationBackwardExWorkspaceSize failed");
    }

    input_size_list_.push_back(x_size_);
    input_size_list_.push_back(x_size_);
    input_size_list_.push_back(input_shape_[1]);
    input_size_list_.push_back(para_size_);
    input_size_list_.push_back(para_size_);

    output_size_list_.push_back(x_size_);
    output_size_list_.push_back(x_size_);
    output_size_list_.push_back(x_size_);

    workspace_size_list_.push_back(para_size_);  // ws gamma
    workspace_size_list_.push_back(para_size_);  // ws dgamma
    workspace_size_list_.push_back(para_size_);  // ws dbeta
    workspace_size_list_.push_back(workspace_size_);
  }
  void DestroyResource() noexcept override {
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyTensorDescriptor(x_desc_), "Destroy x desc failed");
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyTensorDescriptor(dy_desc_), "Destroy dy desc failed");
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyTensorDescriptor(dx_desc_), "Destroy dx desc failed");
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyTensorDescriptor(scale_bias_diff_desc_),
                               "Destroy para desc failed");
  }

 private:
  void SetTensorDescriptor() {
    int batch, channel, height, width;
    batch = 1;
    channel = SizeToInt(input_shape_[0]) * SizeToInt(input_shape_[1]);
    height = SizeToInt(input_shape_[2]);
    width = SizeToInt(input_shape_[3]);
    cudnnTensorFormat_t cudnn_format = CUDNN_TENSOR_NCHW;

    CHECK_CUDNN_RET_WITH_EXCEPT(
      kernel_node_, cudnnSetTensor4dDescriptor(x_desc_, cudnn_format, cudnn_data_type_, batch, channel, height, width),
      "Set x desc failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(
      kernel_node_, cudnnSetTensor4dDescriptor(dy_desc_, cudnn_format, cudnn_data_type_, batch, channel, height, width),
      "Set dy desc failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(
      kernel_node_, cudnnSetTensor4dDescriptor(dx_desc_, cudnn_format, cudnn_data_type_, batch, channel, height, width),
      "Set dx desc failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(
      kernel_node_,
      cudnnSetTensor4dDescriptor(scale_bias_diff_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, channel, 1, 1),
      "Set para desc failed");
  }

  size_t x_size_;
  size_t para_size_;
  size_t workspace_size_;
  cudnnBatchNormMode_t mode_;
  cudnnBatchNormOps_t bn_ops_;
  double epsilon_;
  bool is_null_input_;

  cudnnTensorDescriptor_t x_desc_;
  cudnnTensorDescriptor_t y_desc_;
  cudnnTensorDescriptor_t dy_desc_;
  cudnnTensorDescriptor_t dx_desc_;
  cudnnTensorDescriptor_t dz_desc_;
  cudnnTensorDescriptor_t scale_bias_diff_desc_;
  cudnnActivationDescriptor_t activation_desc_;

  cudnnHandle_t handle_;
  cudnnDataType_t cudnn_data_type_;
  float beta_data_diff_;
  std::vector<size_t> input_shape_;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_INSTANCE_NORM_GRAD_GPU_KERNEL_H_
