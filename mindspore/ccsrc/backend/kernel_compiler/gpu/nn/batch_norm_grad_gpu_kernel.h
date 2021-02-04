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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_BATCH_NORM_GRAD_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_BATCH_NORM_GRAD_GPU_KERNEL_H_

#include <string>
#include <vector>
#include "utils/utils.h"

#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/kernel_constants.h"
#include "backend/kernel_compiler/gpu/cuda_impl/batchnorm_grad_impl.cuh"

namespace mindspore {
namespace kernel {
template <typename T>
class BatchNormGradGpuKernel : public GpuKernel {
 public:
  BatchNormGradGpuKernel()
      : x_size_(0),
        para_size_(0),
        workspace_size_(0),
        reserve_size_(0),
        mode_(CUDNN_BATCHNORM_SPATIAL),
        bn_ops_(CUDNN_BATCHNORM_OPS_BN),
        epsilon_(10e-5),
        is_train_(false),
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
  ~BatchNormGradGpuKernel() override { DestroyResource(); }

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
    auto scale = GetDeviceAddress<float>(inputs, 2);
    auto save_mean = GetDeviceAddress<float>(inputs, 3);
    auto save_variance = GetDeviceAddress<float>(inputs, 4);
    auto reserve_addr = GetDeviceAddress<float>(inputs, 5);
    reserve_size_ = inputs[5]->size;
    void *bias = nullptr;
    T *y = nullptr;
    if (bn_ops_ != CUDNN_BATCHNORM_OPS_BN) {
      bias = GetDeviceAddress<float>(inputs, 6);
      y = GetDeviceAddress<T>(inputs, 7);
    }

    auto dx = GetDeviceAddress<T>(outputs, 0);
    auto dscale = GetDeviceAddress<float>(outputs, 1);
    auto dbias = GetDeviceAddress<float>(outputs, 2);
    T *dz = nullptr;
    if (bn_ops_ == CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION) {
      dz = GetDeviceAddress<T>(outputs, 3);
    }

    void *workspace_addr = nullptr;
    if (workspace_size_ != 0) {
      workspace_addr = GetDeviceAddress<T>(workspace, 0);
    }
    if (is_train_) {
      const float alpha_data_diff = 1;
      const float alpha_param_diff = 1;
      const float beta_param_diff = 0;
      CHECK_CUDNN_RET_WITH_EXCEPT(
        kernel_node_,
        cudnnBatchNormalizationBackwardEx(handle_, mode_, bn_ops_, &alpha_data_diff, &beta_data_diff_,
                                          &alpha_param_diff, &beta_param_diff, x_desc_, x, y_desc_, y, dy_desc_, dy,
                                          dz_desc_, dz, dx_desc_, dx, scale_bias_diff_desc_, scale, bias, dscale, dbias,
                                          epsilon_, save_mean, save_variance, activation_desc_, workspace_addr,
                                          workspace_size_, reserve_addr, reserve_size_),
        "Kernel launch failed");
    } else {
      CalBatchNormGrad(x, dy, scale, save_mean, save_variance, dx, dscale, dbias, epsilon_, batch_, channel_, height_,
                       width_, reinterpret_cast<cudaStream_t>(stream_ptr));
    }
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    kernel_node_ = kernel_node;
    MS_EXCEPTION_IF_NULL(kernel_node);
    std::string kernel_name = AnfAlgo::GetCNodeName(kernel_node);
    if (kernel_name == kBatchNormGradOpName) {
      bn_ops_ = CUDNN_BATCHNORM_OPS_BN;
    } else if (kernel_name == kBatchNormGradWithActivation) {
      bn_ops_ = CUDNN_BATCHNORM_OPS_BN_ACTIVATION;
    } else if (kernel_name == kBatchNormGradWithAddAndActivation) {
      bn_ops_ = CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION;
    } else {
      MS_LOG(EXCEPTION) << "Invalid kernel name: " << kernel_name;
    }

    InitResource();
    mode_ = CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
    epsilon_ = GetAttr<float>(kernel_node, "epsilon");

    cudnn_data_type_ = GetCudnnDataType(TypeIdLabel(AnfAlgo::GetInputDeviceDataType(kernel_node, 0)));
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (bn_ops_ == CUDNN_BATCHNORM_OPS_BN) {
      if (input_num != 6) {
        MS_LOG(EXCEPTION) << "input tensor size is " << input_num << ", " << kernel_name << " should be 6";
      }
    } else {
      if (input_num != 8) {
        MS_LOG(EXCEPTION) << "input tensor size is " << input_num << ", " << kernel_name << "  should be 8";
      }
    }

    auto shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
    if (shape.size() != 4) {
      MS_LOG(EXCEPTION) << "tensor shape is " << shape.size() << ", BatchNormGradGpuKernel should be 4";
    }
    is_null_input_ = CHECK_NULL_INPUT(shape);
    if (is_null_input_) {
      MS_LOG(WARNING) << "BatchNormGradGpuKernel input is null";
      InitSizeLists();
      return true;
    }
    std::string format = AnfAlgo::GetInputFormat(kernel_node, 0);
    auto format_attr = GetAttr<std::string>(kernel_node, "format");
    if (format_attr == kOpFormat_NHWC) {
      format = kOpFormat_NHWC;
    }
    beta_data_diff_ = GetAttrWithDefault(kernel_node, "inplace_algo", std::string("cover")) == "cover" ? 0 : 1;
    SetTensorDescriptor(format, shape);
    InitSizeLists();
    is_train_ = GetAttr<bool>(kernel_node, "is_training");
    return true;
  }

 protected:
  void InitResource() override {
    handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCudnnHandle();
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateTensorDescriptor(&x_desc_), "Create x desc failed");
    if (bn_ops_ != CUDNN_BATCHNORM_OPS_BN) {
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateTensorDescriptor(&y_desc_), "Create y desc failed");
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateActivationDescriptor(&activation_desc_),
                                  "Create activation descriptor failed");
    }
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateTensorDescriptor(&dy_desc_), "Create dy desc failed");

    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateTensorDescriptor(&dx_desc_), "Create dx desc failed");
    if (bn_ops_ == CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION) {
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateTensorDescriptor(&dz_desc_), "Create dz desc failed");
    }
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
    input_size_list_.push_back(para_size_);
    input_size_list_.push_back(para_size_);
    input_size_list_.push_back(para_size_);
    input_size_list_.push_back(reserve_size_);
    if (bn_ops_ != CUDNN_BATCHNORM_OPS_BN) {
      input_size_list_.push_back(para_size_);
      input_size_list_.push_back(x_size_);
    }

    output_size_list_.push_back(x_size_);
    output_size_list_.push_back(para_size_);
    output_size_list_.push_back(para_size_);
    if (bn_ops_ == CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION) {
      output_size_list_.push_back(x_size_);
    }

    workspace_size_list_.push_back(workspace_size_);
  }
  void DestroyResource() noexcept override {
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyTensorDescriptor(x_desc_), "Destroy x desc failed");
    if (bn_ops_ != CUDNN_BATCHNORM_OPS_BN) {
      CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyTensorDescriptor(y_desc_), "Destroy y desc failed");
      CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyActivationDescriptor(activation_desc_),
                                 "Destroy activation descriptor failed");
    }
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyTensorDescriptor(dy_desc_), "Destroy dy desc failed");

    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyTensorDescriptor(dx_desc_), "Destroy dx desc failed");
    if (bn_ops_ == CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION) {
      CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyTensorDescriptor(dz_desc_), "Destroy z desc failed");
    }
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyTensorDescriptor(scale_bias_diff_desc_),
                               "Destroy para desc failed");
  }

 private:
  void SetTensorDescriptor(const std::string &format, const std::vector<size_t> &shape) {
    cudnnTensorFormat_t cudnn_format;
    if (format == kOpFormat_NHWC) {
      batch_ = SizeToInt(shape[0]);
      height_ = SizeToInt(shape[1]);
      width_ = SizeToInt(shape[2]);
      channel_ = SizeToInt(shape[3]);
      cudnn_format = CUDNN_TENSOR_NHWC;
    } else {
      batch_ = SizeToInt(shape[0]);
      channel_ = SizeToInt(shape[1]);
      height_ = SizeToInt(shape[2]);
      width_ = SizeToInt(shape[3]);
      cudnn_format = CUDNN_TENSOR_NCHW;
    }

    CHECK_CUDNN_RET_WITH_EXCEPT(
      kernel_node_,
      cudnnSetTensor4dDescriptor(x_desc_, cudnn_format, cudnn_data_type_, batch_, channel_, height_, width_),
      "Set x desc failed");

    if (bn_ops_ != CUDNN_BATCHNORM_OPS_BN) {
      CHECK_CUDNN_RET_WITH_EXCEPT(
        kernel_node_,
        cudnnSetTensor4dDescriptor(y_desc_, cudnn_format, cudnn_data_type_, batch_, channel_, height_, width_),
        "Set z desc failed");
    }

    CHECK_CUDNN_RET_WITH_EXCEPT(
      kernel_node_,
      cudnnSetTensor4dDescriptor(dy_desc_, cudnn_format, cudnn_data_type_, batch_, channel_, height_, width_),
      "Set dy desc failed");

    CHECK_CUDNN_RET_WITH_EXCEPT(
      kernel_node_,
      cudnnSetTensor4dDescriptor(dx_desc_, cudnn_format, cudnn_data_type_, batch_, channel_, height_, width_),
      "Set dx desc failed");

    if (bn_ops_ == CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION) {
      CHECK_CUDNN_RET_WITH_EXCEPT(
        kernel_node_,
        cudnnSetTensor4dDescriptor(dz_desc_, cudnn_format, cudnn_data_type_, batch_, channel_, height_, width_),
        "Set z desc failed");
    }

    CHECK_CUDNN_RET_WITH_EXCEPT(
      kernel_node_,
      cudnnSetTensor4dDescriptor(scale_bias_diff_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, channel_, 1, 1),
      "Set para desc failed");

    if (bn_ops_ != CUDNN_BATCHNORM_OPS_BN) {
      CHECK_CUDNN_RET_WITH_EXCEPT(
        kernel_node_,
        cudnnSetActivationDescriptor(activation_desc_, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0.0),
        "cudnnSetActivationDescriptor failed");
    }
  }
  int batch_;
  int channel_;
  int height_;
  int width_;
  size_t x_size_;
  size_t para_size_;
  size_t workspace_size_;
  size_t reserve_size_;
  cudnnBatchNormMode_t mode_;
  cudnnBatchNormOps_t bn_ops_;
  double epsilon_;
  bool is_train_;
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
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_FUSED_BATCH_NORM_GRAD_EX_GPU_KERNEL_H_
