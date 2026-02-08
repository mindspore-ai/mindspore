/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include <map>
#include <string>
#include <utility>
#include <vector>
#include "include/utils/utils.h"
#include "kernel/gpu/cuda_impl/cuda_ops/batchnorm_grad_impl.cuh"
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/kernel_constants.h"

namespace mindspore {
namespace kernel {
class BatchNormGradGpuKernelMod : public NativeGpuKernelMod {
 public:
  BatchNormGradGpuKernelMod() { ResetResource(); }
  explicit BatchNormGradGpuKernelMod(const std::string kernel_name) : kernel_name_(kernel_name) { ResetResource(); }
  ~BatchNormGradGpuKernelMod() override { DestroyResource(); }

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    cuda_stream_ = reinterpret_cast<cudaStream_t>(stream_ptr);
    return kernel_func_(this, inputs, workspace, outputs, stream_ptr);
  }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  void ResetResource() noexcept {
    batch_ = 0;
    channel_ = 0;
    height_ = 0;
    width_ = 0;
    x_size_ = 0;
    attrs_pos0_ = 6;
    para_size_ = 0;
    workspace_size_ = 0;
    reserve_size_ = 0;
    mode_ = CUDNN_BATCHNORM_SPATIAL;
    bn_ops_ = CUDNN_BATCHNORM_OPS_BN;
    epsilon_ = 10e-5;
    is_train_ = false;
    is_null_input_ = false;
    x_desc_ = nullptr;
    y_desc_ = nullptr;
    dy_desc_ = nullptr;
    dx_desc_ = nullptr;
    dz_desc_ = nullptr;
    scale_bias_diff_desc_ = nullptr;
    activation_desc_ = nullptr;
    handle_ = nullptr;
    cudnn_data_type_ = CUDNN_DATA_FLOAT;
    beta_data_diff_ = 0;
  }

  std::vector<KernelAttr> GetOpSupport() override;

 protected:
  void InitResource() override;

  void InitSizeLists();
  void DestroyResource() noexcept override;

 private:
  void SetTensorDescriptor(const Format &format, const ShapeVector &shape);
  template <typename T>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<KernelTensor *> &outputs, void *stream_ptr);
  using BatchNormGradFunc =
    std::function<bool(BatchNormGradGpuKernelMod *, const std::vector<KernelTensor *> &,
                       const std::vector<KernelTensor *> &, const std::vector<KernelTensor *> &, void *)>;
  BatchNormGradFunc kernel_func_{};
  static std::map<std::string, std::vector<std::pair<KernelAttr, BatchNormGradGpuKernelMod::BatchNormGradFunc>>>
    kernel_attr_map_;

  int batch_;
  int channel_;
  int height_;
  int width_;
  size_t attrs_pos0_;
  size_t x_size_;
  size_t para_size_;
  size_t workspace_size_;
  size_t reserve_size_;
  cudnnBatchNormMode_t mode_;
  cudnnBatchNormOps_t bn_ops_;
  string kernel_name_;
  double epsilon_;
  bool is_train_;
  bool is_null_input_;
  Format format_;

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
  void *cuda_stream_{nullptr};
  ActivationType activation_type_ = NO_ACTIVATION;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_BATCH_NORM_GRAD_GPU_KERNEL_H_
