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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_BATCH_NORM_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_BATCH_NORM_GPU_KERNEL_H_

#include <map>
#include <string>
#include <utility>
#include <vector>
#include "include/utils/utils.h"
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/kernel_constants.h"

namespace mindspore {
namespace kernel {
class BatchNormGpuKernelMod : public NativeGpuKernelMod {
 public:
  BatchNormGpuKernelMod() { ResetResource(); }
  explicit BatchNormGpuKernelMod(const std::string kernel_name) : kernel_name_(kernel_name) { ResetResource(); }
  ~BatchNormGpuKernelMod() override { DestroyResource(); }

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    cuda_stream_ = reinterpret_cast<cudaStream_t>(stream_ptr);
    return kernel_func_(this, inputs, workspace, outputs);
  }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  void DestroyResource() noexcept override;
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  void InitResource();
  void InitSizeLists();
  void ResetResource() noexcept;
  void SetTensorDescriptor(const Format &format, const ShapeVector &shape);

  template <typename T>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<KernelTensor *> &outputs);
  using BatchNormFunc = std::function<bool(BatchNormGpuKernelMod *, const std::vector<KernelTensor *> &,
                                           const std::vector<KernelTensor *> &, const std::vector<KernelTensor *> &)>;
  BatchNormFunc kernel_func_{};
  static std::map<std::string, std::vector<std::pair<KernelAttr, BatchNormGpuKernelMod::BatchNormFunc>>>
    kernel_attr_map_;

  size_t attr_pos0_{5};
  size_t input_x_size_;
  size_t input_z_size_;
  size_t para_size_;
  size_t output_size_;
  size_t workspace_size_;
  size_t reserve_size_;
  cudnnBatchNormMode_t mode_;
  cudnnBatchNormOps_t bn_ops_;
  string kernel_name_;
  double epsilon_;
  double exp_avg_factor_;
  bool is_train_;
  bool is_null_input_;
  Format format_;
  cudnnTensorDescriptor_t x_desc_;
  cudnnTensorDescriptor_t y_desc_;
  cudnnTensorDescriptor_t z_desc_;
  cudnnTensorDescriptor_t scale_bias_mean_var_desc_;
  cudnnActivationDescriptor_t activation_desc_;

  cudnnHandle_t handle_;
  cudnnDataType_t cudnn_data_type_;
  void *cuda_stream_{nullptr};
  ActivationType activation_type_ = NO_ACTIVATION;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_BATCH_NORM_GPU_KERNEL_H_
