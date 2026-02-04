/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_GPU_NN_BIAS_ADD_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_GPU_NN_BIAS_ADD_GPU_KERNEL_H_
#include <cuda_runtime_api.h>
#include <string>
#include <map>
#include <algorithm>
#include <memory>
#include <vector>
#include <utility>
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/kernel_constants.h"
#include "mindspore/ops/infer/ops_func_impl/bias_add.h"
#include "kernel/gpu/cuda_impl/cuda_ops/bias_add_nhwc.cuh"

namespace mindspore {
namespace kernel {
class BiasAddGpuKernelMod : public NativeGpuKernelMod {
 public:
  BiasAddGpuKernelMod() { InitResource(); }
  ~BiasAddGpuKernelMod() override { DestroyResource(); }

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    return kernel_func_(this, inputs, workspace, outputs, stream_ptr);
  }
  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

 protected:
  void InitResource() override {
    cudnn_handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCudnnHandle();
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateTensorDescriptor(&x_desc_), "cudnnCreateTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateTensorDescriptor(&b_desc_), "cudnnCreateTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateOpTensorDescriptor(&op_desc_),
                                        "cudnnCreateOpTensorDescriptor failed");
  }

  void DestroyResource() noexcept override {
    CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnDestroyTensorDescriptor(x_desc_), "cudnnDestroyTensorDescriptor failed.");
    CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnDestroyTensorDescriptor(b_desc_), "cudnnDestroyTensorDescriptor failed.");
    CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnDestroyOpTensorDescriptor(op_desc_),
                                       "cudnnDestroyOpTensorDescriptor failed.");
  }

  std::vector<KernelAttr> GetOpSupport() override;

  template <typename T>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<KernelTensor *> &outputs, void *stream_ptr);

  using BiasAddLaunchFunc = std::function<bool(BiasAddGpuKernelMod *, const std::vector<kernel::KernelTensor *> &,
                                               const std::vector<kernel::KernelTensor *> &,
                                               const std::vector<kernel::KernelTensor *> &, void *)>;

  std::vector<std::string> format_str_list = {"DEFAULT", "NCHW",  "NHWC", "NHWC4", "HWKC",  "HWCK",  "KCHW",
                                              "CKHW",    "KHWC",  "CHWK", "HW",    "HW4",   "NC",    "NC4",
                                              "NC4HW4",  "NCDHW", "NWC",  "NCW",   "NDHWC", "NC8HW8"};

 private:
  template <typename T>
  bool ComputeNHWC(const T *src_addr, const T *bias_addr, T *output_addr, const size_t num_value,
                   const size_t num_bias);
  BiasAddLaunchFunc kernel_func_;
  static std::vector<std::pair<KernelAttr, BiasAddLaunchFunc>> func_list_;
  cudnnHandle_t cudnn_handle_;
  cudnnDataType_t cudnn_data_type_;
  cudnnTensorDescriptor_t x_desc_;
  cudnnTensorDescriptor_t b_desc_;
  cudnnOpTensorDescriptor_t op_desc_;
  void *cuda_stream_{nullptr};
  bool is_null_input_;
  int64_t format_;
  std::vector<size_t> input_shape_;
  std::vector<size_t> bias_shape_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_GPU_NN_BIAS_ADD_GPU_KERNEL_H_
