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
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/kernel_constants.h"
#include "mindspore/core/ops/bias_add.h"

namespace mindspore {
namespace kernel {
class BiasAddGpuKernelMod : public NativeGpuKernelMod {
 public:
  BiasAddGpuKernelMod() {}
  ~BiasAddGpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    return kernel_func_(this, inputs, workspace, outputs, stream_ptr);
  }
  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs,
             const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) override;

 protected:
  void InitResource() override {
    cudnn_handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCudnnHandle();
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateTensorDescriptor(&x_desc_), "cudnnCreateTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateTensorDescriptor(&b_desc_), "cudnnCreateTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateOpTensorDescriptor(&op_desc_),
                                        "cudnnCreateOpTensorDescriptor failed");
  }

  std::vector<KernelAttr> GetOpSupport() override;

  template <typename T>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<AddressPtr> &outputs, void *stream_ptr);

  using BiasAddLaunchFunc =
    std::function<bool(BiasAddGpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                       const std::vector<kernel::AddressPtr> &, const std::vector<kernel::AddressPtr> &, void *)>;

  std::vector<std::string> format_str_list = {"DEFAULT", "NCHW",  "NHWC", "NHWC4", "HWKC",  "HWCK",  "KCHW",
                                              "CKHW",    "KHWC",  "CHWK", "HW",    "HW4",   "NC",    "NC4",
                                              "NC4HW4",  "NCDHW", "NWC",  "NCW",   "NDHWC", "NC8HW8"};

 private:
  BiasAddLaunchFunc kernel_func_;
  static std::vector<std::pair<KernelAttr, BiasAddLaunchFunc>> func_list_;
  cudnnHandle_t cudnn_handle_;
  cudnnDataType_t cudnn_data_type_;
  cudnnTensorDescriptor_t x_desc_;
  cudnnTensorDescriptor_t b_desc_;
  cudnnOpTensorDescriptor_t op_desc_;
  bool is_null_input_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_GPU_NN_BIAS_ADD_GPU_KERNEL_H_
