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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_BIAS_ADD_GRAD_GPU_KENEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_BIAS_ADD_GRAD_GPU_KENEL_H_

#include <vector>
#include <map>
#include <utility>
#include <string>
#include <algorithm>
#include <memory>
#include "include/runtime/hardware_abstract/kernel_base/common_utils.h"
#include "kernel/gpu/gpu_kernel.h"

namespace mindspore {
namespace kernel {
class BiasAddGradGpuKernelMod : public NativeGpuKernelMod, public MatchKernelHelper<BiasAddGradGpuKernelMod> {
 public:
  BiasAddGradGpuKernelMod() { ResetResource(); }
  ~BiasAddGradGpuKernelMod() override { DestroyResource(); }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    stream_ = stream_ptr;
    return kernel_func_(this, inputs, workspace, outputs);
  }

  const std::vector<std::pair<KernelAttr, KernelRunFunc>> &GetFuncList() const override;

  std::vector<KernelAttr> GetOpSupport() override { return OpSupport(); }

  void DestroyResource() noexcept override;

  void ResetResource() noexcept;

 private:
  template <typename T>
  bool LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                    const std::vector<kernel::KernelTensor *> &workspace,
                    const std::vector<kernel::KernelTensor *> &outputs);
  void MethodSelection();
  void InitResource() override;
  void SetResource();
  void InitSizeLists();

  size_t unit_size_{1};
  bool same_dims_{true};
  bool is_null_input_;
  bool use_cudnn_{false};
  size_t dy_num_{1};  // for own implementation
  size_t db_num_{1};
  size_t num_dims_{0};
  size_t bias_size_{0};   // for own implementation
  ShapeVector dy_shape_;  // for own implementation
  ShapeVector db_shape_;  // for own implementation
  int64_t data_format_{Format::NHWC};
  // for cudnn implementation
  void *stream_{nullptr};
  cudnnHandle_t cudnn_handle_{nullptr};
  cudnnDataType_t cudnn_data_type_{CUDNN_DATA_FLOAT};
  cudnnTensorFormat_t cudnn_compute_format_{CUDNN_TENSOR_NCHW};
  cudnnTensorDescriptor_t dy_desc_{nullptr};
  cudnnTensorDescriptor_t db_desc_{nullptr};
  cudnnReduceTensorDescriptor_t op_desc_{nullptr};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_BIAS_ADD_GRAD_GPU_KENEL_H_
