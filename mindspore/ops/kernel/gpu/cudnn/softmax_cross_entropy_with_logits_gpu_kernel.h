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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICES_GPU_KERNEL_NN_SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICES_GPU_KERNEL_NN_SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_GPU_KERNEL_H_

#include <stdint.h>
#include <vector>
#include <string>
#include <map>
#include <utility>
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/cuda_impl/cuda_ops/cross_entropy_impl.cuh"
#include "kernel/gpu/kernel_constants.h"

namespace mindspore {
namespace kernel {
class SoftmaxCrossEntropyWithLogitsGpuKernelMod : public NativeGpuKernelMod {
 public:
  SoftmaxCrossEntropyWithLogitsGpuKernelMod() = default;
  ~SoftmaxCrossEntropyWithLogitsGpuKernelMod() override { DestroyResource(); }

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    return kernel_func_(this, inputs, workspace, outputs, stream_ptr);
  }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

 protected:
  std::vector<KernelAttr> GetOpSupport() override;
  template <typename T, typename S>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<KernelTensor *> &outputs, void *stream_ptr);

  using SoftmaxCrossEntropyWithLogitsGpuLaunchFunc = std::function<bool(
    SoftmaxCrossEntropyWithLogitsGpuKernelMod *, const std::vector<kernel::KernelTensor *> &,
    const std::vector<kernel::KernelTensor *> &, const std::vector<kernel::KernelTensor *> &, void *)>;

  void DestroyResource() noexcept override {
    CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnDestroyTensorDescriptor(softmax_output_descriptor_),
                                       kernel_name_ + " cudnnDestroyTensorDescriptor failed.");
    CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnDestroyTensorDescriptor(logits_descriptor_),
                                       kernel_name_ + " cudnnDestroyTensorDescriptor failed.");
  }

 private:
  int CheckShapeValidation(const ShapeVector &logits_shape, const ShapeVector &labels_shape) {
    size_t logits_dim_length = logits_shape.size();
    size_t labels_dim_length = labels_shape.size();
    if (logits_dim_length == 0) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of logits cannot be equal to 0, but got "
                    << logits_dim_length;
      return KRET_RESIZE_FAILED;
    }

    if (labels_dim_length != logits_dim_length) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of logits and labels must be the same, but "
                    << "got the dimension of labels: " << labels_dim_length
                    << ", the dimension of logits: " << logits_dim_length;
      return KRET_RESIZE_FAILED;
    }
    if (!std::equal(labels_shape.begin(), labels_shape.end(), logits_shape.begin())) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', the shape of logits and labels must be the same except "
                    << "the last dimension, but got the shape of logits: " << logits_shape
                    << ", the shape of labels: " << labels_shape;
      return KRET_RESIZE_FAILED;
    }
    return KRET_OK;
  }

  SoftmaxCrossEntropyWithLogitsGpuLaunchFunc kernel_func_;
  static std::vector<std::pair<KernelAttr, SoftmaxCrossEntropyWithLogitsGpuLaunchFunc>> func_list_;
  cudnnHandle_t cudnn_handle_{nullptr};
  cudnnTensorDescriptor_t logits_descriptor_{nullptr};
  cudnnTensorDescriptor_t softmax_output_descriptor_{nullptr};
  cudnnSoftmaxAlgorithm_t algo_{CUDNN_SOFTMAX_ACCURATE};
  cudnnSoftmaxMode_t mode_{CUDNN_SOFTMAX_MODE_INSTANCE};
  cudnnDataType_t cudnn_data_type_{CUDNN_DATA_FLOAT};
  size_t logits_size_{0};
  size_t labels_size_{0};
  size_t output1_size_{0};
  size_t output2_size_{0};
  size_t softmax_output_logits_size_{0};
  size_t batch_size_{0};
  size_t channel_size_{0};
  size_t height_{0};
  size_t width_{0};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICES_GPU_KERNEL_NN_SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_GPU_KERNEL_H_
