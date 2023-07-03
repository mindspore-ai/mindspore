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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_SOFTMAX_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_SOFTMAX_GPU_KERNEL_H_

#include <vector>
#include <string>
#include <algorithm>
#include <map>
#include <utility>
#include <functional>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/kernel_constants.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/softmax_impl.cuh"

namespace mindspore {
namespace kernel {
class SoftmaxGpuKernelMod : public NativeGpuKernelMod {
 public:
  SoftmaxGpuKernelMod() = default;
  ~SoftmaxGpuKernelMod() = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    return kernel_func_(this, inputs, workspace, outputs, stream_ptr);
  }
  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

  std::vector<KernelAttr> GetOpSupport() override;

 protected:
  template <typename T>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<AddressPtr> &outputs, void *stream_ptr);

  using SoftmaxGpuLaunchFunc =
    std::function<bool(SoftmaxGpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                       const std::vector<kernel::AddressPtr> &, const std::vector<kernel::AddressPtr> &, void *)>;

  void ResetResource() {
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
    input_shape_.clear();

    // add new
    axis_acc_ = 0;
    outer_size_ = 1;
    inner_size_ = 1;
    shape_.clear();
    is_log_softmax_ = false;
  }

 protected:
  void InitSizeLists() {
    input_size_list_.push_back(input_size_);
    output_size_list_.push_back(output_size_);
    return;
  }

 private:
  // add new method
  int64_t maybe_wrap_dim(int64_t dim, int64_t dim_post_expr) {
    int64_t min = -dim_post_expr;
    int64_t max = dim_post_expr - 1;
    if (dim < min || dim > max) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the value of 'axis' must be in range [-" << dim_post_expr
                        << ", " << dim_post_expr << "), but got " << dim;
    }
    if (dim < 0) dim += dim_post_expr;

    return dim;
  }

  bool is_null_input_{false};
  size_t input_size_{0};
  size_t output_size_{0};
  size_t workspace_size_{0};

  std::vector<size_t> input_shape_{};
  size_t shape_size_{0};
  size_t batch_size_{0};
  size_t height_{0};
  size_t width_{0};
  size_t type_id_size_{0};

  // add new
  std::vector<size_t> shape_{};
  size_t axis_acc_{0};
  size_t outer_size_{1};
  size_t inner_size_{1};
  bool is_log_softmax_{false};

  SoftmaxGpuLaunchFunc kernel_func_;
  static std::vector<std::pair<KernelAttr, SoftmaxGpuLaunchFunc>> func_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_SOFTMAX_GPU_KERNEL_H_
