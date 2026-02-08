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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_MAXPOOL_WITH_ARGMAX_GRAD_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_MAXPOOL_WITH_ARGMAX_GRAD_GPU_KERNEL_H_

#include <algorithm>
#include <vector>
#include <string>
#include <utility>
#include <map>
#include "utils/ms_context.h"
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/cuda_impl/cuda_ops/maxpool_with_argmax_grad_impl.cuh"
#include "kernel/gpu/kernel_constants.h"
#include "mindspore/ops/infer/grad/max_pool_grad_with_argmax.h"
#include "include/runtime/hardware_abstract/kernel_base/ms_factory.h"
#include "include/runtime/hardware_abstract/kernel_base/common_utils.h"

namespace mindspore {
namespace kernel {
class MaxPoolGradWithArgmaxGpuKernelMod : public NativeGpuKernelMod {
 public:
  MaxPoolGradWithArgmaxGpuKernelMod()
      : n_(0),
        c_(0),
        x_height_(0),
        x_width_(0),
        dy_height_(0),
        dy_width_(0),
        is_null_input_(false),
        x_size_(0),
        dy_size_(0),
        index_size_(0),
        dx_size_(0) {}
  ~MaxPoolGradWithArgmaxGpuKernelMod() override = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

 protected:
  std::vector<KernelAttr> GetOpSupport() override;
  using MaxPoolGradWithArgmaxFunc = std::function<bool(
    MaxPoolGradWithArgmaxGpuKernelMod *, const std::vector<kernel::KernelTensor *> &,
    const std::vector<kernel::KernelTensor *> &, const std::vector<kernel::KernelTensor *> &, void *)>;

 private:
  template <typename T, typename S>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<KernelTensor *> &outputs, void *stream_ptr);
  MaxPoolGradWithArgmaxFunc kernel_func_;
  static std::vector<std::pair<KernelAttr, MaxPoolGradWithArgmaxFunc>> func_list_;
  std::string kernel_name_{};
  int n_;
  int c_;
  int x_height_;
  int x_width_;
  int dy_height_;
  int dy_width_;
  bool is_null_input_;

  size_t x_size_;
  size_t dy_size_;
  size_t index_size_;
  size_t dx_size_;
  size_t x_type_size_{1};
  size_t dy_type_size_{1};
  size_t idx_type_size_{1};
  size_t dx_type_size_{1};
  void *stream_ptr_{nullptr};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_MAXPOOL_WITH_ARGMAX_GRAD_GPU_KERNEL_H_
