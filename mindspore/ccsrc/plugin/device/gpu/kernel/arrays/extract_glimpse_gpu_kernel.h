/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_ARRAYS_EXTRACT_GLIMPSE_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_ARRAYS_EXTRACT_GLIMPSE_GPU_KERNEL_H_
#include <vector>
#include <string>
#include <numeric>
#include <memory>
#include <utility>
#include <algorithm>
#include <functional>
#include <map>
#include "mindspore/core/ops/extract_glimpse.h"
#include "abstract/utils.h"
#include "plugin/factory/ms_factory.h"
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/unique_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/extract_glimpse_impl.cuh"

namespace mindspore {
namespace kernel {
class ExtractGlimpseGpuKernelMod : public NativeGpuKernelMod {
 public:
  ExtractGlimpseGpuKernelMod() {}
  ~ExtractGlimpseGpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    MS_EXCEPTION_IF_NULL(kernel_func_);
    return kernel_func_(this, inputs, workspace, outputs, stream_ptr);
  }

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

 protected:
  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &others);
  template <typename T>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<AddressPtr> &outputs, void *stream_ptr);
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  using ExtractGlimpseFunc =
    std::function<bool(ExtractGlimpseGpuKernelMod *, const std::vector<AddressPtr> &, const std::vector<AddressPtr> &,
                       const std::vector<AddressPtr> &, void *)>;
  static std::vector<std::pair<KernelAttr, ExtractGlimpseFunc>> func_list_;
  ExtractGlimpseFunc kernel_func_;
  void *stream_ptr_;
  bool is_null_input_{false};
  size_t inputs_size_{0};
  size_t size_size_{0};
  size_t offsets_size_{0};
  size_t output_size_{0};
  size_t inputs_elements_{0};
  size_t size_elements_{0};
  size_t offsets_elements_{0};
  size_t output_elements_{0};
  size_t batch_cnt_{0};
  size_t channels_{0};
  size_t image_height_{0};
  size_t image_width_{0};
  bool centered_{true};
  bool normalized_{true};
  bool uniform_noise_{true};
  ExtractGlimpsenoiseMode noise_{ExtractGlimpsenoiseMode::UNIFORM};
  std::vector<int64_t> inputs_shape;
  std::vector<int64_t> size_shape;
  std::vector<int64_t> offsets_shape;
  std::vector<int64_t> output_shape;
  // std::vector<KernelTensorPtr> outputs_ = {};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_ARRAYS_EXTRACT_GLIMPSE_GPU_KERNEL_H_
