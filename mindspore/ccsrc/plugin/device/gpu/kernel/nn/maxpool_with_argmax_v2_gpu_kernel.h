/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_MAXPOOL_WITH_ARGMAX_V2_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_MAXPOOL_WITH_ARGMAX_V2_GPU_KERNEL_H_

#include <vector>
#include <string>
#include <map>
#include <utility>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/kernel_constants.h"

namespace mindspore {
namespace kernel {
class MaxPoolWithArgmaxV2FwdGpuKernelMod : public NativeGpuKernelMod {
 public:
  explicit MaxPoolWithArgmaxV2FwdGpuKernelMod(const std::string &kernel_name) : kernel_name_(kernel_name) {}
  ~MaxPoolWithArgmaxV2FwdGpuKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(
    const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
    const std::vector<KernelTensorPtr> &outputs,
    const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost = std::map<uint32_t, tensor::TensorPtr>()) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *cuda_stream) override {
    if (is_null_input_) {
      return true;
    }
    cuda_stream_ = cuda_stream;
    return kernel_func_(this, inputs, outputs);
  }

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T, typename S>
  bool LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &outputs);
  using MaxPoolWithArgmaxV2Func =
    std::function<bool(MaxPoolWithArgmaxV2FwdGpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                       const std::vector<kernel::AddressPtr> &)>;
  static std::vector<std::pair<KernelAttr, MaxPoolWithArgmaxV2Func>> func_list_;
  MaxPoolWithArgmaxV2Func kernel_func_;
  std::string kernel_name_;
  std::vector<int64_t> ksize_;
  std::vector<int64_t> pads_;
  std::vector<int64_t> strides_;
  std::vector<int64_t> dilation_;
  std::vector<int64_t> input_shape_;
  std::vector<int64_t> output_shape_;
  bool is_null_input_{false};
  void *cuda_stream_{nullptr};

  int ksize_h_{0};
  int ksize_w_{0};
  int strides_h_{0};
  int strides_w_{0};
  int pads_h_{0};
  int pads_w_{0};
  int dilation_h_{0};
  int dilation_w_{0};
  int in_n_{0};
  int in_c_{0};
  int in_h_{0};
  int in_w_{0};
  int out_h_{0};
  int out_w_{0};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_MAXPOOL_WITH_ARGMAX_V2_GPU_KERNEL_H_
