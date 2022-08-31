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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_MAXPOOL_WITH_ARGMAX_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_MAXPOOL_WITH_ARGMAX_GPU_KERNEL_H_

#include <algorithm>
#include <vector>
#include <string>
#include <utility>
#include <map>
#include "mindspore/core/utils/ms_context.h"
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/maxpool_with_argmax_impl.cuh"
#include "plugin/device/gpu/kernel/kernel_constants.h"
#include "mindspore/core/ops/max_pool_with_argmax.h"
#include "plugin/factory/ms_factory.h"
#include "mindspore/ccsrc/kernel/common_utils.h"

namespace mindspore {
namespace kernel {
constexpr size_t kInputDimLowerLimit = 4;
constexpr size_t kOutputDimLowerLimit = 4;
constexpr size_t kInputIndexForN = 0;
constexpr size_t kInputIndexForC = 1;
constexpr size_t kInputIndexForH = 2;
constexpr size_t kInputIndexForW = 3;
constexpr size_t kOutputIndexForH = 2;
constexpr size_t kOutputIndexForW = 3;

class MaxPoolWithArgmaxGpuKernelMod : public NativeGpuKernelMod {
 public:
  MaxPoolWithArgmaxGpuKernelMod()
      : n_(0),
        c_(0),
        input_height_(0),
        input_width_(0),
        window_height_(0),
        window_width_(0),
        pad_height_(0),
        pad_width_(0),
        pad_top_(0),
        pad_left_(0),
        stride_height_(0),
        stride_width_(0),
        output_height_(0),
        output_width_(0),
        is_null_input_(false),
        input_size_(0),
        output_size_(0) {}
  ~MaxPoolWithArgmaxGpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs,
             const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) override;

 protected:
  std::vector<KernelAttr> GetOpSupport() override;
  using MaxPoolWithArgmaxFunc =
    std::function<bool(MaxPoolWithArgmaxGpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                       const std::vector<kernel::AddressPtr> &, const std::vector<kernel::AddressPtr> &, void *)>;

 private:
  template <typename T, typename S>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<AddressPtr> &outputs, void *stream_ptr);
  void SetPad();
  MaxPoolWithArgmaxFunc kernel_func_;
  static std::vector<std::pair<KernelAttr, MaxPoolWithArgmaxFunc>> func_list_;

  std::string pad_mode_;
  int n_;
  int c_;
  int input_height_;
  int input_width_;
  int window_height_;
  int window_width_;
  int pad_height_;
  int pad_width_;
  int pad_top_;
  int pad_left_;
  int stride_height_;
  int stride_width_;
  int output_height_;
  int output_width_;
  bool is_null_input_;

  size_t input_size_;
  size_t output_size_;
  void *stream_ptr_{nullptr};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_MAXPOOL_WITH_ARGMAX_GPU_KERNEL_H_
