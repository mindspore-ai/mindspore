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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_MAXPOOLWITHARGMAX_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_MAXPOOLWITHARGMAX_GPU_KERNEL_H_

#include <algorithm>
#include <vector>
#include <string>
#include <map>
#include <utility>
#include "mindspore/core/utils/ms_context.h"
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/maxpool_with_argmax_impl.cuh"
#include "plugin/device/gpu/kernel/kernel_constants.h"

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

class MaxPoolWithArgmaxGpuKernelMod : public NativeGpuKernelMod,
                                      public MatchKernelHelper<MaxPoolWithArgmaxGpuKernelMod> {
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

  std::vector<KernelAttr> GetOpSupport() override { return OpSupport(); }

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs,
             const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    stream_ptr_ = stream_ptr;
    return kernel_func_(this, inputs, workspace, outputs);
  }

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;
  const std::vector<std::pair<KernelAttr, KernelRunFunc>> &GetFuncList() const override;

 protected:
  template <typename T, typename S>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<AddressPtr> &outputs) {
    T *input_addr = GetDeviceAddress<T>(inputs, 0);
    T *output_addr = GetDeviceAddress<T>(outputs, 0);
    S *index_addr = GetDeviceAddress<S>(outputs, 1);
    CalMaxPoolWithArgmax(input_addr, n_, c_, input_height_, input_width_, window_height_, window_width_, stride_height_,
                         stride_width_, pad_top_, pad_left_, output_height_, output_width_, output_addr, index_addr,
                         device_id_, reinterpret_cast<cudaStream_t>(stream_ptr_));
    return true;
  }

 private:
  void SetPad() {
    MS_EXCEPTION_IF_ZERO("stride height", stride_height_);
    MS_EXCEPTION_IF_ZERO("stride width", stride_width_);

    int tmp_height = (input_height_ / stride_height_) * stride_height_ == input_height_
                       ? (input_height_ / stride_height_)
                       : (input_height_ / stride_height_) + 1;
    pad_height_ = std::max<int>(0, (tmp_height - 1) * stride_height_ + window_height_ - input_height_);

    int tmp_width = (input_width_ / stride_width_) * stride_width_ == input_width_ ? (input_width_ / stride_width_)
                                                                                   : (input_width_ / stride_width_) + 1;
    pad_width_ = std::max<int>(0, (tmp_width - 1) * stride_width_ + window_width_ - input_width_);
    pad_top_ = pad_height_ / 2;
    pad_left_ = pad_width_ / 2;
  }

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

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_MAXPOOLWITHARGMAX_GPU_KERNEL_H_
