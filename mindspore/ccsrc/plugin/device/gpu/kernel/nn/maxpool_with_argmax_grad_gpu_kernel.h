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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_MAXPOOLWITHARGMAX_GRAD_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_MAXPOOLWITHARGMAX_GRAD_GPU_KERNEL_H_

#include <algorithm>
#include <vector>
#include <string>
#include <map>
#include <utility>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/maxpool_with_argmax_grad_impl.cuh"
#include "plugin/device/gpu/kernel/kernel_constants.h"

namespace mindspore {
namespace kernel {
constexpr size_t kXDimLowerLimit = 4;
constexpr size_t kDyDimLowerLimit = 4;
constexpr size_t kXIndexForN = 0;
constexpr size_t kXIndexForC = 1;
constexpr size_t kXIndexForH = 2;
constexpr size_t kXIndexForW = 3;
constexpr size_t kDyIndexForH = 2;
constexpr size_t kDyIndexForW = 3;

class MaxPoolWithArgmaxGradGpuKernelMod : public NativeGpuKernelMod,
                                          public MatchKernelHelper<MaxPoolWithArgmaxGradGpuKernelMod> {
 public:
  MaxPoolWithArgmaxGradGpuKernelMod()
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
  ~MaxPoolWithArgmaxGradGpuKernelMod() override = default;
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
    T *dy_addr = GetDeviceAddress<T>(inputs, 1);
    S *index_addr = GetDeviceAddress<S>(inputs, 2);
    T *dx_addr = GetDeviceAddress<T>(outputs, 0);
    CalMaxPoolWithArgmaxGrad(dy_addr, index_addr, n_, c_, x_height_, x_width_, dy_height_, dy_width_, dx_addr,
                             reinterpret_cast<cudaStream_t>(stream_ptr_));
    return true;
  }

 private:
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

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_MAXPOOLWITHARGMAX_GRAD_GPU_KERNEL_H_
