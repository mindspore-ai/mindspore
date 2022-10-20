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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_STFT_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_STFT_CPU_KERNEL_H_

#include <functional>
#include <memory>
#include <vector>
#include <iostream>
#include <string>
#include <complex>
#include <map>
#include <utility>
#include <algorithm>
#include <unordered_map>

#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
using complex64 = std::complex<float>;
using complex128 = std::complex<double>;
const complex128 kSTFTNegI{0, -1};
const complex128 kSTFTComplexZero{0, 0};
class STFTCpuKernelMod : public NativeCpuKernelMod, public MatchKernelHelper<STFTCpuKernelMod> {
 public:
  STFTCpuKernelMod() = default;
  explicit STFTCpuKernelMod(const std::string &kernel_type) : kernel_type_(kernel_type) {}
  ~STFTCpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override {
    return kernel_func_(this, inputs, workspace, outputs);
  }

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

  const std::vector<std::pair<KernelAttr, KernelRunFunc>> &GetFuncList() const override;

 protected:
  std::vector<KernelAttr> GetOpSupport() override { return OpSupport(); }

 private:
  template <typename T, typename S, typename R, typename DataFT, typename DataFS, typename DataFR>
  bool LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<kernel::AddressPtr> &outputs);

  std::string kernel_type_{"Unknown"};
  TypeId input_type_1_{kTypeUnknown};
  TypeId input_type_2_{kTypeUnknown};
  TypeId output_type_{kTypeUnknown};
  std::vector<int64_t> input_shape_1_;
  std::vector<int64_t> input_shape_2_;
  std::vector<int64_t> output_shape_;
  bool normalized_;
  bool onesided_;
  bool return_complex_;
  bool has_batches_{false};

  int64_t n_fft_;
  int64_t fft_length_;  // relative to n_fft_ and onesided_, refer to w in formular
  int64_t hop_length_;
  int64_t win_length_;
  int64_t batches_{1};  // batch size when input is 2D (without vmap)
  int64_t input_len_;   // last dimension of input
  int64_t n_frames_;    // num of windows
  int64_t window_left_;
  bool pad_window_{false};
  int64_t w_skip_;
  complex128 norm_coe_{1.0, 0};
  size_t parallel_num_;  // batches_ * w

  complex128 temp_ = kSTFTComplexZero;
  complex128 complex_w_ = kSTFTComplexZero;
  complex128 complex_input_ = kSTFTComplexZero;

  // for vmap
  int64_t batch_rank_{0};
  int64_t vmap_batches_{1};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_STFT_CPU_KERNEL_H_
