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

#include "kernel/cpu/cpu_kernel.h"
#include "include/runtime/hardware_abstract/kernel_base/ms_factory.h"

namespace mindspore {
namespace kernel {
namespace stft_cpu {
using complex64 = std::complex<float>;
using complex128 = std::complex<double>;
const complex128 kSTFTNegI{0, -1};
const complex128 kSTFTComplexZero{0, 0};
class STFTCpuKernelMod : public NativeCpuKernelMod, public MatchKernelHelper<STFTCpuKernelMod> {
 public:
  STFTCpuKernelMod() = default;
  explicit STFTCpuKernelMod(const std::string &kernel_type) : kernel_type_(kernel_type) {}
  ~STFTCpuKernelMod() override = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs) override {
    return kernel_func_(this, inputs, workspace, outputs);
  }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  const std::vector<std::pair<KernelAttr, KernelRunFunc>> &GetFuncList() const override;

 protected:
  std::vector<KernelAttr> GetOpSupport() override { return OpSupport(); }

 private:
  template <typename T, typename S, typename R, typename DataFT, typename DataFS, typename DataFR>
  bool LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<kernel::KernelTensor *> &outputs);

  std::string kernel_type_{"Unknown"};
  TypeId input_type_1_{kTypeUnknown};
  TypeId input_type_2_{kTypeUnknown};
  TypeId output_type_{kTypeUnknown};
  std::vector<int64_t> input_shape_1_{};
  std::vector<int64_t> input_shape_2_{};
  std::vector<int64_t> output_shape_{};
  bool normalized_{false};
  bool onesided_{false};
  bool return_complex_{false};
  bool has_batches_{false};

  int64_t n_fft_{0};
  int64_t fft_length_{0};  // relative to n_fft_ and onesided_, refer to w in formular
  int64_t hop_length_{0};
  int64_t win_length_{0};
  int64_t batches_{1};    // batch size when input is 2D (without vmap)
  int64_t input_len_{0};  // last dimension of input
  int64_t n_frames_{0};   // num of windows
  int64_t window_left_{0};
  bool pad_window_{false};
  int64_t w_skip_{0};
  complex128 norm_coe_{1.0, 0};
  size_t parallel_num_{0};  // batches_ * w

  // for vmap
  int64_t batch_rank_{0};
  int64_t vmap_batches_{1};
};
}  // namespace stft_cpu
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_STFT_CPU_KERNEL_H_
