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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_MATH_FFT_WITH_SIZE_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_MATH_FFT_WITH_SIZE_GPU_KERNEL_H_

#include <cufft.h>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <map>
#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include <functional>
#include "include/op_enum.h"
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/cuda_impl/cuda_ops/fft_with_size_impl.cuh"
#include "utils/misc.h"

namespace mindspore {
namespace kernel {
enum class FFTVariety { unknown = 0, fft = 1, ifft = 2, rfft = 3, irfft = 4 };

class FFTWithSizeGpuKernelMod : public NativeGpuKernelMod {
 public:
  FFTWithSizeGpuKernelMod() = default;
  ~FFTWithSizeGpuKernelMod() override { ResetResource(); };

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override {
    if (!FFTVarietyInResize(inputs, outputs)) {
      return RET_FAILED;
    }
    return resize_func_(this, inputs, outputs);
  }

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    return launch_func_(this, inputs, workspace, outputs, stream_ptr);
  }

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  bool FFTVarietyInResize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs);
  void ResetResource() noexcept;
  bool MakeCufftPlan(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs);
  int ResizeBase(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs);
  int ResizeFFT(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs);
  int ResizeIFFT(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs);
  int ResizeRFFT(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs);
  int ResizeIRFFT(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs);
  using FFTWithSizeResizeFunc = std::function<int(FFTWithSizeGpuKernelMod *, const std::vector<KernelTensor *> &,
                                                  const std::vector<KernelTensor *> &)>;
  FFTWithSizeResizeFunc resize_func_{};

  template <typename T>
  bool LaunchFFT(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                 const std::vector<KernelTensor *> &outputs, void *stream_ptr);
  template <typename T>
  bool LaunchIFFT(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                  const std::vector<KernelTensor *> &outputs, void *stream_ptr);
  template <typename S, typename T>
  bool LaunchRFFT(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                  const std::vector<KernelTensor *> &outputs, void *stream_ptr);
  template <typename S, typename T>
  bool LaunchIRFFT(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                   const std::vector<KernelTensor *> &outputs, void *stream_ptr);
  using FFTWithSizeLaunchFunc =
    std::function<bool(FFTWithSizeGpuKernelMod *, const std::vector<KernelTensor *> &,
                       const std::vector<KernelTensor *> &, const std::vector<KernelTensor *> &, void *)>;
  FFTWithSizeLaunchFunc launch_func_{};

 private:
  int64_t rank_{1};
  bool is_inverse_{false};  // 0: forward, 1: inverse
  bool is_real_{false};
  ops::NormMode norm_type_{ops::NormMode::BACKWARD};  // forward, backward, ortho
  // is_onesided controls whether frequency is halved when signal is real, which means is_real_ is true.
  // The default value is true. cufft does not support full freq with real signal. We use cast as a temporary solution.
  bool is_onesided_{true};

  cufftHandle cufft_plan_{0};
  cufftType cufft_type_{CUFFT_C2C};
  cublasHandle_t scale_plan_{nullptr};
  double scale_factor_{1.0};

  FFTVariety fft_variety_{FFTVariety::unknown};
  size_t data_type_bytes_{1};
  std::vector<int64_t> x_shape_;
  std::vector<int64_t> y_shape_;
  int x_elements_{0};
  int y_elements_{0};
  int fit_index_{0};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_MATH_FFT_WITH_SIZE_GPU_KERNEL_H_
