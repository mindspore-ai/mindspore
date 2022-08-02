/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPONGE_PME_PME_IRFFT_2D_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPONGE_PME_PME_IRFFT_2D_KERNEL_H_
#include <cuda_runtime_api.h>
#include <cufft.h>
#include <vector>
#include <string>
#include <map>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_common.h"
#include "plugin/device/gpu/kernel/cuda_impl/sponge/pme/pme_irfft_2d_impl.cuh"
namespace mindspore {
namespace kernel {
template <typename T>
using Complex = mindspore::utils::Complex<T>;
template <typename T>
class PMEIRFFT2DGpuKernelMod : public DeprecatedNativeGpuKernelMod {
 public:
  PMEIRFFT2DGpuKernelMod() = default;
  ~PMEIRFFT2DGpuKernelMod() override = default;

  bool Init(const CNodePtr &kernel_node) override {
    constexpr int NUM_2 = 2;
    fftx = static_cast<int>(GetAttr<int64_t>(kernel_node, "fftx"));
    ffty = static_cast<int>(GetAttr<int64_t>(kernel_node, "ffty"));
    Nfft = fftx * ffty;
    Nall = fftx * (ffty - 1) * NUM_2;

    cufftPlan2d(&FFT_plan_c2r, fftx, (ffty - 1) * NUM_2, CUFFT_C2R);

    InitSizeLists();
    return true;
  }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    auto input_tensor = GetDeviceAddress<Complex<T>>(inputs, 0);
    auto output_tensor = GetDeviceAddress<T>(outputs, 0);

    cufftSetStream(FFT_plan_c2r, reinterpret_cast<cudaStream_t>(stream_ptr));
    PMEIRFFT2D<T>(Nfft, input_tensor, output_tensor, FFT_plan_c2r, reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(Nfft * sizeof(Complex<T>));
    output_size_list_.push_back(Nall * sizeof(T));
  }

 private:
  int fftx;
  int ffty;
  int Nall;
  int Nfft;
  cufftHandle FFT_plan_c2r;
};
}  // namespace kernel
}  // namespace mindspore
#endif
