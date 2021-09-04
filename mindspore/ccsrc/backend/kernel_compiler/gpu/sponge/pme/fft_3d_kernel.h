/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPONGE_PME_FFT_3D_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPONGE_PME_FFT_3D_KERNEL_H_
#include <cuda_runtime_api.h>
#include <cufft.h>
#include <vector>
#include <string>
#include <map>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "runtime/device/gpu/cuda_common.h"
#include "backend/kernel_compiler/gpu/cuda_impl/sponge/pme/fft_3d_impl.cuh"
namespace mindspore {
namespace kernel {
template <typename T>
using Complex = mindspore::utils::Complex<T>;
template <typename T>
class FFT3DGpuKernel : public GpuKernel {
 public:
  FFT3DGpuKernel() = default;
  ~FFT3DGpuKernel() override = default;

  bool Init(const CNodePtr &kernel_node) override {
    fftx = static_cast<int>(GetAttr<int64_t>(kernel_node, "fftx"));
    ffty = static_cast<int>(GetAttr<int64_t>(kernel_node, "ffty"));
    fftz = static_cast<int>(GetAttr<int64_t>(kernel_node, "fftz"));
    Nall = fftx * ffty * fftz;
    Nfft = fftx * ffty * (fftz / 2 + 1);

    cufftPlan3d(&FFT_plan_r2c, fftx, ffty, fftz, CUFFT_R2C);

    InitSizeLists();
    return true;
  }

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    auto input_tensor = GetDeviceAddress<T>(inputs, 0);
    auto output_tensor = GetDeviceAddress<Complex<T>>(outputs, 0);

    cufftSetStream(FFT_plan_r2c, reinterpret_cast<cudaStream_t>(stream_ptr));
    FFT3D<T>(Nfft, input_tensor, output_tensor, FFT_plan_r2c, reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(Nall * sizeof(T));
    output_size_list_.push_back(Nfft * sizeof(Complex<T>));
  }

 private:
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
  int fftx;
  int ffty;
  int fftz;
  int Nall;
  int Nfft;
  cufftHandle FFT_plan_r2c;
};
}  // namespace kernel
}  // namespace mindspore
#endif
