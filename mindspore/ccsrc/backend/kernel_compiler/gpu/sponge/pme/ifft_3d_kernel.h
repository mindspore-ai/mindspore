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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPONGE_PME_IFFT_3D_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPONGE_PME_IFFT_3D_KERNEL_H_
#include <cuda_runtime_api.h>
#include <cufft.h>
#include <vector>
#include <string>
#include <map>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "runtime/device/gpu/cuda_common.h"
#include "backend/kernel_compiler/gpu/cuda_impl/sponge/pme/ifft_3d_impl.cuh"
namespace mindspore {
namespace kernel {
template <typename T>
using Complex = mindspore::utils::Complex<T>;
template <typename T>
class IFFT3DGpuKernel : public GpuKernel {
 public:
  IFFT3DGpuKernel() = default;
  ~IFFT3DGpuKernel() override = default;

  bool Init(const CNodePtr &kernel_node) override {
    fftx = static_cast<int>(GetAttr<int64_t>(kernel_node, "fftx"));
    ffty = static_cast<int>(GetAttr<int64_t>(kernel_node, "ffty"));
    fftz = static_cast<int>(GetAttr<int64_t>(kernel_node, "fftz"));
    Nfft = fftx * ffty * fftz;
    Nall = fftx * ffty * (fftz - 1) * 2;

    cufftPlan3d(&FFT_plan_c2r, fftx, ffty, (fftz - 1) * 2, CUFFT_C2R);

    InitSizeLists();
    return true;
  }

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    auto input_tensor = GetDeviceAddress<Complex<T>>(inputs, 0);
    auto output_tensor = GetDeviceAddress<T>(outputs, 0);

    cufftSetStream(FFT_plan_c2r, reinterpret_cast<cudaStream_t>(stream_ptr));
    IFFT3D<T>(Nfft, input_tensor, output_tensor, FFT_plan_c2r, reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(Nfft * sizeof(Complex<T>));
    output_size_list_.push_back(Nall * sizeof(T));
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
  cufftHandle FFT_plan_c2r;
};
}  // namespace kernel
}  // namespace mindspore
#endif
