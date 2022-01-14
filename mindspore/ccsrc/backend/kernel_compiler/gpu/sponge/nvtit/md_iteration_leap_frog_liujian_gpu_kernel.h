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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MD_ITERATION_LEAP_FROG_LIUJIAN_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MD_ITERATION_LEAP_FROG_LIUJIAN_GPU_KERNEL_H_

#include "backend/kernel_compiler/gpu/cuda_impl/sponge/nvtit/md_iteration_leap_frog_liujian_gpu_impl.cuh"
#include <cuda_runtime_api.h>
#include <map>
#include <string>
#include <vector>

#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "runtime/device/gpu/cuda_common.h"

namespace mindspore {
namespace kernel {
constexpr size_t kIdx2 = 2;
constexpr size_t kIdx3 = 3;
constexpr size_t kIdx4 = 4;
constexpr size_t kIdx5 = 5;
constexpr size_t kIdx6 = 6;
constexpr size_t kIdx7 = 7;

template <typename T, typename T1>
class MDIterationLeapFrogLiujianCudaGpuKernelMod : public NativeGpuKernelMod {
 public:
  MDIterationLeapFrogLiujianCudaGpuKernelMod() {}
  ~MDIterationLeapFrogLiujianCudaGpuKernelMod() override = default;

  bool Init(const CNodePtr &kernel_node) override {
    // get bond_numbers
    kernel_node_ = kernel_node;
    atom_numbers = static_cast<int>(GetAttr<int64_t>(kernel_node, "atom_numbers"));
    half_dt = static_cast<float>(GetAttr<float>(kernel_node, "half_dt"));
    dt = static_cast<float>(GetAttr<float>(kernel_node, "dt"));
    exp_gamma = static_cast<float>(GetAttr<float>(kernel_node, "exp_gamma"));
    const double kCoef1 = 3.;
    const double kCoef2 = 4.;
    float4_numbers = ceil(kCoef1 * static_cast<double>(atom_numbers) / kCoef2);
    InitSizeLists();
    return true;
  }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    auto inverse_mass = GetDeviceAddress<float>(inputs, 0);
    auto sqrt_mass_inverse = GetDeviceAddress<float>(inputs, 1);
    auto vel = GetDeviceAddress<float>(inputs, kIdx2);
    auto crd = GetDeviceAddress<float>(inputs, kIdx3);
    auto frc = GetDeviceAddress<float>(inputs, kIdx4);
    auto acc = GetDeviceAddress<float>(inputs, kIdx5);
    auto rand_state = GetDeviceAddress<float>(inputs, kIdx6);
    auto rand_frc = GetDeviceAddress<float>(inputs, kIdx7);

    auto output = GetDeviceAddress<float>(outputs, 0);

    MD_Iteration_Leap_Frog_With_LiuJian(atom_numbers, half_dt, dt, exp_gamma, float4_numbers, inverse_mass,
                                        sqrt_mass_inverse, vel, crd, frc, acc,
                                        reinterpret_cast<curandStatePhilox4_32_10_t *>(rand_state), rand_frc, output,
                                        reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

 protected:
  void InitSizeLists() override {
    const int kAtomSize = 3;

    input_size_list_.push_back(atom_numbers * sizeof(float));
    input_size_list_.push_back(atom_numbers * sizeof(float));
    input_size_list_.push_back(atom_numbers * kAtomSize * sizeof(float));
    input_size_list_.push_back(atom_numbers * kAtomSize * sizeof(float));
    input_size_list_.push_back(atom_numbers * kAtomSize * sizeof(float));
    input_size_list_.push_back(atom_numbers * kAtomSize * sizeof(float));
    input_size_list_.push_back(float4_numbers * sizeof(curandStatePhilox4_32_10_t));
    input_size_list_.push_back(atom_numbers * kAtomSize * sizeof(float));

    output_size_list_.push_back(atom_numbers * kAtomSize * sizeof(T));
  }

 private:
  int atom_numbers;
  float half_dt;
  float dt;
  float exp_gamma;
  int float4_numbers;
};
}  // namespace kernel
}  // namespace mindspore
#endif
