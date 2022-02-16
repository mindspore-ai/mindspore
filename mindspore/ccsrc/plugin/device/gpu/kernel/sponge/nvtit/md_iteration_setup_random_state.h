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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MD_ITERATION_SETUP_RANDOM_STATE_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MD_ITERATION_SETUP_RANDOM_STATE_GPU_KERNEL_H_

#include "plugin/device/gpu/kernel/cuda_impl/sponge/nvtit/md_iteration_setup_random_state_gpu_impl.cuh"
#include <cuda_runtime_api.h>
#include <map>
#include <string>
#include <vector>

#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_common.h"

namespace mindspore {
namespace kernel {
template <typename T, typename T1>
class MDIterationSetupRandStateGpuKernelMod : public NativeGpuKernelMod {
 public:
  MDIterationSetupRandStateGpuKernelMod() {}
  ~MDIterationSetupRandStateGpuKernelMod() override = default;

  bool Init(const CNodePtr &kernel_node) override {
    // get bond_numbers
    kernel_node_ = kernel_node;
    atom_numbers = static_cast<int>(GetAttr<int64_t>(kernel_node, "atom_numbers"));
    seed = static_cast<int>(GetAttr<int64_t>(kernel_node, "seed"));
    float4_numbers = ceil(3. * static_cast<double>(atom_numbers) / 4.);
    InitSizeLists();
    return true;
  }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    auto output = GetDeviceAddress<float>(outputs, 0);
    curandStatePhilox4_32_10_t *rand_state = reinterpret_cast<curandStatePhilox4_32_10_t *>(output);
    MD_Iteration_Setup_Random_State(float4_numbers, rand_state, seed, reinterpret_cast<cudaStream_t>(stream_ptr));

    return true;
  }

 protected:
  void InitSizeLists() override { output_size_list_.push_back(sizeof(curandStatePhilox4_32_10_t) * float4_numbers); }

 private:
  int atom_numbers;
  int seed;
  int float4_numbers;
};
}  // namespace kernel
}  // namespace mindspore
#endif
