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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MD_ITERATION_LEAP_FROG_WITH_MAX_VEL_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MD_ITERATION_LEAP_FROG_WITH_MAX_VEL_KERNEL_H_

#include "backend/kernel_compiler/gpu/cuda_impl/sponge/nvtit/md_iteration_leap_frog_with_max_vel_impl.cuh"
#include <cuda_runtime_api.h>
#include <map>
#include <string>
#include <vector>

#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "runtime/device/gpu/cuda_common.h"

namespace mindspore {
namespace kernel {
template <typename T>
class MDIterationLeapFrogWithMaxVelGpuKernel : public GpuKernel {
 public:
  MDIterationLeapFrogWithMaxVelGpuKernel() {}
  ~MDIterationLeapFrogWithMaxVelGpuKernel() override = default;

  bool Init(const CNodePtr &kernel_node) override {
    // get bond_numbers
    kernel_node_ = kernel_node;
    atom_numbers = static_cast<int>(GetAttr<int64_t>(kernel_node, "atom_numbers"));
    dt = static_cast<float>(GetAttr<float>(kernel_node, "dt"));
    max_velocity = static_cast<float>(GetAttr<float>(kernel_node, "max_velocity"));
    InitSizeLists();
    return true;
  }

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    auto vel = GetDeviceAddress<float>(inputs, 0);
    auto crd = GetDeviceAddress<float>(inputs, 1);
    auto frc = GetDeviceAddress<float>(inputs, 2);
    auto acc = GetDeviceAddress<float>(inputs, 3);
    auto inverse_mass = GetDeviceAddress<float>(inputs, 4);

    MDIterationLeapFrogWithMaxVelocity(atom_numbers, vel, crd, frc, acc, inverse_mass, dt, max_velocity,
                                       reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(atom_numbers * 3 * sizeof(T));
    input_size_list_.push_back(atom_numbers * 3 * sizeof(T));
    input_size_list_.push_back(atom_numbers * 3 * sizeof(T));
    input_size_list_.push_back(atom_numbers * 3 * sizeof(T));
    input_size_list_.push_back(atom_numbers * sizeof(T));

    output_size_list_.push_back(sizeof(T));
  }

 private:
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
  int atom_numbers;
  float dt;
  float max_velocity;
};
}  // namespace kernel
}  // namespace mindspore
#endif
