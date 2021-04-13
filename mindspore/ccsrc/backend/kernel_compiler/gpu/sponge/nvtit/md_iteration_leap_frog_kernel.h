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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPONG_NVTIT_MD_ITERATION_LEAP_FROG_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPONG_NVTIT_MD_ITERATION_LEAP_FROG_KERNEL_H_

#include <cuda_runtime_api.h>
#include <vector>
#include <string>
#include <map>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "runtime/device/gpu/cuda_common.h"
#include "backend/kernel_compiler/gpu/cuda_impl/sponge/nvtit/md_iteration_leap_frog_impl.cuh"

namespace mindspore {
namespace kernel {
template <typename T>
class MDIterationLeapFrogGpuKernel : public GpuKernel {
 public:
  MDIterationLeapFrogGpuKernel() : ele_mass_inverse(1) {}
  ~MDIterationLeapFrogGpuKernel() override = default;

  bool Init(const CNodePtr &kernel_node) override {
    kernel_node_ = kernel_node;
    float4_numbers = static_cast<int>(GetAttr<int64_t>(kernel_node, "float4_numbers"));
    atom_numbers = static_cast<int>(GetAttr<int64_t>(kernel_node, "atom_numbers"));
    half_dt = static_cast<float>(GetAttr<float>(kernel_node, "half_dt"));
    dt = static_cast<float>(GetAttr<float>(kernel_node, "dt"));
    exp_gamma = static_cast<float>(GetAttr<float>(kernel_node, "exp_gamma"));
    is_max_velocity = static_cast<int>(GetAttr<int64_t>(kernel_node, "is_max_velocity"));
    max_velocity = static_cast<float>(GetAttr<float>(kernel_node, "max_velocity"));

    auto shape_mass_inverse = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    auto shape_qrt_mass = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);

    for (size_t i = 0; i < shape_mass_inverse.size(); i++) ele_mass_inverse *= shape_mass_inverse[i];
    for (size_t i = 0; i < shape_qrt_mass.size(); i++) ele_sqrt_mass *= shape_qrt_mass[i];

    InitSizeLists();
    return true;
  }

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    auto d_mass_inverse = GetDeviceAddress<const T>(inputs, 0);
    auto d_sqrt_mass = GetDeviceAddress<const T>(inputs, 1);

    auto vel_f = GetDeviceAddress<T>(outputs, 0);
    auto crd_f = GetDeviceAddress<T>(outputs, 1);
    auto frc_f = GetDeviceAddress<T>(outputs, 2);
    auto acc_f = GetDeviceAddress<T>(outputs, 3);

    MDIterationLeapFrog(float4_numbers, atom_numbers, half_dt, dt, exp_gamma, is_max_velocity, max_velocity,
                        d_mass_inverse, d_sqrt_mass, vel_f, crd_f, frc_f, acc_f,
                        reinterpret_cast<cudaStream_t>(stream_ptr));

    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(ele_mass_inverse * sizeof(T));
    input_size_list_.push_back(ele_sqrt_mass * sizeof(T));

    output_size_list_.push_back(3 * atom_numbers * sizeof(T));
    output_size_list_.push_back(3 * atom_numbers * sizeof(T));
    output_size_list_.push_back(3 * atom_numbers * sizeof(T));
    output_size_list_.push_back(3 * atom_numbers * sizeof(T));
  }

 private:
  size_t ele_mass_inverse = 1;
  size_t ele_sqrt_mass = 1;

  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
  int float4_numbers;
  int atom_numbers;
  float half_dt;
  float dt;
  float exp_gamma;
  int is_max_velocity;
  float max_velocity;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPONG_NVTIT_MD_ITERATION_LEAP_FROG_KERNEL_H_
