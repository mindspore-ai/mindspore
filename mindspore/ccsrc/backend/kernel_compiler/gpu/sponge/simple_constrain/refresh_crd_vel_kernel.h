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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPONG_SIMPLE_CONSTRAIN_REFRESH_CRD_VEL_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPONG_SIMPLE_CONSTRAIN_REFRESH_CRD_VEL_KERNEL_H_

#include "backend/kernel_compiler/gpu/cuda_impl/sponge/simple_constrain/refresh_crd_vel_impl.cuh"

#include <cuda_runtime_api.h>
#include <map>
#include <string>
#include <vector>

#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "runtime/device/gpu/cuda_common.h"

namespace mindspore {
namespace kernel {
template <typename T, typename T1>
class RefreshCrdVelGpuKernel : public GpuKernel {
 public:
  RefreshCrdVelGpuKernel() : ele_crd(1) {}
  ~RefreshCrdVelGpuKernel() override = default;

  bool Init(const CNodePtr &kernel_node) override {
    // get bond_numbers
    kernel_node_ = kernel_node;
    atom_numbers = static_cast<int>(GetAttr<int64_t>(kernel_node, "atom_numbers"));
    dt_inverse = static_cast<float>(GetAttr<float>(kernel_node, "dt_inverse"));
    dt = static_cast<float>(GetAttr<float>(kernel_node, "dt"));
    exp_gamma = static_cast<float>(GetAttr<float>(kernel_node, "exp_gamma"));
    half_exp_gamma_plus_half = static_cast<float>(GetAttr<float>(kernel_node, "half_exp_gamma_plus_half"));
    auto shape_crd = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    auto shape_vel = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
    auto shape_test_frc = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 2);
    auto shape_mass_inverse = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 3);

    for (size_t i = 0; i < shape_crd.size(); i++) ele_crd *= shape_crd[i];
    for (size_t i = 0; i < shape_vel.size(); i++) ele_vel *= shape_vel[i];
    for (size_t i = 0; i < shape_test_frc.size(); i++) ele_test_frc *= shape_test_frc[i];
    for (size_t i = 0; i < shape_mass_inverse.size(); i++) ele_mass_inverse *= shape_mass_inverse[i];

    InitSizeLists();
    return true;
  }

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    auto crd = GetDeviceAddress<T>(inputs, 0);
    auto vel = GetDeviceAddress<T>(inputs, 1);
    auto test_frc = GetDeviceAddress<T>(inputs, 2);
    auto mass_inverse = GetDeviceAddress<T>(inputs, 3);

    refreshcrdvel(atom_numbers, dt_inverse, dt, exp_gamma, half_exp_gamma_plus_half, test_frc, mass_inverse, crd, vel,
                  reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(ele_crd * sizeof(T));
    input_size_list_.push_back(ele_vel * sizeof(T));
    input_size_list_.push_back(ele_test_frc * sizeof(T));
    input_size_list_.push_back(ele_mass_inverse * sizeof(T));

    output_size_list_.push_back(sizeof(T));
  }

 private:
  size_t ele_crd = 1;
  size_t ele_vel = 1;
  size_t ele_test_frc = 1;
  size_t ele_mass_inverse = 1;

  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
  int atom_numbers;
  float dt_inverse;
  float dt;
  float exp_gamma;
  float half_exp_gamma_plus_half;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPONG_SIMPLE_CONSTRAIN_REFRESH_CRD_VEL_KERNEL_H_
