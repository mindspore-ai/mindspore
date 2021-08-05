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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPONG_RESTRAIN_RESTRAIN_FORCE_ATOM_ENERGY_VIRIAL_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPONG_RESTRAIN_RESTRAIN_FORCE_ATOM_ENERGY_VIRIAL_KERNEL_H_

#include "backend/kernel_compiler/gpu/cuda_impl/sponge/restrain/restrain_force_atom_energy_virial_impl.cuh"

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
class RestrainForceWithAtomenergyAndVirialGpuKernel : public GpuKernel {
 public:
  RestrainForceWithAtomenergyAndVirialGpuKernel() : ele_crd(1) {}
  ~RestrainForceWithAtomenergyAndVirialGpuKernel() override = default;

  bool Init(const CNodePtr &kernel_node) override {
    // get bond_numbers
    kernel_node_ = kernel_node;
    restrain_numbers = static_cast<int>(GetAttr<int64_t>(kernel_node, "restrain_numbers"));
    atom_numbers = static_cast<int>(GetAttr<int64_t>(kernel_node, "atom_numbers"));
    weight = static_cast<int>(GetAttr<float>(kernel_node, "weight"));
    auto shape_restrain_list = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    auto shape_crd = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
    auto shape_crd_ref = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 2);
    auto shape_scaler = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 3);

    for (size_t i = 0; i < shape_crd.size(); i++) ele_crd *= shape_crd[i];
    for (size_t i = 0; i < shape_scaler.size(); i++) ele_scaler *= shape_scaler[i];
    for (size_t i = 0; i < shape_restrain_list.size(); i++) ele_restrain_list *= shape_restrain_list[i];
    for (size_t i = 0; i < shape_crd_ref.size(); i++) ele_crd_ref *= shape_crd_ref[i];

    InitSizeLists();
    return true;
  }

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    auto restrain_list = GetDeviceAddress<const T1>(inputs, 0);
    auto crd_f = GetDeviceAddress<const T>(inputs, 1);
    auto crd_ref = GetDeviceAddress<const T>(inputs, 2);
    auto scaler_f = GetDeviceAddress<T>(inputs, 3);

    auto atom_ene = GetDeviceAddress<T>(outputs, 0);
    auto atom_virial = GetDeviceAddress<T>(outputs, 1);
    auto frc = GetDeviceAddress<T>(outputs, 2);

    restrainforcewithatomenergyandvirial(restrain_numbers, atom_numbers, restrain_list, crd_f, crd_ref, weight,
                                         scaler_f, atom_ene, atom_virial, frc,
                                         reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(ele_restrain_list * sizeof(T1));
    input_size_list_.push_back(ele_crd * sizeof(T));
    input_size_list_.push_back(ele_crd_ref * sizeof(T));
    input_size_list_.push_back(ele_scaler * sizeof(T));

    output_size_list_.push_back(atom_numbers * sizeof(T));
    output_size_list_.push_back(atom_numbers * sizeof(T));
    output_size_list_.push_back(3 * atom_numbers * sizeof(T));
  }

 private:
  size_t ele_crd = 1;
  size_t ele_scaler = 1;
  size_t ele_restrain_list = 1;
  size_t ele_crd_ref = 1;

  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
  int restrain_numbers;
  int atom_numbers;
  float weight;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPONG_RESTRAIN_RESTRAIN_FORCE_KERNEL_H_
