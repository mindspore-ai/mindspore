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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPONG_BOND_BOND_ATOM_ENERGY_CUDA_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPONG_BOND_BOND_ATOM_ENERGY_CUDA_GPU_KERNEL_H_

#include "plugin/device/gpu/kernel/cuda_impl/sponge/bond/bond_atom_energy_cuda_gpu_impl.cuh"

#include <cuda_runtime_api.h>
#include <map>
#include <string>
#include <vector>

#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/hal/device/cuda_common.h"

namespace mindspore {
namespace kernel {
template <typename T, typename T1>
class BondAtomEnergyCudaGpuKernelMod : public NativeGpuKernelMod {
 public:
  BondAtomEnergyCudaGpuKernelMod() : ele_uint_crd(1) {}
  ~BondAtomEnergyCudaGpuKernelMod() override = default;

  bool Init(const CNodePtr &kernel_node) override {
    kernel_node_ = kernel_node;
    bond_numbers = static_cast<int>(GetAttr<int64_t>(kernel_node, "bond_numbers"));
    atom_numbers = static_cast<int>(GetAttr<int64_t>(kernel_node, "atom_numbers"));
    auto shape_uint_crd = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    auto shape_scaler = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
    auto shape_atom_a = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 2);
    auto shape_atom_b = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 3);
    auto shape_bond_k = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 4);
    auto shape_bond_r0 = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 5);

    for (size_t i = 0; i < shape_uint_crd.size(); i++) ele_uint_crd *= shape_uint_crd[i];
    for (size_t i = 0; i < shape_scaler.size(); i++) ele_scaler *= shape_scaler[i];
    for (size_t i = 0; i < shape_atom_a.size(); i++) ele_atom_a *= shape_atom_a[i];
    for (size_t i = 0; i < shape_atom_b.size(); i++) ele_atom_b *= shape_atom_b[i];
    for (size_t i = 0; i < shape_bond_k.size(); i++) ele_bond_k *= shape_bond_k[i];
    for (size_t i = 0; i < shape_bond_r0.size(); i++) ele_bond_r0 *= shape_bond_r0[i];

    InitSizeLists();
    return true;
  }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    auto uint_crd_f = GetDeviceAddress<const T1>(inputs, 0);
    auto scaler_f = GetDeviceAddress<T>(inputs, 1);
    auto atom_a = GetDeviceAddress<const T1>(inputs, 2);
    auto atom_b = GetDeviceAddress<const T1>(inputs, 3);
    auto bond_k = GetDeviceAddress<T>(inputs, 4);
    auto bond_r0 = GetDeviceAddress<T>(inputs, 5);

    auto atom_ene = GetDeviceAddress<T>(outputs, 0);

    BondAtomEnergy(bond_numbers, atom_numbers, uint_crd_f, scaler_f, atom_a, atom_b, bond_k, bond_r0, atom_ene,
                   reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(ele_uint_crd * sizeof(T1));
    input_size_list_.push_back(ele_scaler * sizeof(T));
    input_size_list_.push_back(ele_atom_a * sizeof(T1));
    input_size_list_.push_back(ele_atom_b * sizeof(T1));
    input_size_list_.push_back(ele_bond_k * sizeof(T));
    input_size_list_.push_back(ele_bond_r0 * sizeof(T));

    output_size_list_.push_back(atom_numbers * sizeof(T));
  }

 private:
  size_t ele_uint_crd = 1;
  size_t ele_scaler = 1;
  size_t ele_atom_a = 1;
  size_t ele_atom_b = 1;
  size_t ele_bond_k = 1;
  size_t ele_bond_r0 = 1;

  int bond_numbers;
  int atom_numbers;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPONG_BOND_BOND_ATOM_ENERGY_CUDA_GPU_KERNEL_H_
