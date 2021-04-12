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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPONG_NB14_DIHEDRAL_14_CF_ENERGY_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPONG_NB14_DIHEDRAL_14_CF_ENERGY_KERNEL_H_

#include <cuda_runtime_api.h>
#include <vector>
#include <string>
#include <map>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "runtime/device/gpu/cuda_common.h"
#include "backend/kernel_compiler/gpu/cuda_impl/sponge/nb14/dihedral_14_cf_energy_impl.cuh"

namespace mindspore {
namespace kernel {
template <typename T, typename T1>
class Dihedral14CFEnergyGpuKernel : public GpuKernel {
 public:
  Dihedral14CFEnergyGpuKernel() : ele_uint_crd(1) {}
  ~Dihedral14CFEnergyGpuKernel() override = default;

  bool Init(const CNodePtr &kernel_node) override {
    kernel_node_ = kernel_node;
    dihedral_14_numbers = static_cast<int>(GetAttr<int64_t>(kernel_node, "dihedral_14_numbers"));
    atom_numbers = static_cast<int>(GetAttr<int64_t>(kernel_node, "atom_numbers"));

    auto shape_uint_crd = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    auto shape_LJtype = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
    auto shape_charge = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 2);
    auto shape_boxlength_f = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 3);
    auto shape_a_14 = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 4);
    auto shape_b_14 = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 5);
    auto shape_cf_scale_factor = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 6);

    for (size_t i = 0; i < shape_uint_crd.size(); i++) ele_uint_crd *= shape_uint_crd[i];
    for (size_t i = 0; i < shape_LJtype.size(); i++) ele_LJtype *= shape_LJtype[i];
    for (size_t i = 0; i < shape_charge.size(); i++) ele_charge *= shape_charge[i];
    for (size_t i = 0; i < shape_boxlength_f.size(); i++) ele_boxlength_f *= shape_boxlength_f[i];
    for (size_t i = 0; i < shape_a_14.size(); i++) ele_a_14 *= shape_a_14[i];
    for (size_t i = 0; i < shape_b_14.size(); i++) ele_b_14 *= shape_b_14[i];
    for (size_t i = 0; i < shape_cf_scale_factor.size(); i++) ele_cf_scale_factor *= shape_cf_scale_factor[i];

    InitSizeLists();
    return true;
  }

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    auto uint_crd_f = GetDeviceAddress<const T1>(inputs, 0);
    auto LJtype = GetDeviceAddress<const T1>(inputs, 1);
    auto charge = GetDeviceAddress<const T>(inputs, 2);
    auto boxlength_f = GetDeviceAddress<T>(inputs, 3);
    auto a_14 = GetDeviceAddress<const T1>(inputs, 4);
    auto b_14 = GetDeviceAddress<const T1>(inputs, 5);
    auto cf_scale_factor = GetDeviceAddress<T>(inputs, 6);
    auto ene = GetDeviceAddress<T>(outputs, 0);
    auto uint_crd_with_LJ = GetDeviceAddress<T>(workspace, 0);

    Dihedral14CFEnergy(dihedral_14_numbers, atom_numbers, uint_crd_f, LJtype, charge, uint_crd_with_LJ, boxlength_f,
                       a_14, b_14, cf_scale_factor, ene, reinterpret_cast<cudaStream_t>(stream_ptr));

    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(ele_uint_crd * sizeof(T1));
    input_size_list_.push_back(ele_LJtype * sizeof(T1));
    input_size_list_.push_back(ele_charge * sizeof(T));
    input_size_list_.push_back(ele_boxlength_f * sizeof(T));
    input_size_list_.push_back(ele_a_14 * sizeof(T1));
    input_size_list_.push_back(ele_b_14 * sizeof(T1));
    input_size_list_.push_back(ele_cf_scale_factor * sizeof(T));
    workspace_size_list_.push_back(atom_numbers * sizeof(UINT_VECTOR_LJ_TYPE));
    output_size_list_.push_back(dihedral_14_numbers * sizeof(T));
  }

 private:
  size_t ele_uint_crd = 1;
  size_t ele_LJtype = 1;
  size_t ele_charge = 1;
  size_t ele_boxlength_f = 1;
  size_t ele_a_14 = 1;
  size_t ele_b_14 = 1;
  size_t ele_cf_scale_factor = 1;

  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
  int dihedral_14_numbers;
  int atom_numbers;
  struct UINT_VECTOR_LJ_TYPE {
    unsigned int uint_x;
    unsigned int uint_y;
    unsigned int uint_z;
    int LJ_type;
    float charge;
  };
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPONG_NB14_DIHEDRAL_14_LJ_FORCE_WITH_DIRECT_CF_KERNEL_H_
