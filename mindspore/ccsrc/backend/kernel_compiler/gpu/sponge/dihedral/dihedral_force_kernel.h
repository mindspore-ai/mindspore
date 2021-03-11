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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPONGE_DIHEDRAL_DIHEDRAL_FORCE_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPONGE_DIHEDRAL_DIHEDRAL_FORCE_KERNEL_H_
#include <cuda_runtime_api.h>
#include <vector>
#include <string>
#include <map>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "runtime/device/gpu/cuda_common.h"
#include "backend/kernel_compiler/gpu/cuda_impl/sponge/dihedral/dihedral_force_impl.cuh"
namespace mindspore {
namespace kernel {
template <typename T, typename T1>
class DihedralForceGpuKernel : public GpuKernel {
 public:
  DihedralForceGpuKernel() : ele_uint_crd(1) {}
  ~DihedralForceGpuKernel() override = default;

  bool Init(const CNodePtr &kernel_node) override {
    kernel_node_ = kernel_node;
    dihedral_numbers = static_cast<int>(GetAttr<int64_t>(kernel_node, "dihedral_numbers"));
    auto shape_uint_crd = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    auto shape_scaler = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
    auto shape_atom_a = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 2);
    auto shape_atom_b = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 3);
    auto shape_atom_c = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 4);
    auto shape_atom_d = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 5);
    auto shape_ipn = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 6);
    auto shape_pk = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 7);
    auto shape_gamc = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 8);
    auto shape_gams = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 9);
    auto shape_pn = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 10);

    for (size_t i = 0; i < shape_uint_crd.size(); i++) ele_uint_crd *= shape_uint_crd[i];
    for (size_t i = 0; i < shape_scaler.size(); i++) ele_scaler *= shape_scaler[i];
    for (size_t i = 0; i < shape_atom_a.size(); i++) ele_atom_a *= shape_atom_a[i];
    for (size_t i = 0; i < shape_atom_b.size(); i++) ele_atom_b *= shape_atom_b[i];
    for (size_t i = 0; i < shape_atom_c.size(); i++) ele_atom_c *= shape_atom_c[i];
    for (size_t i = 0; i < shape_atom_d.size(); i++) ele_atom_d *= shape_atom_d[i];
    for (size_t i = 0; i < shape_ipn.size(); i++) ele_ipn *= shape_ipn[i];
    for (size_t i = 0; i < shape_pk.size(); i++) ele_pk *= shape_pk[i];
    for (size_t i = 0; i < shape_gamc.size(); i++) ele_gamc *= shape_gamc[i];
    for (size_t i = 0; i < shape_gams.size(); i++) ele_gams *= shape_gams[i];
    for (size_t i = 0; i < shape_pn.size(); i++) ele_pn *= shape_pn[i];
    InitSizeLists();
    return true;
  }

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    auto uint_crd_f = GetDeviceAddress<const T1>(inputs, 0);
    auto scaler_f = GetDeviceAddress<T>(inputs, 1);
    auto atom_a = GetDeviceAddress<const T1>(inputs, 2);
    auto atom_b = GetDeviceAddress<const T1>(inputs, 3);
    auto atom_c = GetDeviceAddress<const T1>(inputs, 4);
    auto atom_d = GetDeviceAddress<const T1>(inputs, 5);
    auto ipn = GetDeviceAddress<const T1>(inputs, 6);
    auto pk = GetDeviceAddress<T>(inputs, 7);
    auto gamc = GetDeviceAddress<T>(inputs, 8);
    auto gams = GetDeviceAddress<T>(inputs, 9);
    auto pn = GetDeviceAddress<T>(inputs, 10);

    auto frc_f = GetDeviceAddress<T>(outputs, 0);
    DihedralForce(dihedral_numbers, ele_uint_crd, uint_crd_f, scaler_f, atom_a, atom_b, atom_c, atom_d, ipn, pk, gamc,
                  gams, pn, frc_f, reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(ele_uint_crd * sizeof(T1));
    input_size_list_.push_back(ele_scaler * sizeof(T));
    input_size_list_.push_back(ele_atom_a * sizeof(T1));
    input_size_list_.push_back(ele_atom_b * sizeof(T1));
    input_size_list_.push_back(ele_atom_c * sizeof(T1));
    input_size_list_.push_back(ele_atom_d * sizeof(T1));
    input_size_list_.push_back(ele_ipn * sizeof(T1));
    input_size_list_.push_back(ele_pk * sizeof(T));
    input_size_list_.push_back(ele_gamc * sizeof(T));
    input_size_list_.push_back(ele_gams * sizeof(T));
    input_size_list_.push_back(ele_pn * sizeof(T));

    output_size_list_.push_back(ele_uint_crd * 3 * sizeof(T));
  }

 private:
  size_t ele_uint_crd = 1;
  size_t ele_scaler = 1;
  size_t ele_atom_a = 1;
  size_t ele_atom_b = 1;
  size_t ele_atom_c = 1;
  size_t ele_atom_d = 1;
  size_t ele_ipn = 1;
  size_t ele_pk = 1;
  size_t ele_gamc = 1;
  size_t ele_gams = 1;
  size_t ele_pn = 1;

  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
  int dihedral_numbers;
};
}  // namespace kernel
}  // namespace mindspore
#endif
