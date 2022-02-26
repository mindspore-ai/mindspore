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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPONG_SIMPLE_CONSTRAIN_LAST_CRD_TO_DR_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPONG_SIMPLE_CONSTRAIN_LAST_CRD_TO_DR_KERNEL_H_

#include "plugin/device/gpu/kernel/cuda_impl/sponge/simple_constrain/last_crd_to_dr_impl.cuh"

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
class LastCrdToDrGpuKernelMod : public NativeGpuKernelMod {
 public:
  LastCrdToDrGpuKernelMod() : ele_atom_crd(1) {}
  ~LastCrdToDrGpuKernelMod() override = default;

  bool Init(const CNodePtr &kernel_node) override {
    // get bond_numbers
    kernel_node_ = kernel_node;
    constrain_pair_numbers = static_cast<int>(GetAttr<int64_t>(kernel_node, "constrain_pair_numbers"));

    auto shape_atom_crd = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    auto shape_quater_cof = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
    auto shape_uint_dr_to_dr = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 2);
    auto shape_atom_i_serials = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 3);
    auto shape_atom_j_serials = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 4);
    auto shape_constant_rs = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 5);
    auto shape_constrain_ks = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 6);

    for (size_t i = 0; i < shape_atom_crd.size(); i++) ele_atom_crd *= shape_atom_crd[i];
    for (size_t i = 0; i < shape_quater_cof.size(); i++) ele_quater_cof *= shape_quater_cof[i];
    for (size_t i = 0; i < shape_uint_dr_to_dr.size(); i++) ele_uint_dr_to_dr *= shape_uint_dr_to_dr[i];
    for (size_t i = 0; i < shape_atom_i_serials.size(); i++) ele_atom_i_serials *= shape_atom_i_serials[i];
    for (size_t i = 0; i < shape_atom_j_serials.size(); i++) ele_atom_j_serials *= shape_atom_j_serials[i];
    for (size_t i = 0; i < shape_constant_rs.size(); i++) ele_constant_rs *= shape_constant_rs[i];
    for (size_t i = 0; i < shape_constrain_ks.size(); i++) ele_constrain_ks *= shape_constrain_ks[i];

    InitSizeLists();
    return true;
  }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    auto atom_crd = GetDeviceAddress<const T>(inputs, 0);
    auto quater_cof = GetDeviceAddress<const T>(inputs, 1);
    auto uint_dr_to_dr = GetDeviceAddress<const T>(inputs, 2);
    auto atom_i_serials = GetDeviceAddress<const T1>(inputs, 3);
    auto atom_j_serials = GetDeviceAddress<const T1>(inputs, 4);
    auto constant_rs = GetDeviceAddress<const T>(inputs, 5);
    auto constrain_ks = GetDeviceAddress<const T>(inputs, 6);

    auto constrain_pair = GetDeviceAddress<T>(workspace, 0);

    auto pair_dr = GetDeviceAddress<T>(outputs, 0);

    lastcrdtodr(constrain_pair_numbers, atom_crd, quater_cof, uint_dr_to_dr, constrain_pair, atom_i_serials,
                atom_j_serials, constant_rs, constrain_ks, pair_dr, reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(ele_atom_crd * sizeof(T));
    input_size_list_.push_back(ele_quater_cof * sizeof(T));
    input_size_list_.push_back(ele_uint_dr_to_dr * sizeof(T));
    input_size_list_.push_back(ele_atom_i_serials * sizeof(T1));
    input_size_list_.push_back(ele_atom_j_serials * sizeof(T1));
    input_size_list_.push_back(ele_constant_rs * sizeof(T));
    input_size_list_.push_back(ele_constrain_ks * sizeof(T));

    workspace_size_list_.push_back(constrain_pair_numbers * sizeof(CONSTRAIN_PAIR));

    output_size_list_.push_back(3 * constrain_pair_numbers * sizeof(T));
  }

 private:
  size_t ele_atom_crd = 1;
  size_t ele_quater_cof = 1;
  size_t ele_uint_dr_to_dr = 1;
  size_t ele_atom_i_serials = 1;
  size_t ele_atom_j_serials = 1;
  size_t ele_constant_rs = 1;
  size_t ele_constrain_ks = 1;

  int constrain_pair_numbers;
  struct CONSTRAIN_PAIR {
    int atom_i_serial;
    int atom_j_serial;
    float constant_r;
    float constrain_k;
  };
};
}  // namespace kernel
}  // namespace mindspore
#endif
