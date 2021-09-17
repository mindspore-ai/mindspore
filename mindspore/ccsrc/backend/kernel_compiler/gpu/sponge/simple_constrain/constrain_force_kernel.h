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
/**
 * Note:
 *  ConstrainForce. This is an experimental interface that is subject to change and/or deletion.
 */

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPONG_SIMPLE_CONSTRAIN_CONSTRAIN_FORCE_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPONG_SIMPLE_CONSTRAIN_CONSTRAIN_FORCE_KERNEL_H_

#include "backend/kernel_compiler/gpu/cuda_impl/sponge/simple_constrain/constrain_force_virial_impl.cuh"

#include <cuda_runtime_api.h>
#include <map>
#include <string>
#include <vector>

#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "runtime/device/gpu/cuda_common.h"

namespace mindspore {
namespace kernel {
template <typename T, typename T1, typename T2>
class ConstrainForceGpuKernel : public GpuKernel {
 public:
  ConstrainForceGpuKernel() : ele_crd(1) {}
  ~ConstrainForceGpuKernel() override = default;

  bool Init(const CNodePtr &kernel_node) override {
    kernel_node_ = kernel_node;
    atom_numbers = static_cast<int>(GetAttr<int64_t>(kernel_node, "atom_numbers"));
    constrain_pair_numbers = static_cast<int>(GetAttr<int64_t>(kernel_node, "constrain_pair_numbers"));
    iteration_numbers = static_cast<int>(GetAttr<int64_t>(kernel_node, "iteration_numbers"));
    half_exp_gamma_plus_half = static_cast<float>(GetAttr<float>(kernel_node, "half_exp_gamma_plus_half"));

    auto shape_crd = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    auto shape_quarter_cof = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
    auto shape_mass_inverse = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 2);

    auto shape_scaler = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 3);
    auto shape_pair_dr = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 4);
    auto shape_atom_i_serials = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 5);
    auto shape_atom_j_serials = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 6);
    auto shape_constant_rs = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 7);
    auto shape_constrain_ks = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 8);

    for (size_t i = 0; i < shape_scaler.size(); i++) ele_scaler *= shape_scaler[i];
    for (size_t i = 0; i < shape_pair_dr.size(); i++) ele_pair_dr *= shape_pair_dr[i];
    for (size_t i = 0; i < shape_atom_i_serials.size(); i++) ele_atom_i_serials *= shape_atom_i_serials[i];
    for (size_t i = 0; i < shape_atom_j_serials.size(); i++) ele_atom_j_serials *= shape_atom_j_serials[i];
    for (size_t i = 0; i < shape_constant_rs.size(); i++) ele_constant_rs *= shape_constant_rs[i];
    for (size_t i = 0; i < shape_constrain_ks.size(); i++) ele_constrain_ks *= shape_constrain_ks[i];

    for (size_t i = 0; i < shape_crd.size(); i++) ele_crd *= shape_crd[i];
    for (size_t i = 0; i < shape_quarter_cof.size(); i++) ele_quarter_cof *= shape_quarter_cof[i];
    for (size_t i = 0; i < shape_mass_inverse.size(); i++) ele_mass_inverse *= shape_mass_inverse[i];

    InitSizeLists();
    return true;
  }

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    auto crd = GetDeviceAddress<const T>(inputs, 0);
    auto quarter_cof = GetDeviceAddress<const T>(inputs, 1);
    auto mass_inverse = GetDeviceAddress<const T>(inputs, 2);

    auto scaler = GetDeviceAddress<const T>(inputs, 3);
    auto pair_dr = GetDeviceAddress<const T>(inputs, 4);
    auto atom_i_serials = GetDeviceAddress<const T1>(inputs, 5);
    auto atom_j_serials = GetDeviceAddress<const T1>(inputs, 6);
    auto constant_rs = GetDeviceAddress<const T>(inputs, 7);
    auto constrain_ks = GetDeviceAddress<const T>(inputs, 8);

    auto constrain_pair = GetDeviceAddress<T>(workspace, 0);

    auto uint_crd = GetDeviceAddress<T2>(outputs, 0);

    auto test_frc_f = GetDeviceAddress<T>(outputs, 1);
    auto d_atom_virial = GetDeviceAddress<T>(outputs, 2);

    set_zero_force_with_virial(atom_numbers, constrain_pair_numbers, test_frc_f, d_atom_virial,
                               reinterpret_cast<cudaStream_t>(stream_ptr));

    for (int i = 0; i < iteration_numbers; i++) {
      refresh_uint_crd_update(atom_numbers, half_exp_gamma_plus_half, crd, quarter_cof, test_frc_f, mass_inverse,
                              uint_crd, reinterpret_cast<cudaStream_t>(stream_ptr));

      constrain_force_cycle_update(atom_numbers, constrain_pair_numbers, uint_crd, scaler, constrain_pair, pair_dr,
                                   atom_i_serials, atom_j_serials, constant_rs, constrain_ks, test_frc_f,
                                   reinterpret_cast<cudaStream_t>(stream_ptr));
    }
    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(ele_crd * sizeof(T));
    input_size_list_.push_back(ele_quarter_cof * sizeof(T));
    input_size_list_.push_back(ele_mass_inverse * sizeof(T));

    input_size_list_.push_back(ele_scaler * sizeof(T));
    input_size_list_.push_back(ele_pair_dr * sizeof(T));
    input_size_list_.push_back(ele_atom_i_serials * sizeof(T1));
    input_size_list_.push_back(ele_atom_j_serials * sizeof(T1));
    input_size_list_.push_back(ele_constant_rs * sizeof(T));
    input_size_list_.push_back(ele_constrain_ks * sizeof(T));

    workspace_size_list_.push_back(constrain_pair_numbers * sizeof(CONSTRAIN_PAIR));

    output_size_list_.push_back(3 * atom_numbers * sizeof(T2));
    output_size_list_.push_back(3 * atom_numbers * sizeof(T));
    output_size_list_.push_back(constrain_pair_numbers * sizeof(T));
  }

 private:
  size_t ele_scaler = 1;
  size_t ele_pair_dr = 1;
  size_t ele_atom_i_serials = 1;
  size_t ele_atom_j_serials = 1;
  size_t ele_constant_rs = 1;
  size_t ele_constrain_ks = 1;
  size_t ele_crd = 1;
  size_t ele_quarter_cof = 1;
  size_t ele_mass_inverse = 1;

  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
  int atom_numbers;
  int constrain_pair_numbers;
  int iteration_numbers;
  int need_pressure;
  float half_exp_gamma_plus_half;
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
