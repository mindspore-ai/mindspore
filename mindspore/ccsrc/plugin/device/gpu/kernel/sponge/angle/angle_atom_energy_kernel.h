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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPONGE_ANGLE_ANGLE_ATOM_ENERGY_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPONGE_ANGLE_ANGLE_ATOM_ENERGY_KERNEL_H_
#include <cuda_runtime_api.h>
#include <vector>
#include <string>
#include <map>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/hal/device/cuda_common.h"
#include "plugin/device/gpu/kernel/cuda_impl/sponge/angle/angle_atom_energy_impl.cuh"
namespace mindspore {
namespace kernel {
template <typename T, typename T1>
class AngleAtomEnergyGpuKernelMod : public NativeGpuKernelMod {
 public:
  AngleAtomEnergyGpuKernelMod() : ele_uint_crd(1) {}
  ~AngleAtomEnergyGpuKernelMod() override = default;

  bool Init(const CNodePtr &kernel_node) override {
    kernel_node_ = kernel_node;
    angle_numbers = static_cast<int>(GetAttr<int64_t>(kernel_node, "angle_numbers"));
    auto shape_uint_crd = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    auto shape_scaler = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
    auto shape_atom_a = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 2);
    auto shape_atom_b = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 3);
    auto shape_atom_c = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 4);
    auto shape_angle_k = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 5);
    auto shape_angle_theta0 = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 6);

    for (size_t i = 0; i < shape_uint_crd.size(); i++) ele_uint_crd *= shape_uint_crd[i];
    for (size_t i = 0; i < shape_scaler.size(); i++) ele_scaler *= shape_scaler[i];
    for (size_t i = 0; i < shape_atom_a.size(); i++) ele_atom_a *= shape_atom_a[i];
    for (size_t i = 0; i < shape_atom_b.size(); i++) ele_atom_b *= shape_atom_b[i];
    for (size_t i = 0; i < shape_atom_c.size(); i++) ele_atom_c *= shape_atom_c[i];
    for (size_t i = 0; i < shape_angle_k.size(); i++) ele_angle_k *= shape_angle_k[i];
    for (size_t i = 0; i < shape_angle_theta0.size(); i++) ele_angle_theta0 *= shape_angle_theta0[i];
    InitSizeLists();
    return true;
  }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    auto uint_crd_f = GetDeviceAddress<const T1>(inputs, 0);
    auto scaler_f = GetDeviceAddress<T>(inputs, 1);
    auto atom_a = GetDeviceAddress<const T1>(inputs, 2);
    auto atom_b = GetDeviceAddress<const T1>(inputs, 3);
    auto atom_c = GetDeviceAddress<const T1>(inputs, 4);
    auto angle_k = GetDeviceAddress<T>(inputs, 5);
    auto angle_theta0 = GetDeviceAddress<T>(inputs, 6);

    auto ene = GetDeviceAddress<T>(outputs, 0);
    AngleAtomEnergy(angle_numbers, ele_uint_crd, uint_crd_f, scaler_f, atom_a, atom_b, atom_c, angle_k, angle_theta0,
                    ene, reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(ele_uint_crd * sizeof(T1));
    input_size_list_.push_back(ele_scaler * sizeof(T));
    input_size_list_.push_back(ele_atom_a * sizeof(T1));
    input_size_list_.push_back(ele_atom_b * sizeof(T1));
    input_size_list_.push_back(ele_atom_c * sizeof(T1));
    input_size_list_.push_back(ele_angle_k * sizeof(T));
    input_size_list_.push_back(ele_angle_theta0 * sizeof(T));

    output_size_list_.push_back(ele_uint_crd * sizeof(T));
  }

 private:
  size_t ele_uint_crd = 1;
  size_t ele_scaler = 1;
  size_t ele_atom_a = 1;
  size_t ele_atom_b = 1;
  size_t ele_atom_c = 1;
  size_t ele_angle_k = 1;
  size_t ele_angle_theta0 = 1;

  int angle_numbers;
};
}  // namespace kernel
}  // namespace mindspore
#endif
