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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPONG_VATOM_V2_FORCE_REDISTRIBUTE_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPONG_VATOM_V2_FORCE_REDISTRIBUTE_KERNEL_H_

#include "backend/kernel_compiler/gpu/cuda_impl/sponge/vatom/v2_force_redistribute_impl.cuh"

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
class v2ForceRedistributeGpuKernel : public GpuKernel {
 public:
  v2ForceRedistributeGpuKernel() : ele_uint_crd(1) {}
  ~v2ForceRedistributeGpuKernel() override = default;

  bool Init(const CNodePtr &kernel_node) override {
    // get bond_numbers
    kernel_node_ = kernel_node;
    atom_numbers = static_cast<int>(GetAttr<int64_t>(kernel_node, "atom_numbers"));
    // virtual_numbers = static_cast<int>(GetAttr<int64_t>(kernel_node, "virtual_numbers"));
    auto shape_virtual_numbers = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    auto shape_uint_crd = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
    auto shape_v_info = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 2);
    auto shape_frc = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 3);

    for (size_t i = 0; i < shape_uint_crd.size(); i++) ele_uint_crd *= shape_uint_crd[i];
    for (size_t i = 0; i < shape_v_info.size(); i++) ele_v_info *= shape_v_info[i];
    for (size_t i = 0; i < shape_frc.size(); i++) ele_frc *= shape_frc[i];

    InitSizeLists();
    return true;
  }

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    auto virtual_numbers = GetDeviceAddress<T1>(inputs, 0);
    auto uint_crd = GetDeviceAddress<T2>(inputs, 1);
    auto v_info = GetDeviceAddress<T>(inputs, 2);
    auto frc = GetDeviceAddress<T>(inputs, 3);

    v2ForceRedistribute(atom_numbers, virtual_numbers, v_info, uint_crd, frc,
                        reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(sizeof(T1));
    input_size_list_.push_back(ele_uint_crd * sizeof(T2));
    input_size_list_.push_back(ele_v_info * sizeof(T));
    input_size_list_.push_back(ele_frc * sizeof(T));

    output_size_list_.push_back(sizeof(T));
  }

 private:
  size_t ele_uint_crd = 1;
  size_t ele_v_info = 1;
  size_t ele_virtual_numbers = 1;
  size_t ele_frc = 1;

  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
  // int virtual_numbers;
  int atom_numbers;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPONG_VATOM_V2_FORCE_REDISTRIBUTE_KERNEL_H_
